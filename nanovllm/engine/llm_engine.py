import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        # Honor model's generation_config.eos_token_id (may be a list, e.g. Qwen3.5
        # has both <|im_end|> and <|endoftext|>). Fall back to tokenizer's single eos.
        eos_ids: set[int] = set()
        try:
            from transformers import GenerationConfig
            gen_cfg = GenerationConfig.from_pretrained(config.model)
            cfg_eos = gen_cfg.eos_token_id
            if isinstance(cfg_eos, (list, tuple)):
                eos_ids.update(int(x) for x in cfg_eos)
            elif cfg_eos is not None:
                eos_ids.add(int(cfg_eos))
        except Exception:
            pass
        if self.tokenizer.eos_token_id is not None:
            eos_ids.add(int(self.tokenizer.eos_token_id))
        config.eos = eos_ids if eos_ids else {-1}
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        finished = []
        num_tokens = 0

        # Phase 1: Decode all running seqs that have completed prefill
        decode_seqs = self.scheduler.schedule_decode()
        if decode_seqs:
            token_ids = self.model_runner.call("run", decode_seqs, False)
            self.scheduler.postprocess_decode(decode_seqs, token_ids)
            for seq in decode_seqs:
                if seq.is_finished:
                    finished.append((seq.seq_id, seq.completion_token_ids, seq))
            num_tokens = -len(decode_seqs)

        # Phase 2: Prefill new/partial seqs (slots freed by finished decode seqs are now available)
        prefill_seqs = self.scheduler.schedule_prefill()
        if prefill_seqs:
            token_ids = self.model_runner.call("run", prefill_seqs, True)
            self.scheduler.postprocess_prefill(prefill_seqs, token_ids)
            for seq in prefill_seqs:
                if seq.is_finished:
                    finished.append((seq.seq_id, seq.completion_token_ids, seq))
            num_tokens = sum(getattr(seq, '_prefill_chunk_size', len(seq)) for seq in prefill_seqs)

        return finished, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            finished, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids, seq in finished:
                ttft = seq.first_token_time - seq.created_time
                n_completion = seq.num_completion_tokens
                tpot = (seq.finished_time - seq.first_token_time) / max(n_completion - 1, 1)
                latency = seq.finished_time - seq.created_time
                # ITL: inter-token latencies (time between consecutive tokens)
                ts = seq.token_timestamps
                itl = [ts[i] - ts[i-1] for i in range(1, len(ts))] if len(ts) > 1 else []
                outputs[seq_id] = {
                    "token_ids": token_ids,
                    "metrics": {
                        "ttft": ttft,
                        "tpot": tpot,
                        "latency": latency,
                        "itl": itl,
                        "itl_mean": sum(itl) / len(itl) if itl else 0.0,
                        "prompt_tokens": seq.num_prompt_tokens,
                        "completion_tokens": n_completion,
                    },
                }
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        for o in outputs:
            o["text"] = self.tokenizer.decode(o["token_ids"])
        if use_tqdm:
            pbar.close()
        return outputs
