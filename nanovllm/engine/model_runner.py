import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


def _create_model(config: Config):
    hf_config = config.hf_config
    text_config = config.hf_text_config
    model_type = getattr(text_config, 'model_type', '')
    if 'qwen3_5' in model_type:
        from nanovllm.models.qwen3_5_moe import Qwen3_5MoeForCausalLM
        return Qwen3_5MoeForCausalLM(text_config)
    else:
        from nanovllm.models.qwen3 import Qwen3ForCausalLM
        return Qwen3ForCausalLM(hf_config)


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(config.hf_text_config.torch_dtype)
        torch.set_default_device("cuda")
        self.model = _create_model(config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.num_gdn_layers = 0  # set before warmup; updated in allocate_kv_cache
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        # For MoE models, use smaller warmup to avoid OOM from gather ops
        if self.num_gdn_layers > 0:
            num_seqs = 1
            warmup_len = min(max_model_len, 256)
        else:
            warmup_len = max_model_len
            num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * warmup_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        text_config = config.hf_text_config
        dtype = getattr(text_config, 'torch_dtype', torch.bfloat16)
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype, torch.bfloat16)
        dtype_size = torch.tensor([], dtype=dtype).element_size()

        # Count attention layers (layers with k_cache/v_cache)
        num_attn_layers = sum(1 for m in self.model.modules()
                              if hasattr(m, "k_cache") and hasattr(m, "v_cache"))
        if num_attn_layers == 0:
            num_attn_layers = text_config.num_hidden_layers

        total_kv_heads = text_config.num_key_value_heads
        # When total_kv_heads < world_size, vLLM gives each rank ONE kv head
        # (replicated across ceil(world_size/total_kv_heads) ranks). This gives
        # correct GQA grouping inside flash_attn (group = num_q_local / 1).
        # The previous formula left num_kv_heads = total_kv_heads (= 2) which
        # made flash_attn split q heads across the wrong kv heads.
        num_kv_heads = max(1, total_kv_heads // self.world_size)
        head_dim = getattr(text_config, "head_dim", text_config.hidden_size // text_config.num_attention_heads)

        # Allocate GDN state if model has GDN layers
        self._allocate_gdn_state(config)

        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        block_bytes = 2 * num_attn_layers * self.block_size * num_kv_heads * head_dim * dtype_size
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0

        if num_attn_layers > 0:
            self.kv_cache = torch.empty(2, num_attn_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
            layer_id = 0
            for module in self.model.modules():
                if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                    module.k_cache = self.kv_cache[0, layer_id]
                    module.v_cache = self.kv_cache[1, layer_id]
                    layer_id += 1

    def _allocate_gdn_state(self, config: Config):
        """Pre-allocate conv_state and temporal_state for GDN layers."""
        from nanovllm.layers.gdn import GDNAttention
        gdn_layers = [m for m in self.model.modules() if isinstance(m, GDNAttention)]
        self.num_gdn_layers = len(gdn_layers)
        if self.num_gdn_layers == 0:
            return

        max_seqs = min(config.max_num_seqs, 128)  # limit GDN state memory
        # Allocate ONE EXTRA slot at the end for "graph padding": when CUDA graph
        # is captured at bs=N but actual decode bs is k<N, the [k:N] graph
        # positions still execute and need somewhere to write their (garbage)
        # GDN state — must NOT be a slot used by a real sequence.
        self.gdn_dummy_slot = max_seqs
        gdn_total_slots = max_seqs + 1
        for layer in gdn_layers:
            layer.conv_state = torch.zeros(
                gdn_total_slots, layer.conv_dim, layer.conv_kernel_size - 1,
                dtype=torch.bfloat16, device="cuda")
            layer.temporal_state = torch.zeros(
                gdn_total_slots,
                layer.num_v_heads // layer.tp_size,
                layer.head_k_dim, layer.head_v_dim,
                dtype=torch.float32, device="cuda")

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            # Chunked prefill: only process _prefill_chunk_size tokens (or all remaining)
            chunk_size = getattr(seq, '_prefill_chunk_size', seq.num_prompt_tokens - seq.num_cached_tokens)
            start_pos = seq.num_cached_tokens
            end_pos = start_pos + chunk_size

            input_ids.extend(seq[start_pos:end_pos])
            positions.extend(list(range(start_pos, end_pos)))
            seqlen_q = chunk_size
            seqlen_k = end_pos  # attention sees all tokens up to end_pos
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            # Slot mapping: only for the chunk being processed
            num_cached_blocks = seq.num_cached_tokens // self.block_size
            num_end_blocks = (end_pos + self.block_size - 1) // self.block_size
            for i in range(num_cached_blocks, num_end_blocks):
                start = seq.block_table[i] * self.block_size
                if i != num_end_blocks - 1:
                    end = start + self.block_size
                else:
                    remaining_in_block = end_pos - i * self.block_size
                    end = start + remaining_in_block
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache or chunked prefill
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        gdn_state_indices = None
        seq_lens = None
        if self.num_gdn_layers > 0:
            gdn_state_indices = torch.tensor([seq.gdn_state_idx for seq in seqs], dtype=torch.int32, device="cuda")
            seq_lens = [getattr(seq, '_prefill_chunk_size', seq.num_prompt_tokens - seq.num_cached_tokens) for seq in seqs]
            # Zero GDN state for first chunk of new sequences
            if any(seq.num_cached_tokens == 0 for seq in seqs):
                from nanovllm.layers.gdn import GDNAttention
                new_indices = torch.tensor([seq.gdn_state_idx for seq in seqs if seq.num_cached_tokens == 0],
                                          dtype=torch.int64, device="cuda")
                for m in self.model.modules():
                    if isinstance(m, GDNAttention):
                        if m.conv_state is not None:
                            m.conv_state[new_indices] = 0
                        if m.temporal_state is not None:
                            m.temporal_state[new_indices] = 0
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables, gdn_state_indices, seq_lens)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        gdn_state_indices = None
        if self.num_gdn_layers > 0:
            gdn_state_indices = torch.tensor([seq.gdn_state_idx for seq in seqs], dtype=torch.int32, device="cuda")
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables, gdn_state_indices=gdn_state_indices)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            if "gdn_state_indices" in graph_vars and context.gdn_state_indices is not None:
                # Pad [bs:graph_bs] to a dummy GDN slot so the captured graph's
                # tail positions write their garbage to a slot that no real seq
                # uses (otherwise they overwrite an active seq's GDN state).
                graph_vars["gdn_state_indices"].fill_(self.gdn_dummy_slot)
                graph_vars["gdn_state_indices"][:bs] = context.gdn_state_indices
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        text_config = config.hf_text_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, text_config.hidden_size)
        # Initialize gdn_state_indices to unique slots [0, 1, 2, ..., max_bs-1]
        # so that warmup/capture do not all write to slot 0 (race condition that
        # leaves slot 0 in a polluted state and degrades the captured graph).
        gdn_state_indices = (
            torch.arange(max_bs, dtype=torch.int32, device="cuda")
            if self.num_gdn_layers > 0 else None
        )
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs],
                       gdn_state_indices=gdn_state_indices[:bs] if gdn_state_indices is not None else None)
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
        if gdn_state_indices is not None:
            graph_vars["gdn_state_indices"] = gdn_state_indices
        self.graph_vars = graph_vars
