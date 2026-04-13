"""Profile vLLM decode via VLLM_TORCH_PROFILER_DIR + llm.start_profile/stop_profile.

Workload matches profile_decode_nano.py.
"""
import os
import time
from random import randint, seed

# MUST be set before importing vllm
os.environ["VLLM_TORCH_PROFILER_DIR"] = "/tmp/vllm_traces"
os.makedirs("/tmp/vllm_traces", exist_ok=True)
# Clear old traces
for f in os.listdir("/tmp/vllm_traces"):
    os.remove(os.path.join("/tmp/vllm_traces", f))


def main():
    model = "/root/zjh/huggingface/Qwen3.5-35B-A3B"
    seed(42)
    num_seqs = 64

    prompts = [
        [randint(0, 10000) for _ in range(randint(50, 256))]
        for _ in range(num_seqs)
    ]

    from vllm import LLM, SamplingParams
    from vllm.config import ProfilerConfig
    sampling_params = [
        SamplingParams(temperature=1.0, ignore_eos=True, max_tokens=30)
        for _ in range(num_seqs)
    ]

    profiler_config = ProfilerConfig(
        profiler="torch",
        torch_profiler_dir="/tmp/vllm_traces",
        torch_profiler_with_stack=False,
    )
    llm = LLM(model, tensor_parallel_size=4, max_model_len=2048, max_num_seqs=64,
              trust_remote_code=True, disable_log_stats=True,
              profiler_config=profiler_config)

    # Warmup
    llm.generate([{"prompt_token_ids": [0]*10}],
                 SamplingParams(max_tokens=10, ignore_eos=True))

    vllm_prompts = [{"prompt_token_ids": p} for p in prompts]

    print("\nProfiler pass: prefill + decode under vLLM profiler...", flush=True)
    llm.start_profile()
    t0 = time.perf_counter()
    outputs = llm.generate(vllm_prompts, sampling_params)
    elapsed = time.perf_counter() - t0
    llm.stop_profile()

    total_completion = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"\nElapsed: {elapsed:.2f}s, total_completion: {total_completion} tokens, "
          f"throughput: {total_completion/elapsed:.1f} tok/s")
    print(f"Trace files in /tmp/vllm_traces:")
    for f in sorted(os.listdir("/tmp/vllm_traces")):
        size = os.path.getsize(f"/tmp/vllm_traces/{f}")
        print(f"  {f}: {size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
