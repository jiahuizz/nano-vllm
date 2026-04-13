"""Profile nano-vllm decode steps with torch.profiler.

Goal: find where decode time goes per step. Workload matches bench_compare
(ShareGPT-like): 64 prompts of various lengths, batch_size 64, run prefill
then capture profiler around N decode steps.
"""
import json
import time
import torch
from random import randint, seed
from nanovllm import LLM, SamplingParams


def main():
    model = "/root/zjh/huggingface/Qwen3.5-35B-A3B"
    seed(42)
    num_seqs = 64

    # Fixed workload: 64 random prompts of varied lengths (matches bench_compare)
    prompts = [
        [randint(0, 10000) for _ in range(randint(50, 256))]
        for _ in range(num_seqs)
    ]
    # Each seq generates 30 tokens — enough to amortize prefill, profiler
    # output stays manageable (~tens of MB)
    sampling_params = [
        SamplingParams(temperature=1.0, ignore_eos=True, max_tokens=30)
        for _ in range(num_seqs)
    ]

    llm = LLM(model, tensor_parallel_size=4, max_model_len=2048, max_num_seqs=64)

    # NOTE: LLM init already runs warmup_model() and capture_cudagraph(), so
    # CUDA kernels and graphs are already JIT-compiled. Don't do an extra
    # warmup generate() call — back-to-back generate() trips a known
    # block_manager assert in nano-vllm. One generate() call is enough.

    print("\nProfiler pass: prefill + decode under torch.profiler...", flush=True)
    prof_sp = sampling_params  # use the original max_tokens=200

    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, prof_sp, use_tqdm=False)
        elapsed = time.perf_counter() - t0

    total_completion = sum(o["metrics"]["completion_tokens"] for o in outputs)
    print(f"\nElapsed: {elapsed:.2f}s, total_completion: {total_completion} tokens, "
          f"throughput: {total_completion/elapsed:.1f} tok/s")

    # Kernel breakdown
    print("\n=== Top 30 GPU kernels by self CUDA time ===")
    kavg = prof.key_averages()
    sorted_kavg = sorted(kavg, key=lambda k: k.self_device_time_total, reverse=True)
    print(f"{'name':<60}{'count':>10}{'cuda_total_ms':>16}{'cuda_avg_us':>16}")
    print("-" * 102)
    total_cuda_ms = sum(k.self_device_time_total for k in sorted_kavg) / 1000
    for k in sorted_kavg[:30]:
        cuda_ms = k.self_device_time_total / 1000
        cuda_avg_us = (k.self_device_time_total / max(k.count, 1))
        pct = 100 * cuda_ms / max(total_cuda_ms, 1e-9)
        print(f"{k.key[:58]:<60}{k.count:>10}{cuda_ms:>14.2f}({pct:4.1f}%){cuda_avg_us:>14.1f}")

    print(f"\nTotal CUDA time across all kernels: {total_cuda_ms:.1f} ms")
    print(f"Total CPU time across all kernels: "
          f"{sum(k.self_cpu_time_total for k in sorted_kavg) / 1000:.1f} ms")

    # Save full table to json for offline diff
    out = {
        "elapsed": elapsed,
        "total_completion": total_completion,
        "throughput": total_completion / elapsed,
        "total_cuda_ms": total_cuda_ms,
        "total_cpu_ms": sum(k.self_cpu_time_total for k in sorted_kavg) / 1000,
        "kernels": [
            {
                "name": k.key,
                "count": k.count,
                "cuda_total_us": k.self_device_time_total,
                "cpu_total_us": k.self_cpu_time_total,
            }
            for k in sorted_kavg[:100]
        ],
    }
    with open("/tmp/profile_nano.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved /tmp/profile_nano.json")


if __name__ == "__main__":
    main()
