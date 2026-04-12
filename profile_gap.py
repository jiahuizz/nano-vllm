"""Profile performance gap between nano-vllm and vLLM.

Measures TTFT, TPOT, throughput with IDENTICAL methodology on both engines.
Must be run directly on GPU server (not via subprocess wrapper).

Usage:
  # Run each engine separately, then compare:
  python profile_gap.py --engine nanovllm --output nano.json
  python profile_gap.py --engine vllm --output vllm.json
  python profile_gap.py --compare nano.json vllm.json
"""
import argparse
import json
import time
import sys
import numpy as np
from random import randint, seed

# ─── Workload ───────────────────────────────────────────────────────
def make_workload(num_seqs, max_input_len, max_output_len, rng_seed=42):
    seed(rng_seed)
    prompts = [
        [randint(0, 10000) for _ in range(randint(max_input_len // 4, max_input_len))]
        for _ in range(num_seqs)
    ]
    output_lens = [randint(max_output_len // 4, max_output_len) for _ in range(num_seqs)]
    return prompts, output_lens


# ─── Profiling: nano-vllm ──────────────────────────────────────────
def profile_nanovllm(model, prompts, output_lens, tp, enforce_eager):
    from nanovllm import LLM, SamplingParams

    llm = LLM(model, enforce_eager=enforce_eager, tensor_parallel_size=tp,
              max_model_len=2048, max_num_seqs=32)
    sampling_params = [SamplingParams(temperature=1.0, ignore_eos=True, max_tokens=ol)
                       for ol in output_lens]

    # Warmup
    llm.generate(["warmup"], SamplingParams(), use_tqdm=False)

    # Benchmark
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    elapsed = time.perf_counter() - t0

    # Extract per-request metrics
    requests = []
    for o in outputs:
        m = o["metrics"]
        requests.append({
            "prompt_tokens": m["prompt_tokens"],
            "completion_tokens": m["completion_tokens"],
            "ttft": m["ttft"],
            "tpot": m["tpot"],
            "latency": m["latency"],
        })

    return elapsed, requests


# ─── Profiling: vLLM ──────────────────────────────────────────────
def profile_vllm(model, prompts, output_lens, tp, enforce_eager):
    from vllm import LLM, SamplingParams

    llm = LLM(model, enforce_eager=enforce_eager, tensor_parallel_size=tp,
              max_model_len=2048, max_num_seqs=32, trust_remote_code=True,
              disable_log_stats=False)
    sampling_params = [SamplingParams(temperature=1.0, ignore_eos=True, max_tokens=ol)
                       for ol in output_lens]
    vllm_prompts = [{"prompt_token_ids": p} for p in prompts]

    # Warmup
    llm.generate([{"prompt_token_ids": [0]*10}],
                 SamplingParams(max_tokens=10, ignore_eos=True))

    # Benchmark
    t0 = time.perf_counter()
    outputs = llm.generate(vllm_prompts, sampling_params)
    elapsed = time.perf_counter() - t0

    # Extract per-request metrics (vLLM 0.19.0 RequestStateStats)
    requests = []
    for o in outputs:
        n_completion = len(o.outputs[0].token_ids)
        n_prompt = len(o.prompt_token_ids)

        # vLLM metrics
        m = o.metrics
        ttft = m.first_token_latency if m and m.first_token_latency else None

        # TPOT from monotonic timestamps
        if m and m.first_token_ts and m.last_token_ts and n_completion > 1:
            tpot = (m.last_token_ts - m.first_token_ts) / (n_completion - 1)
        else:
            tpot = None

        # e2e latency
        if m and m.last_token_ts and m.queued_ts:
            latency = m.last_token_ts - m.queued_ts
        else:
            latency = None

        requests.append({
            "prompt_tokens": n_prompt,
            "completion_tokens": n_completion,
            "ttft": ttft,
            "tpot": tpot,
            "latency": latency,
            # Raw timestamps for debugging
            "_arrival_time": m.arrival_time if m else None,
            "_queued_ts": m.queued_ts if m else None,
            "_scheduled_ts": m.scheduled_ts if m else None,
            "_first_token_ts": m.first_token_ts if m else None,
            "_last_token_ts": m.last_token_ts if m else None,
            "_first_token_latency": m.first_token_latency if m else None,
        })

    return elapsed, requests


# ─── Analysis ──────────────────────────────────────────────────────
def analyze(engine_name, elapsed, requests, total_prompt):
    total_completion = sum(r["completion_tokens"] for r in requests)
    throughput = total_completion / elapsed

    ttfts = [r["ttft"] for r in requests if r["ttft"] is not None]
    tpots = [r["tpot"] for r in requests if r["tpot"] is not None]
    latencies = [r["latency"] for r in requests if r["latency"] is not None]

    result = {
        "engine": engine_name,
        "elapsed": elapsed,
        "total_prompt": total_prompt,
        "total_completion": total_completion,
        "throughput": throughput,
        "num_requests": len(requests),
    }

    if ttfts:
        ttfts_ms = np.array(ttfts) * 1000
        result["ttft_mean"] = float(np.mean(ttfts_ms))
        result["ttft_p50"] = float(np.percentile(ttfts_ms, 50))
        result["ttft_p99"] = float(np.percentile(ttfts_ms, 99))
        result["ttft_min"] = float(np.min(ttfts_ms))
        result["ttft_max"] = float(np.max(ttfts_ms))
        # Prefill throughput estimate: total prompts / max TTFT
        result["prefill_throughput"] = total_prompt / max(ttfts)

    if tpots:
        tpots_ms = np.array(tpots) * 1000
        result["tpot_mean"] = float(np.mean(tpots_ms))
        result["tpot_p50"] = float(np.percentile(tpots_ms, 50))
        result["tpot_p99"] = float(np.percentile(tpots_ms, 99))
        result["tpot_min"] = float(np.min(tpots_ms))
        result["tpot_max"] = float(np.max(tpots_ms))

    if latencies:
        lats = np.array(latencies)
        result["latency_mean"] = float(np.mean(lats))
        result["latency_p50"] = float(np.percentile(lats, 50))
        result["latency_p99"] = float(np.percentile(lats, 99))

    # Derived: decode phase time and throughput
    if ttfts:
        max_ttft = max(ttfts)
        decode_time = elapsed - max_ttft
        if decode_time > 0:
            result["decode_throughput"] = total_completion / decode_time
            result["decode_time"] = decode_time
            result["prefill_time"] = max_ttft

    result["requests"] = requests
    return result


def print_result(r):
    print(f"\n{'='*65}")
    print(f"  {r['engine']}  ({r['num_requests']} requests)")
    print(f"{'='*65}")
    print(f"  Tokens       : {r['total_prompt']} prompt + {r['total_completion']} completion")
    print(f"  Total time   : {r['elapsed']:.3f}s")
    print(f"  Throughput   : {r['throughput']:.1f} tok/s")
    if "prefill_time" in r:
        print(f"{'─'*65}")
        print(f"  Prefill time : {r['prefill_time']:.3f}s  → {r.get('prefill_throughput', 0):.1f} prompt tok/s")
        print(f"  Decode time  : {r['decode_time']:.3f}s  → {r.get('decode_throughput', 0):.1f} completion tok/s")
    print(f"{'─'*65}")
    print(f"  {'Metric':<14} {'Mean':>10} {'P50':>10} {'P99':>10} {'Min':>10} {'Max':>10}")
    print(f"{'─'*65}")
    for name, prefix in [("TTFT (ms)", "ttft"), ("TPOT (ms)", "tpot")]:
        if f"{prefix}_mean" in r:
            print(f"  {name:<14} {r[f'{prefix}_mean']:>10.2f} {r[f'{prefix}_p50']:>10.2f} "
                  f"{r[f'{prefix}_p99']:>10.2f} {r[f'{prefix}_min']:>10.2f} {r[f'{prefix}_max']:>10.2f}")
    if "latency_mean" in r:
        print(f"  {'Latency (s)':<14} {r['latency_mean']:>10.3f} {r['latency_p50']:>10.3f} {r['latency_p99']:>10.3f}")
    print(f"{'='*65}")


def compare(results):
    print(f"\n{'='*65}")
    print(f"  Performance Gap Analysis")
    print(f"{'='*65}")

    r0, r1 = results
    print(f"\n  {'Metric':<30} {r0['engine']:>15} {r1['engine']:>15} {'Ratio':>10}")
    print(f"  {'─'*70}")

    comparisons = [
        ("Throughput (tok/s)", "throughput", True),
        ("Prefill throughput (tok/s)", "prefill_throughput", True),
        ("Decode throughput (tok/s)", "decode_throughput", True),
        ("Prefill time (s)", "prefill_time", False),
        ("Decode time (s)", "decode_time", False),
        ("TTFT mean (ms)", "ttft_mean", False),
        ("TPOT mean (ms)", "tpot_mean", False),
        ("TPOT P99 (ms)", "tpot_p99", False),
    ]

    for label, key, higher_better in comparisons:
        v0 = r0.get(key)
        v1 = r1.get(key)
        if v0 is not None and v1 is not None:
            ratio = v1 / v0 if v0 != 0 else float('inf')
            fmt = ".1f" if "throughput" in key.lower() else (".3f" if "time" in key.lower() else ".2f")
            print(f"  {label:<30} {v0:>15{fmt}} {v1:>15{fmt}} {ratio:>9.2f}x")

    # Contribution analysis
    print(f"\n  Time Budget Breakdown:")
    for r in results:
        if "prefill_time" in r and "decode_time" in r:
            pf_pct = r["prefill_time"] / r["elapsed"] * 100
            dc_pct = r["decode_time"] / r["elapsed"] * 100
            print(f"    {r['engine']}: prefill {r['prefill_time']:.3f}s ({pf_pct:.1f}%) + "
                  f"decode {r['decode_time']:.3f}s ({dc_pct:.1f}%) = {r['elapsed']:.3f}s")

    # Gap attribution
    if "prefill_time" in r0 and "prefill_time" in r1:
        total_gap = r0["elapsed"] - r1["elapsed"]
        prefill_gap = r0.get("prefill_time", 0) - r1.get("prefill_time", 0)
        decode_gap = r0.get("decode_time", 0) - r1.get("decode_time", 0)
        if abs(total_gap) > 0.001:
            print(f"\n  Gap Attribution (nano-vllm is {total_gap:.3f}s slower):")
            print(f"    Prefill gap: {prefill_gap:.3f}s ({prefill_gap/total_gap*100:.1f}% of total gap)")
            print(f"    Decode gap:  {decode_gap:.3f}s ({decode_gap/total_gap*100:.1f}% of total gap)")

    print(f"\n{'='*65}")


def main():
    parser = argparse.ArgumentParser(description="Profile nano-vllm vs vLLM performance gap")
    parser.add_argument("--engine", choices=["nanovllm", "vllm"])
    parser.add_argument("--compare", nargs=2, metavar="JSON")
    parser.add_argument("--model", default="/root/zjh/huggingface/Qwen3.5-35B-A3B")
    parser.add_argument("--num-seqs", type=int, default=32)
    parser.add_argument("--max-input-len", type=int, default=256)
    parser.add_argument("--max-output-len", type=int, default=256)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--enforce-eager", action="store_true", default=False)
    parser.add_argument("--output", default=None, help="Save result JSON")
    args = parser.parse_args()

    if args.compare:
        results = []
        for f in args.compare:
            with open(f) as fp:
                r = json.load(fp)
            results.append(r)
        for r in results:
            r_print = {k: v for k, v in r.items() if k != "requests"}
            print_result(r_print)
        compare([{k: v for k, v in r.items() if k != "requests"} for r in results])
        return

    if not args.engine:
        parser.error("--engine required")

    prompts, output_lens = make_workload(args.num_seqs, args.max_input_len, args.max_output_len)
    total_prompt = sum(len(p) for p in prompts)
    total_output = sum(output_lens)

    print(f"Engine: {args.engine}")
    print(f"Model: {args.model}")
    print(f"Workload: {len(prompts)} seqs, {total_prompt} prompt tok, {total_output} target output tok")
    print(f"Config: TP={args.tp}, eager={args.enforce_eager}")

    if args.engine == "nanovllm":
        elapsed, requests = profile_nanovllm(args.model, prompts, output_lens, args.tp, args.enforce_eager)
        engine_name = "nano-vllm"
    else:
        elapsed, requests = profile_vllm(args.model, prompts, output_lens, args.tp, args.enforce_eager)
        engine_name = "vLLM"

    result = analyze(engine_name, elapsed, requests, total_prompt)
    r_print = {k: v for k, v in result.items() if k != "requests"}
    print_result(r_print)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
