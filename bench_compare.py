"""Unified benchmark: nano-vllm vs vLLM with isolated environments.

Each engine runs in its own venv via subprocess. Run them SEPARATELY to avoid
GPU memory / page cache interference, then compare results.

Usage:
  # Step 1: Run each engine independently
  python bench_compare.py --engine nanovllm --output-json result_nano.json
  python bench_compare.py --engine vllm     --output-json result_vllm.json

  # Step 2: Compare
  python bench_compare.py --compare result_nano.json result_vllm.json
"""
import argparse
import json
import subprocess
import sys
import time
import numpy as np
from random import randint, seed

NANO_PYTHON = "/root/zjh/venv/bin/python3"
VLLM_PYTHON = "/root/zjh/venv-vllm/bin/python3"

# ---------------------------------------------------------------------------
# Workload generation
# ---------------------------------------------------------------------------

def load_sharegpt(dataset_path, num_seqs, model_path):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    with open(dataset_path) as f:
        data = json.load(f)
    data = [d for d in data if len(d.get("conversations", [])) >= 2]
    prompts, output_lens = [], []
    for item in data:
        if len(prompts) >= num_seqs:
            break
        human = item["conversations"][0]["value"]
        assistant = item["conversations"][1]["value"]
        prompt_ids = tokenizer.encode(human)
        completion_ids = tokenizer.encode(assistant)
        if len(prompt_ids) < 4 or len(completion_ids) < 4:
            continue
        if len(prompt_ids) > 1024 or len(completion_ids) > 1024:
            continue
        prompts.append(prompt_ids)
        output_lens.append(len(completion_ids))
    return prompts, output_lens


def make_random_workload(num_seqs, max_input_len, max_output_len, rng_seed=42):
    seed(rng_seed)
    prompts = [
        [randint(0, 10000) for _ in range(randint(max_input_len // 4, max_input_len))]
        for _ in range(num_seqs)
    ]
    output_lens = [randint(max_output_len // 4, max_output_len) for _ in range(num_seqs)]
    return prompts, output_lens


# ---------------------------------------------------------------------------
# Subprocess worker (inlined as string, runs in the target venv)
# ---------------------------------------------------------------------------

BENCH_WORKER = r'''
import json, sys, time

workload_path = sys.argv[1]
with open(workload_path) as f:
    workload = json.load(f)
engine = workload["engine"]
model_path = workload["model"]
prompt_token_ids = workload["prompts"]
output_lens = workload["output_lens"]
tp_size = workload["tp"]
enforce_eager = workload["enforce_eager"]

max_num_seqs = workload.get("max_num_seqs", 32)

if engine == "nanovllm":
    from nanovllm import LLM, SamplingParams
    llm = LLM(model_path, enforce_eager=enforce_eager, tensor_parallel_size=tp_size,
              max_model_len=2048, max_num_seqs=max_num_seqs)
    sampling_params = [SamplingParams(temperature=1.0, ignore_eos=True, max_tokens=ol) for ol in output_lens]
    llm.generate(["warmup"], SamplingParams(), use_tqdm=False)
    t0 = time.perf_counter()
    outputs = llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    elapsed = time.perf_counter() - t0
    total_completion = sum(o["metrics"]["completion_tokens"] for o in outputs)
    ttfts = [o["metrics"]["ttft"] for o in outputs]
    tpots = [o["metrics"]["tpot"] for o in outputs]
    latencies = [o["metrics"]["latency"] for o in outputs]
    result = {"elapsed": elapsed, "total_completion": total_completion,
              "ttfts": ttfts, "tpots": tpots, "latencies": latencies}

elif engine == "vllm":
    from vllm import LLM, SamplingParams
    llm = LLM(model_path, enforce_eager=enforce_eager, tensor_parallel_size=tp_size,
              max_model_len=2048, max_num_seqs=max_num_seqs, trust_remote_code=True,
              disable_log_stats=False)
    sampling_params = [SamplingParams(temperature=1.0, ignore_eos=True, max_tokens=ol) for ol in output_lens]
    prompts = [{"prompt_token_ids": p} for p in prompt_token_ids]
    llm.generate([{"prompt_token_ids": [0]*10}], SamplingParams(max_tokens=10, ignore_eos=True))
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.perf_counter() - t0
    total_completion = sum(len(o.outputs[0].token_ids) for o in outputs)
    ttfts, tpots, latencies = [], [], []
    for o in outputs:
        m = o.metrics
        n = len(o.outputs[0].token_ids)
        if m and m.first_token_latency:
            ttfts.append(m.first_token_latency)
        if m and m.first_token_ts and m.last_token_ts and n > 1:
            tpots.append((m.last_token_ts - m.first_token_ts) / (n - 1))
        if m and m.last_token_ts and m.queued_ts:
            latencies.append(m.last_token_ts - m.queued_ts)
    result = {"elapsed": elapsed, "total_completion": total_completion,
              "ttfts": ttfts or None, "tpots": tpots or None, "latencies": latencies or None}

import tempfile, os
result_path = workload_path.replace(".json", "_result.json")
with open(result_path, "w") as f:
    json.dump(result, f)
print("__RESULT_FILE__" + result_path)
'''


def run_engine(python_path, engine, workload_dict):
    import tempfile
    workload_dict["engine"] = engine
    # Write workload to temp file (avoids ARG_MAX for large workloads)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(workload_dict, f)
        workload_path = f.name

    print(f"  Starting {engine} ({python_path}) ...")
    proc = subprocess.run(
        [python_path, "-c", BENCH_WORKER, workload_path],
        capture_output=True, text=True, timeout=3600,
        cwd="/root/zjh/nano-vllm",
    )

    for line in proc.stdout.splitlines():
        if line.startswith("__RESULT_FILE__"):
            result_path = line[len("__RESULT_FILE__"):]
            with open(result_path) as f:
                return json.load(f)

    print(f"  ERROR: {engine} did not produce results")
    for line in proc.stderr.strip().splitlines()[-20:]:
        print(f"    {line}")
    return None


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def format_result(engine_name, r, total_prompt):
    throughput = r["total_completion"] / r["elapsed"]
    out = {
        "engine": engine_name,
        "elapsed": r["elapsed"],
        "total_prompt": total_prompt,
        "total_completion": r["total_completion"],
        "throughput": throughput,
    }
    if r.get("ttfts"):
        ttfts = np.array(r["ttfts"]) * 1000
        tpots = np.array(r["tpots"]) * 1000
        lats = np.array(r["latencies"])
        out.update({
            "ttft_mean": np.mean(ttfts), "ttft_p50": np.percentile(ttfts, 50), "ttft_p99": np.percentile(ttfts, 99),
            "tpot_mean": np.mean(tpots), "tpot_p50": np.percentile(tpots, 50), "tpot_p99": np.percentile(tpots, 99),
            "lat_mean": np.mean(lats), "lat_p50": np.percentile(lats, 50), "lat_p99": np.percentile(lats, 99),
        })
    return out


def print_result(r):
    print(f"\n{'='*55}")
    print(f"  {r['engine']}")
    print(f"{'='*55}")
    print(f"  Tokens     : {r['total_prompt']} prompt + {r['total_completion']} completion")
    print(f"  Time       : {r['elapsed']:.2f}s")
    print(f"  Throughput : {r['throughput']:.1f} tok/s")
    if r.get("ttft_mean") is not None:
        print(f"{'─'*55}")
        print(f"  {'Metric':<14} {'Mean':>10} {'P50':>10} {'P99':>10}")
        print(f"{'─'*55}")
        print(f"  {'TTFT (ms)':<14} {r['ttft_mean']:>10.1f} {r['ttft_p50']:>10.1f} {r['ttft_p99']:>10.1f}")
        print(f"  {'TPOT (ms)':<14} {r['tpot_mean']:>10.1f} {r['tpot_p50']:>10.1f} {r['tpot_p99']:>10.1f}")
        print(f"  {'Latency (s)':<14} {r['lat_mean']:>10.2f} {r['lat_p50']:>10.2f} {r['lat_p99']:>10.2f}")
    print(f"{'='*55}")


def print_comparison(results):
    if len(results) < 2:
        return
    print(f"\n{'='*55}")
    print(f"  Comparison")
    print(f"{'='*55}")
    for i, r in enumerate(results):
        print(f"  {r['engine']:<12} : {r['throughput']:>8.1f} tok/s  ({r['elapsed']:.2f}s)")
    base = results[0]
    for r in results[1:]:
        ratio = r["throughput"] / base["throughput"]
        print(f"  {r['engine']} / {base['engine']} throughput: {ratio:.2f}x")
    print(f"{'='*55}")


# ---------------------------------------------------------------------------
# Compare mode: load two JSON result files and compare
# ---------------------------------------------------------------------------

def compare_results(files):
    results = []
    for f in files:
        with open(f) as fp:
            results.append(json.load(fp))
    for r in results:
        print_result(r)
    print_comparison(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark nano-vllm vs vLLM (isolated envs)")
    parser.add_argument("--engine", choices=["nanovllm", "vllm"], default=None,
                        help="Run a single engine benchmark")
    parser.add_argument("--compare", nargs="+", metavar="JSON",
                        help="Compare result JSON files (no benchmark run)")
    parser.add_argument("--model", default="/root/zjh/huggingface/Qwen3.5-35B-A3B")
    parser.add_argument("--dataset", choices=["random", "sharegpt"], default="random")
    parser.add_argument("--dataset-path", default=None, help="Path to ShareGPT JSON")
    parser.add_argument("--num-seqs", type=int, default=32)
    parser.add_argument("--max-num-seqs", type=int, default=32, help="concurrent batch size")
    parser.add_argument("--max-input-len", type=int, default=256)
    parser.add_argument("--max-output-len", type=int, default=256)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--enforce-eager", action="store_true", default=False)
    parser.add_argument("--no-cuda-graph", dest="enforce_eager", action="store_true")
    parser.add_argument("--output-json", default=None, help="Save result to JSON")
    args = parser.parse_args()

    # Compare mode
    if args.compare:
        compare_results(args.compare)
        return

    if args.engine is None:
        parser.error("--engine is required (use --compare to compare existing results)")

    # Generate workload
    if args.dataset == "sharegpt":
        assert args.dataset_path, "--dataset-path required for sharegpt"
        prompt_token_ids, output_lens = load_sharegpt(args.dataset_path, args.num_seqs, args.model)
        print(f"Dataset: ShareGPT ({len(prompt_token_ids)} seqs loaded)")
    else:
        prompt_token_ids, output_lens = make_random_workload(
            args.num_seqs, args.max_input_len, args.max_output_len)
        print(f"Dataset: random ({len(prompt_token_ids)} seqs, input≤{args.max_input_len}, output≤{args.max_output_len})")

    total_in = sum(len(p) for p in prompt_token_ids)
    total_out = sum(output_lens)
    print(f"Model: {args.model}, TP={args.tp}, eager={args.enforce_eager}")
    print(f"Total: {total_in} prompt tok + {total_out} output tok (target)")

    workload = {
        "model": args.model,
        "prompts": prompt_token_ids,
        "output_lens": output_lens,
        "tp": args.tp,
        "enforce_eager": args.enforce_eager,
        "max_num_seqs": args.max_num_seqs,
    }

    python_path = NANO_PYTHON if args.engine == "nanovllm" else VLLM_PYTHON
    engine_label = "nano-vllm" if args.engine == "nanovllm" else "vLLM"

    r = run_engine(python_path, args.engine, workload)
    if r:
        result = format_result(engine_label, r, total_in)
        print_result(result)
        if args.output_json:
            with open(args.output_json, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\nResult saved to {args.output_json}")


if __name__ == "__main__":
    main()
