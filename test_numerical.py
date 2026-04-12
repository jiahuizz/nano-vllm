"""Numerical correctness test: compare nano-vllm greedy output with HF reference.

Two phases:
1. HF: load model with device_map='auto', greedy generate, save token ids
2. nano-vllm: load model with TP=4, greedy generate (temp=0), compare token ids
"""
import torch
import json
import os

MODEL_PATH = "/root/zjh/huggingface/Qwen3.5-35B-A3B"
PROMPTS = [
    "What is the capital of France?",
    "Explain gravity in one sentence.",
    "The sum of 2 and 3 is",
    "Hello, how are you today?",
    "Write a Python function to sort a list.",
]
MAX_NEW_TOKENS = 20
HF_CACHE = "/tmp/hf_reference.json"


def run_hf():
    """Run HF greedy generation as golden reference."""
    print("=== Phase 1: HF Reference ===", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print("Loading HF model (device_map=auto)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()

    results = []
    for prompt in PROMPTS:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids, max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,  # greedy
                temperature=None, top_p=None,
            )
        generated_ids = outputs[0][input_ids.shape[1]:].tolist()
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        results.append({
            "prompt": prompt,
            "prompt_tokens": input_ids[0].tolist(),
            "generated_ids": generated_ids,
            "generated_text": generated_text,
        })
        print(f"  Prompt: {prompt}")
        print(f"  Output ({len(generated_ids)} tok): {generated_text[:120]}")

    with open(HF_CACHE, "w") as f:
        json.dump(results, f)
    print(f"\nHF reference saved to {HF_CACHE}", flush=True)

    # Free HF model memory
    del model
    torch.cuda.empty_cache()
    import gc; gc.collect()
    return results


def run_nanovllm(hf_results):
    """Run nano-vllm greedy generation and compare with HF."""
    print("\n=== Phase 2: nano-vllm ===", flush=True)
    from nanovllm import LLM, SamplingParams

    llm = LLM(MODEL_PATH, tensor_parallel_size=4, max_model_len=2048, max_num_seqs=32)
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS, ignore_eos=True)

    print("\n=== Phase 3: Comparison ===", flush=True)
    all_pass = True
    for i, hf_ref in enumerate(hf_results):
        prompt = hf_ref["prompt"]
        prompt_ids = hf_ref["prompt_tokens"]
        hf_ids = hf_ref["generated_ids"]

        outs = llm.generate([prompt_ids], sp, use_tqdm=False)
        # token_ids is completion only (no prompt)
        nano_ids = outs[0]["token_ids"][:MAX_NEW_TOKENS]
        nano_text = outs[0]["text"]
        n_completion = outs[0]["metrics"]["completion_tokens"]
        print(f"  [debug] completion={n_completion}, nano_ids len={len(nano_ids)}")

        # Compare token by token
        match_count = 0
        first_diff = -1
        for j in range(min(len(hf_ids), len(nano_ids))):
            if hf_ids[j] == nano_ids[j]:
                match_count += 1
            elif first_diff == -1:
                first_diff = j

        total = min(len(hf_ids), len(nano_ids))
        match_pct = match_count / total * 100 if total > 0 else 0
        passed = match_count == total

        status = "PASS" if passed else "DIFF"
        print(f"\n[{status}] Prompt {i}: \"{prompt}\"")
        print(f"  HF  ({len(hf_ids)} tok): {hf_ref['generated_text'][:100]}")
        print(f"  Nano({len(nano_ids)} tok): {nano_text[:100]}")
        print(f"  Token match: {match_count}/{total} ({match_pct:.1f}%)")
        if first_diff >= 0:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(MODEL_PATH)
            print(f"  First diff at position {first_diff}: HF={hf_ids[first_diff]}({tok.decode([hf_ids[first_diff]])}) vs Nano={nano_ids[first_diff]}({tok.decode([nano_ids[first_diff]])})")
            all_pass = False

    print(f"\n{'='*50}")
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME DIFFS'}")
    print(f"{'='*50}")


def main():
    # Phase 1: HF reference (or load from cache)
    if os.path.exists(HF_CACHE):
        print(f"Loading cached HF reference from {HF_CACHE}", flush=True)
        with open(HF_CACHE) as f:
            hf_results = json.load(f)
        for r in hf_results:
            print(f"  Prompt: {r['prompt']}")
            print(f"  Output: {r['generated_text'][:100]}")
    else:
        hf_results = run_hf()

    # Phase 2+3: nano-vllm + comparison
    run_nanovllm(hf_results)


if __name__ == "__main__":
    main()
