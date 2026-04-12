"""Correctness test for GDN prefill with FLA Triton kernels."""
from nanovllm import LLM, SamplingParams
from random import randint, seed


def main():
    llm = LLM("/root/zjh/huggingface/Qwen3.5-35B-A3B",
              tensor_parallel_size=4, max_model_len=2048, max_num_seqs=32)

    # Test 1: consistency (3 runs of same prompt)
    print("=== Test 1: Consistency (temp=0.01) ===", flush=True)
    sp = SamplingParams(temperature=0.01, max_tokens=80)
    outs = llm.generate(["What is 2+3?"] * 3, sp, use_tqdm=False)
    for i, o in enumerate(outs):
        print(f"Run {i}: {repr(o['text'][:120])}")
    print(f"Consistent: {outs[0]['text'] == outs[1]['text'] == outs[2]['text']}")

    # Test 2: quality check
    print("\n=== Test 2: Quality ===", flush=True)
    sp2 = SamplingParams(temperature=0.01, max_tokens=60)
    for p in ["What is the capital of France?", "Explain gravity in one sentence."]:
        o = llm.generate([p], sp2, use_tqdm=False)[0]
        t = o["text"].replace("\n", " ")[:200]
        print(f"Q: {p}")
        print(f"A: {t}")
        print()

    # Test 3: 32-seq batch
    print("=== Test 3: 32-seq batch ===", flush=True)
    seed(42)
    ps = [[randint(0, 10000) for _ in range(randint(64, 256))] for _ in range(32)]
    ss = [SamplingParams(temperature=1.0, ignore_eos=True, max_tokens=randint(64, 256))
          for _ in range(32)]
    os3 = llm.generate(ps, ss, use_tqdm=False)
    print(f"Completed: {len(os3)}/32")
    print(f"Total tokens: {sum(o['metrics']['completion_tokens'] for o in os3)}")


if __name__ == "__main__":
    main()
