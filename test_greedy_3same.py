"""Strict correctness test: 3 identical prompts, temperature=0, must produce identical text."""
from nanovllm import LLM, SamplingParams


def main():
    llm = LLM("/root/zjh/huggingface/Qwen3.5-35B-A3B",
              tensor_parallel_size=4, max_model_len=2048, max_num_seqs=32)
    sp = SamplingParams(temperature=0.0, max_tokens=80)

    print("=== 3 identical prompts, temp=0, max_tokens=80 ===")
    outs = llm.generate(["What is the capital of France?"] * 3, sp, use_tqdm=False)
    texts = [o["text"] for o in outs]
    for i, t in enumerate(texts):
        print(f"Run {i}: {repr(t[:120])}")
    print(f"\nAll identical: {texts[0] == texts[1] == texts[2]}")
    if texts[0] != texts[1]:
        # Find first diff position
        for i, (c0, c1) in enumerate(zip(texts[0], texts[1])):
            if c0 != c1:
                print(f"First diff at char {i}: {repr(texts[0][max(0,i-10):i+10])} vs {repr(texts[1][max(0,i-10):i+10])}")
                break


if __name__ == "__main__":
    main()
