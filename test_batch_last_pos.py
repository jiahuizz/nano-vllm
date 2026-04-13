"""Find the batch size threshold for the 'last seq corrupts' bug."""
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    model_path = "/root/zjh/huggingface/Qwen3.5-35B-A3B"
    llm = LLM(model_path, tensor_parallel_size=4, max_model_len=2048, max_num_seqs=32)
    tok = AutoTokenizer.from_pretrained(model_path)
    sp = SamplingParams(temperature=0.0, max_tokens=200)

    base_q = "Why is the sky blue?"
    other_qs = [
        "What is the capital of France?",
        "Explain gravity in one sentence.",
        "Write hello world in Python.",
        "What is 2+2?",
        "Name a color.",
        "List a fruit.",
        "Pick a number.",
        "Say hi.",
    ]

    def chat(q):
        return tok.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)

    def is_bad(text):
        # Simple heuristic: look for repeated single-char tokens or "n. n. n."
        return ("* * * * *" in text or "* *  * *" in text or
                "3. 3. 3. 3." in text or "3.\n3.\n3.\n3." in text or
                ". . . . . . . ." in text)

    # Test batch sizes 1..8 with sky as the last
    for bs in [1, 2, 3, 4, 5, 6, 7, 8]:
        prompts = [chat(other_qs[i]) for i in range(bs - 1)] + [chat(base_q)]
        outs = llm.generate(prompts, sp, use_tqdm=False)
        last = outs[-1]["text"]
        bad = is_bad(last)
        tag = "BAD!! " if bad else "OK    "
        print(f"bs={bs}: {tag} sky tail: ...{last[-80:].replace(chr(10), ' | ')}")


if __name__ == "__main__":
    main()
