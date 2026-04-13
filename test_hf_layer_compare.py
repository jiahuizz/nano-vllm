"""Layer-by-layer comparison: nano-vllm batch vs HF batch.

Phase 1: Run HF with 3 identical prompts (batch), capture hidden states per layer.
Phase 2: Run nano-vllm with same 3 prompts (batch), hook each layer, compare.
Find the first layer where nano-vllm diverges from HF.
"""
import os
import torch

MODEL = "/root/zjh/huggingface/Qwen3.5-35B-A3B"
PROMPT = "What is the capital of France?"
N_COPIES = 3
HF_DUMP = "/tmp/hf_layer_hidden.pt"


def run_hf():
    """Run HF with 3 same prompts, dump hidden states per layer."""
    print("=== Phase 1: HF Reference ===", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("Loading HF model (device_map=auto)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    model.eval()

    inputs = tok([PROMPT] * N_COPIES, return_tensors="pt", padding=True).to(model.device)
    prompt_len = inputs.input_ids.shape[1]
    print(f"Input shape: {inputs.input_ids.shape}, prompt_len={prompt_len}", flush=True)

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    # hidden_states is tuple of (num_layers+1) tensors, each [batch, seq, hidden]
    # hidden_states[0] = embedding output
    # hidden_states[i] for i=1..num_layers = output of layer i-1
    hidden_states = outputs.hidden_states
    print(f"Num hidden states: {len(hidden_states)}, shape of each: {hidden_states[0].shape}", flush=True)

    # Save to file, bring all to CPU
    dump = {
        "prompt_len": prompt_len,
        "num_layers": len(hidden_states) - 1,  # excluding embedding
        "hidden_states": [h.cpu() for h in hidden_states],
        "logits": outputs.logits.cpu() if hasattr(outputs, "logits") else None,
    }
    torch.save(dump, HF_DUMP)
    print(f"Saved {len(hidden_states)} hidden states to {HF_DUMP}", flush=True)

    del model
    torch.cuda.empty_cache()
    import gc; gc.collect()


def run_nanovllm():
    """Run nano-vllm with 3 same prompts, hook each layer, compare with HF."""
    print("\n=== Phase 2: nano-vllm + Layer-by-layer Compare ===", flush=True)

    assert os.path.exists(HF_DUMP), f"Run Phase 1 first (missing {HF_DUMP})"
    hf_dump = torch.load(HF_DUMP, map_location="cpu", weights_only=False)
    hf_hidden = hf_dump["hidden_states"]  # list of [3, L, hidden]
    prompt_len = hf_dump["prompt_len"]
    print(f"Loaded HF dump: {len(hf_hidden)} layer outputs, prompt_len={prompt_len}", flush=True)

    from nanovllm import LLM, SamplingParams
    llm = LLM(MODEL, tensor_parallel_size=4, max_model_len=2048, max_num_seqs=32)
    model = llm.model_runner.model

    # Capture output of each layer
    nano_outputs = {}

    def make_hook(idx):
        def hook(module, inputs, output):
            x = output[0] if isinstance(output, tuple) else output
            if isinstance(x, torch.Tensor):
                nano_outputs[idx] = x.detach().cpu()
        return hook

    # Hook embed_tokens, each layer, and final norm
    hooks = []
    hooks.append(model.model.embed_tokens.register_forward_hook(make_hook(("embed", 0))))
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_hook(make_hook(("layer", i))))
    hooks.append(model.model.norm.register_forward_hook(make_hook(("norm", 0))))

    # Run prefill with 3 identical prompts
    sp = SamplingParams(temperature=0.0, max_tokens=1)
    llm.generate([PROMPT] * N_COPIES, sp, use_tqdm=False)

    for h in hooks:
        h.remove()

    print(f"\nCaptured {len(nano_outputs)} layer outputs from nano-vllm", flush=True)

    # Compare embedding
    print("\n--- Layer-by-layer comparison (max_diff from HF) ---", flush=True)
    L = prompt_len
    num_layers = hf_dump["num_layers"]

    # HF hidden_states[0] is embedding, [1..num_layers] are per-layer outputs
    # nano embed: [3*L, hidden] concatenated
    # nano layer[i]: [3*L, hidden] concatenated

    def compare(hf_h, nano_h, name):
        # hf_h: [3, L, hidden], nano_h: [3*L, hidden] or [3, L, hidden]
        if nano_h is None:
            print(f"  {name}: <not captured>")
            return
        slices = []
        if nano_h.dim() == 3:
            slices = [nano_h[i].float() for i in range(3)]
        else:
            slices = [nano_h[i*L:(i+1)*L].float() for i in range(3)]
        # Diff from HF
        diffs_hf = [(hf_h[i].float() - slices[i]).abs().max().item() for i in range(3)]
        # Internal diff (seq0 vs seq1, seq0 vs seq2)
        d_01 = (slices[0] - slices[1]).abs().max().item()
        d_02 = (slices[0] - slices[2]).abs().max().item()
        max_hf = max(diffs_hf)
        tag = "OK" if max_hf < 0.5 else "DIFFHF"
        internal = "SAME" if max(d_01, d_02) < 1e-5 else "INTERNAL_DIFF"
        print(f"  [{tag}][{internal}] {name}: vs_hf=[{diffs_hf[0]:.4f},{diffs_hf[1]:.4f},{diffs_hf[2]:.4f}]  s0-s1={d_01:.6f} s0-s2={d_02:.6f}")

    # Embedding
    if ("embed", 0) in nano_outputs:
        compare(hf_hidden[0], nano_outputs[("embed", 0)], "embed")

    # Layers
    for i in range(num_layers):
        key = ("layer", i)
        if key not in nano_outputs:
            continue
        compare(hf_hidden[i + 1], nano_outputs[key], f"layer[{i}]")

    print("\n(Note: HF uses BF16+device_map, nano-vllm uses BF16+TP=4. Diff <0.5 is likely OK; larger diffs indicate bug.)")


def main():
    if not os.path.exists(HF_DUMP):
        run_hf()
    run_nanovllm()


if __name__ == "__main__":
    main()
