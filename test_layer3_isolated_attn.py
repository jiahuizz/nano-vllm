"""Run nano-vllm's Qwen3_5MoeAttention in isolation (TP=1, single process).

Loads layer 3 weights from checkpoint, runs forward on HF's input, compares.
"""
import os, torch
import torch.distributed as dist

MODEL = "/root/zjh/huggingface/Qwen3.5-35B-A3B"
HF_DUMP = "/tmp/hf_layer3_v2.pt"


def main():
    # 1) Init dist with 1 process so nano-vllm's modules can call dist.get_world_size()
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group(backend="nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.bfloat16)

    # 2) Load HF dump
    hf = torch.load(HF_DUMP, weights_only=False)
    hf_input = hf["00_input"]  # [1, 7, 2048] fp32
    if hf_input.dim() == 3:
        hf_input = hf_input[0]  # [7, 2048]
    hf_input_bf16 = hf_input.to(torch.bfloat16).cuda()
    print(f"hf_input shape: {hf_input.shape}, max_abs: {hf_input.abs().max():.4f}")

    # 3) Build nano-vllm Qwen3_5MoeAttention for layer 3
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(MODEL).text_config
    print(f"num_attention_heads={cfg.num_attention_heads}, num_kv_heads={cfg.num_key_value_heads}, head_dim={getattr(cfg, 'head_dim', 256)}")

    from nanovllm.models.qwen3_5_moe import Qwen3_5MoeAttention
    sa = Qwen3_5MoeAttention(
        hidden_size=cfg.hidden_size,
        num_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=getattr(cfg, 'head_dim', 256),
        max_position=cfg.max_position_embeddings,
        rms_norm_eps=cfg.rms_norm_eps,
        rope_theta=getattr(cfg, 'rope_theta', 10000000),
        partial_rotary_factor=getattr(cfg, 'partial_rotary_factor', 0.25),
    ).cuda()

    # 4) Load layer 3 weights from checkpoint
    from safetensors import safe_open
    from glob import glob
    L = 3
    PREFIX = "model.language_model"  # multimodal model: language_model namespace
    weight_keys = {
        f"{PREFIX}.layers.{L}.self_attn.q_proj.weight": ("q_proj", "weight"),
        f"{PREFIX}.layers.{L}.self_attn.k_proj.weight": ("k_proj", "weight"),
        f"{PREFIX}.layers.{L}.self_attn.v_proj.weight": ("v_proj", "weight"),
        f"{PREFIX}.layers.{L}.self_attn.o_proj.weight": ("o_proj", "weight"),
        f"{PREFIX}.layers.{L}.self_attn.q_norm.weight": ("q_norm", "weight"),
        f"{PREFIX}.layers.{L}.self_attn.k_norm.weight": ("k_norm", "weight"),
    }
    found = {}
    for f in sorted(glob(os.path.join(MODEL, "*.safetensors"))):
        with safe_open(f, "pt", "cpu") as sf:
            for k in sf.keys():
                if k in weight_keys:
                    found[k] = sf.get_tensor(k)
    print(f"Found {len(found)}/{len(weight_keys)} weights")
    for k, (mod, w) in weight_keys.items():
        if k not in found:
            print(f"MISSING: {k}")
            continue
        target_param = getattr(getattr(sa, mod), w)
        loaded = found[k]
        print(f"  {k}: ckpt={tuple(loaded.shape)} dst={tuple(target_param.shape)}")
        if loaded.shape != target_param.shape:
            # ColumnParallelLinear at TP=1 should match
            print(f"    SHAPE MISMATCH")
            continue
        target_param.data.copy_(loaded.cuda())

    # 5) We need to set up context for nano's Attention layer (it reads from get_context)
    from nanovllm.utils.context import set_context, reset_context
    cu = torch.tensor([0, 7], dtype=torch.int32, device="cuda")
    slot = torch.tensor([0]*7, dtype=torch.int32, device="cuda")
    set_context(True, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=7, max_seqlen_k=7,
                slot_mapping=slot, context_lens=None, block_tables=None,
                gdn_state_indices=None, seq_lens=[7])

    # 6) Forward — first run on HF's exact input (sanity check)
    positions = torch.arange(7, dtype=torch.int64, device="cuda")
    with torch.no_grad():
        nano_out_clean = sa(positions, hf_input_bf16)
    reset_context()

    hf_out = hf["10_o_after_oproj"]
    if hf_out.dim() == 3:
        hf_out = hf_out[0]
    hf_out = hf_out.cuda()
    print(f"\n=== Test 1: HF input → nano output (sanity) ===")
    diff_clean = (hf_out - nano_out_clean.float()).abs().max().item()
    print(f"max_diff: {diff_clean:.6f}")

    # 7) Now perturb input by ~0.016 (matching layer 2 output drift in TP=4 run)
    # Use deterministic noise matching the structure (small-scale randn)
    torch.manual_seed(42)
    noise = (torch.randn_like(hf_input_bf16) * 0.005).to(torch.bfloat16)
    print(f"\nnoise max_abs: {noise.abs().max().item():.6f}")
    perturbed_input = hf_input_bf16 + noise

    set_context(True, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=7, max_seqlen_k=7,
                slot_mapping=slot, context_lens=None, block_tables=None,
                gdn_state_indices=None, seq_lens=[7])
    with torch.no_grad():
        nano_out_perturbed = sa(positions, perturbed_input)
    reset_context()

    print(f"\n=== Test 2: HF input + noise(~0.016) → nano output ===")
    input_drift = (perturbed_input.float() - hf_input_bf16.float()).abs().max().item()
    output_drift = (nano_out_perturbed.float() - nano_out_clean.float()).abs().max().item()
    output_vs_hf = (nano_out_perturbed.float() - hf_out).abs().max().item()
    print(f"input drift  : {input_drift:.6f}")
    print(f"output drift : {output_drift:.6f}  (= 'amplification' of input drift)")
    print(f"output vs HF : {output_vs_hf:.6f}")
    print(f"amplification ratio: {output_drift / max(input_drift, 1e-9):.1f}x")

    # 8) Test 3: try larger noise (0.016 directly)
    noise2 = (torch.randn_like(hf_input_bf16) * 0.016).to(torch.bfloat16)
    perturbed_input2 = hf_input_bf16 + noise2
    set_context(True, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=7, max_seqlen_k=7,
                slot_mapping=slot, context_lens=None, block_tables=None,
                gdn_state_indices=None, seq_lens=[7])
    with torch.no_grad():
        nano_out_p2 = sa(positions, perturbed_input2)
    reset_context()
    in_d2 = (perturbed_input2.float() - hf_input_bf16.float()).abs().max().item()
    out_d2 = (nano_out_p2.float() - nano_out_clean.float()).abs().max().item()
    print(f"\n=== Test 3: HF input + noise(0.016 max) → nano output ===")
    print(f"input drift  : {in_d2:.6f}")
    print(f"output drift : {out_d2:.6f}")
    print(f"amplification ratio: {out_d2 / max(in_d2, 1e-9):.1f}x")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
