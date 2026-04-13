"""Check if conv_state/temporal_state slot 0/1/2 stay consistent AFTER decode step."""
import torch
from nanovllm import LLM, SamplingParams


def main():
    llm = LLM("/root/zjh/huggingface/Qwen3.5-35B-A3B",
              tensor_parallel_size=4, max_model_len=2048, max_num_seqs=32,
              enforce_eager=True)
    model = llm.model_runner.model

    from nanovllm.layers.gdn import GDNAttention

    target = model.model.layers[0].linear_attn
    assert isinstance(target, GDNAttention)

    orig_forward = target.forward
    call_count = [0]

    def patched_forward(self, hidden_states):
        call_count[0] += 1
        n = call_count[0]

        # Log state BEFORE forward
        cs_before = self.conv_state[:3].clone().detach().cpu()
        ts_before = self.temporal_state[:3].clone().detach().cpu()
        print(f"[call {n}] BEFORE: conv_state slot0==slot1: {torch.allclose(cs_before[0], cs_before[1])}, slot0==slot2: {torch.allclose(cs_before[0], cs_before[2])}", flush=True)
        print(f"[call {n}] BEFORE: temp_state slot0==slot1: {torch.allclose(ts_before[0], ts_before[1])}, slot0==slot2: {torch.allclose(ts_before[0], ts_before[2])}", flush=True)

        out = orig_forward(hidden_states)

        cs_after = self.conv_state[:3].clone().detach().cpu()
        ts_after = self.temporal_state[:3].clone().detach().cpu()
        print(f"[call {n}] AFTER: conv_state slot0==slot1: {torch.allclose(cs_after[0], cs_after[1])}, slot0==slot2: {torch.allclose(cs_after[0], cs_after[2])}", flush=True)
        print(f"[call {n}] AFTER: temp_state slot0==slot1: {torch.allclose(ts_after[0], ts_after[1])}, slot0==slot2: {torch.allclose(ts_after[0], ts_after[2])}", flush=True)

        # Also check: out should have 3 identical rows
        if out.shape[0] == 3:
            d01 = (out[0] - out[1]).abs().max().item()
            d02 = (out[0] - out[2]).abs().max().item()
            print(f"[call {n}] OUT: shape={tuple(out.shape)} s0-s1={d01:.6e} s0-s2={d02:.6e}", flush=True)
        elif out.shape[0] == 21:
            d01 = (out[:7] - out[7:14]).abs().max().item()
            d02 = (out[:7] - out[14:21]).abs().max().item()
            print(f"[call {n}] OUT: shape={tuple(out.shape)} (prefill) s0-s1={d01:.6e} s0-s2={d02:.6e}", flush=True)
        return out

    import types
    target.forward = types.MethodType(patched_forward, target)

    sp = SamplingParams(temperature=0.0, max_tokens=3)
    llm.generate(["What is the capital of France?"] * 3, sp, use_tqdm=False)


if __name__ == "__main__":
    main()
