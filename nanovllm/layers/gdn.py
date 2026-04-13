import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import os

from nanovllm.layers.linear import (
    MergedColumnParallelLinear, ColumnParallelLinear, RowParallelLinear, divide,
)
from nanovllm.layers.layernorm import RMSNormGated
from nanovllm.utils.context import get_context


_GDN_DEBUG_SEEN: set[tuple[int, int]] = set()


class GDNAttention(nn.Module):
    """Gated Delta Net linear attention layer.

    Replaces softmax attention in 30/40 layers of Qwen3.5.
    Maintains conv_state + temporal_state (like RNN) instead of KV cache.
    """

    def __init__(
        self,
        hidden_size: int,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        conv_kernel_size: int,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.tp_size = tp_size
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.key_dim = head_k_dim * num_k_heads
        self.value_dim = head_v_dim * num_v_heads
        self.conv_kernel_size = conv_kernel_size

        # Merged projection: Q, K, V, Z
        # Q: key_dim, K: key_dim, V: value_dim, Z: value_dim (gate)
        self.in_proj_qkvz = MergedColumnParallelLinear(
            hidden_size,
            [self.key_dim, self.key_dim, self.value_dim, self.value_dim],
            bias=False,
        )

        # B (update gate) and A (decay gate)
        self.in_proj_ba = MergedColumnParallelLinear(
            hidden_size,
            [num_v_heads, num_v_heads],
            bias=False,
        )

        # Causal 1D convolution (applied to concatenated Q, K, V)
        self.conv_dim = (self.key_dim * 2 + self.value_dim) // tp_size
        self.conv1d = nn.Module()
        self.conv1d.weight = nn.Parameter(
            torch.empty(self.conv_dim, conv_kernel_size))
        self.conv1d.weight.weight_loader = self._conv_weight_loader

        # State dynamics parameters
        self.A_log = nn.Parameter(torch.empty(num_v_heads // tp_size, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.ones(num_v_heads // tp_size))
        self.A_log.weight_loader = self._sharded_weight_loader
        self.dt_bias.weight_loader = self._sharded_weight_loader

        # Output
        self.norm = RMSNormGated(head_v_dim, eps=rms_norm_eps, add_unit_offset=False)
        self.out_proj = RowParallelLinear(self.value_dim, hidden_size, bias=False)

        # State buffers (assigned externally by model_runner)
        self.conv_state: torch.Tensor | None = None
        self.temporal_state: torch.Tensor | None = None

    def _conv_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        # Conv weight from checkpoint: [conv_dim, 1, kernel_size] or [conv_dim, kernel_size]
        # conv_dim = key_dim*2 + value_dim = Q + K + V (before TP)
        # Must shard each of Q, K, V separately, then concatenate
        if loaded_weight.dim() == 3:
            loaded_weight = loaded_weight.squeeze(1)
        tp_rank = dist.get_rank()
        tp_size = dist.get_world_size()
        # Split into Q, K, V portions
        full_key_dim = self.key_dim  # already = num_k_heads * head_k_dim (full, not sharded)
        full_value_dim = self.value_dim
        q_w, k_w, v_w = loaded_weight.split([full_key_dim, full_key_dim, full_value_dim], dim=0)
        # Shard each portion by TP
        q_shard = q_w.chunk(tp_size, dim=0)[tp_rank]
        k_shard = k_w.chunk(tp_size, dim=0)[tp_rank]
        v_shard = v_w.chunk(tp_size, dim=0)[tp_rank]
        param.data.copy_(torch.cat([q_shard, k_shard, v_shard], dim=0))

    def _sharded_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        tp_rank = dist.get_rank()
        shard_size = param.data.size(0)
        start_idx = tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param.data.copy_(loaded_weight)

    def _causal_conv1d_prefill(self, x: torch.Tensor, seq_lens: list[int], state_indices: torch.Tensor) -> torch.Tensor:
        """Apply causal 1D conv during prefill, save final conv state."""
        weight = self.conv1d.weight  # [conv_dim, kernel]
        k = self.conv_kernel_size
        output = torch.zeros_like(x)
        offset = 0
        for i, slen in enumerate(seq_lens):
            xi = x[offset:offset + slen]  # [slen, conv_dim]
            # Pad and convolve per sequence
            xi_t = xi.t().unsqueeze(0)  # [1, conv_dim, slen]
            xi_padded = F.pad(xi_t, (k - 1, 0))  # causal padding
            # Depthwise conv via manual matmul (grouped conv1d)
            out_t = F.conv1d(xi_padded, weight.unsqueeze(1), groups=self.conv_dim)
            output[offset:offset + slen] = out_t.squeeze(0).t()
            # Save last k-1 tokens as conv state
            if self.conv_state is not None:
                idx = state_indices[i]
                self.conv_state[idx, :, :] = xi[-k + 1:].t() if slen >= k - 1 else F.pad(xi.t(), (k - 1 - slen, 0))
            offset += slen
        return output

    def _causal_conv1d_decode(self, x: torch.Tensor, state_indices: torch.Tensor) -> torch.Tensor:
        """Single-step causal conv1d for decode. Uses vLLM's causal_conv1d_update kernel.
        Activation is fused ("silu") to match vLLM's fast path.
        """
        from nanovllm.layers.causal_conv1d_vllm import causal_conv1d_update
        # state_indices must be int32 (kernel requirement) AND must be the same
        # tensor identity across calls so CUDA graph replay sees fresh values.
        # We rely on the caller (gdn_state_indices in context) to already be int32.
        assert state_indices.dtype == torch.int32, (
            f"state_indices must be int32 for CUDA graph safety, got {state_indices.dtype}"
        )
        return causal_conv1d_update(
            x=x,
            conv_state=self.conv_state,
            weight=self.conv1d.weight,
            bias=None,
            activation="silu",  # fused silu inside kernel, vLLM style
            conv_state_indices=state_indices,
        )

    def _delta_rule_prefill(self, q, k, v, g, beta, seq_lens, state_indices):
        """Chunked delta rule for prefill using Triton FLA kernels.

        Replaces the HF PyTorch for-loop with a single batched call via cu_seqlens.
        """
        from nanovllm.layers.fla_ops import chunk_gated_delta_rule

        # GQA expansion: num_k_heads -> num_v_heads
        num_v_heads = v.size(1)
        n_rep = num_v_heads // q.size(1)
        if n_rep > 1:
            q = q.repeat_interleave(n_rep, dim=1)
            k = k.repeat_interleave(n_rep, dim=1)

        # Pack all sequences into [1, total_tokens, H, D] for FLA's cu_seqlens API
        q = q.unsqueeze(0)   # [1, T, H, K]
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        g = g.unsqueeze(0)   # [1, T, H]
        beta = beta.unsqueeze(0)

        # Build cu_seqlens: [0, len0, len0+len1, ...]
        cu_seqlens = torch.zeros(len(seq_lens) + 1, dtype=torch.int32, device=q.device)
        for i, slen in enumerate(seq_lens):
            cu_seqlens[i + 1] = cu_seqlens[i] + slen

        # Prepare initial_state: FLA expects [N, H, V, K], nano-vllm stores [N, H, K, V]
        init_state = None
        if self.temporal_state is not None:
            init_state = self.temporal_state[state_indices].transpose(-1, -2).contiguous()

        o, final_state = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=init_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=True,
        )

        # Write back final_state: FLA returns [N, H, V, K] -> transpose to [N, H, K, V]
        if self.temporal_state is not None and final_state is not None:
            self.temporal_state[state_indices] = final_state.transpose(-1, -2).to(self.temporal_state.dtype)

        return o.squeeze(0)  # [T, H, V]

    def _delta_rule_decode(self, q, k, v, g, beta, state_indices):
        """Single-step delta rule for decode. Fused Triton kernel."""
        # g and beta are not used — the fused kernel computes them internally from a, b, A_log, dt_bias
        # We pass a and b through the context (set in forward before calling this)
        from nanovllm.layers.gdn_kernels import fused_delta_rule_decode
        return fused_delta_rule_decode(
            q, k, v,
            self._decode_a, self._decode_b,
            self.A_log, self.dt_bias,
            self.temporal_state, state_indices,
            use_l2norm=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        context = get_context()
        num_tokens = hidden_states.size(0)

        # 1. Input projection
        qkvz = self.in_proj_qkvz(hidden_states)
        ba = self.in_proj_ba(hidden_states)

        qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
        z_size = self.value_dim // self.tp_size
        qkv, z = qkvz.split([qkv_size, z_size], dim=-1)
        # Match vLLM: explicitly make b and a contiguous (ba.chunk returns views)
        b, a = ba.chunk(2, dim=-1)
        b = b.contiguous()
        a = a.contiguous()

        # 2. Causal Conv1d on QKV
        state_indices = context.gdn_state_indices
        seq_lens = context.seq_lens
        if context.is_prefill:
            if seq_lens is None:
                # Warmup: treat entire batch as one sequence, no state
                seq_lens = [num_tokens]
                state_indices = torch.zeros(1, dtype=torch.int32, device=hidden_states.device)
            qkv = F.silu(self._causal_conv1d_prefill(qkv, seq_lens, state_indices))
            # For prefill, we need q/k/v split for the prefill path (FLA chunked delta rule)
            q_size = self.key_dim // self.tp_size
            k_size = self.key_dim // self.tp_size
            v_size = self.value_dim // self.tp_size
            q, k, v_flat = qkv.split([q_size, k_size, v_size], dim=-1)
            q = q.view(num_tokens, -1, self.head_k_dim)
            k = k.view(num_tokens, -1, self.head_k_dim)
            v = v_flat.view(num_tokens, -1, self.head_v_dim)
            z = z.view(num_tokens, -1, self.head_v_dim)
            # Prefill: compute g, beta explicitly for HF's chunk implementation
            g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
            beta = torch.sigmoid(b)
            attn_out = self._delta_rule_prefill(q, k, v, g, beta, seq_lens, state_indices)
        else:
            # Decode: pass mixed_qkv directly to vLLM's packed decode kernel.
            # Don't split into q/k/v — the kernel reads q/k/v via offsets from mixed_qkv.
            # (Silu is already fused into _causal_conv1d_decode via activation="silu".)
            mixed_qkv = self._causal_conv1d_decode(qkv, state_indices)
            z = z.view(num_tokens, -1, self.head_v_dim)

            from nanovllm.layers.fla_ops.fused_recurrent import (
                fused_recurrent_gated_delta_rule_packed_decode,
            )
            num_v_heads_tp = self.num_v_heads // self.tp_size
            # Pre-allocate output in [B, 1, HV, V] shape as required
            core_attn_out = torch.zeros(
                num_tokens, 1, num_v_heads_tp, self.head_v_dim,
                dtype=hidden_states.dtype, device=hidden_states.device,
            )
            # temporal_state is [max_seqs, HV, K, V] but kernel expects [..., HV, V, K]
            # We need to transpose last 2 dims. Use a contiguous buffer view.
            # NOTE: this transposes IN-PLACE via .transpose which returns a view;
            # the kernel requires stride(-1)==1 so we must materialize.
            ssm_state_view = self.temporal_state.transpose(-1, -2).contiguous()
            # state_indices already int32; passing it directly preserves tensor
            # identity for CUDA graph replay (a `.to(int32)` would allocate a new
            # tensor each call and make the captured pointer stale).
            assert state_indices.dtype == torch.int32
            fused_recurrent_gated_delta_rule_packed_decode(
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                scale=self.head_k_dim ** -0.5,
                initial_state=ssm_state_view,
                out=core_attn_out,
                ssm_state_indices=state_indices,
                use_qk_l2norm_in_kernel=True,
            )
            # Write back updated state: kernel updates ssm_state_view in place (ht=h0)
            self.temporal_state.copy_(ssm_state_view.transpose(-1, -2).contiguous())
            # core_attn_out has shape [B, 1, HV, V], squeeze to [B, HV, V]
            attn_out = core_attn_out.squeeze(1)

        # 6. Output: RMSNormGated(output, z) -> out_proj
        attn_out = attn_out.reshape(-1, self.head_v_dim)
        z_flat = z.reshape(-1, self.head_v_dim)
        attn_out = self.norm(attn_out, z_flat)
        attn_out = attn_out.view(num_tokens, -1)  # [N, value_dim/tp]

        debug_enabled = (
            os.getenv("NANOVLLM_DEBUG_GDN_TP") == "1"
            and context.is_prefill
            and context.seq_lens is not None
            and len(context.seq_lens) == 3
            and sum(context.seq_lens) == num_tokens
            and context.seq_lens[0] == context.seq_lens[1] == context.seq_lens[2]
        )

        if debug_enabled:
            rank = dist.get_rank()
            key = (rank, id(self))
            if key not in _GDN_DEBUG_SEEN:
                _GDN_DEBUG_SEEN.add(key)

                def _max_pair_diff(x: torch.Tensor) -> float:
                    l0, l1, l2 = context.seq_lens
                    s0 = x[:l0].float()
                    s1 = x[l0:l0 + l1].float()
                    s2 = x[l0 + l1:l0 + l1 + l2].float()
                    return max(
                        (s0 - s1).abs().max().item(),
                        (s0 - s2).abs().max().item(),
                    )

                local_out = F.linear(attn_out, self.out_proj.weight, None)
                local_diff = _max_pair_diff(local_out)
                with open(f"/tmp/gdn_tp_rank{rank}.log", "a", encoding="utf-8") as f:
                    f.write(
                        f"local_diff={local_diff:.8f} "
                        f"attn_out_diff={_max_pair_diff(attn_out):.8f}\n"
                    )

        output = self.out_proj(attn_out)

        if debug_enabled:
            rank = dist.get_rank()
            key = (rank, id(self), "reduced")
            if key not in _GDN_DEBUG_SEEN:
                _GDN_DEBUG_SEEN.add(key)

                def _max_pair_diff_out(x: torch.Tensor) -> float:
                    l0, l1, l2 = context.seq_lens
                    s0 = x[:l0].float()
                    s1 = x[l0:l0 + l1].float()
                    s2 = x[l0 + l1:l0 + l1 + l2].float()
                    return max(
                        (s0 - s1).abs().max().item(),
                        (s0 - s2).abs().max().item(),
                    )

                with open(f"/tmp/gdn_tp_rank{rank}.log", "a", encoding="utf-8") as f:
                    f.write(f"reduced_diff={_max_pair_diff_out(output):.8f}\n")
        return output
