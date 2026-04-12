import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.layers.linear import (
    MergedColumnParallelLinear, ColumnParallelLinear, RowParallelLinear, divide,
)
from nanovllm.layers.layernorm import RMSNormGated
from nanovllm.utils.context import get_context


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
        """Single-step causal conv1d for decode. Fused Triton kernel."""
        from nanovllm.layers.gdn_kernels import fused_conv1d_update
        return fused_conv1d_update(x, self.conv_state, self.conv1d.weight, state_indices, silu=False)

    def _delta_rule_prefill(self, q, k, v, g, beta, seq_lens, state_indices):
        """Chunked delta rule for prefill using HF-compatible implementation."""
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import torch_chunk_gated_delta_rule

        # GQA expansion: num_k_heads -> num_v_heads
        num_v_heads = v.size(1)
        n_rep = num_v_heads // q.size(1)
        if n_rep > 1:
            q = q.repeat_interleave(n_rep, dim=1)
            k = k.repeat_interleave(n_rep, dim=1)

        offset = 0
        outputs = []
        for i, slen in enumerate(seq_lens):
            # Reshape to [batch=1, seq_len, num_heads, head_dim] for HF format
            qi = q[offset:offset + slen].unsqueeze(0)
            ki = k[offset:offset + slen].unsqueeze(0)
            vi = v[offset:offset + slen].unsqueeze(0)
            gi = g[offset:offset + slen].unsqueeze(0).unsqueeze(-1)  # [1, seq, heads, 1] -> [1, seq, heads]
            # Actually g is already [seq, heads], just unsqueeze batch
            gi = g[offset:offset + slen].unsqueeze(0)
            bi = beta[offset:offset + slen].unsqueeze(0)

            idx = state_indices[i]
            init_state = self.temporal_state[idx].unsqueeze(0) if self.temporal_state is not None else None

            o, final_state = torch_chunk_gated_delta_rule(
                qi, ki, vi, gi, bi,
                initial_state=init_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            outputs.append(o.squeeze(0))
            if self.temporal_state is not None and final_state is not None:
                self.temporal_state[idx] = final_state.squeeze(0)
            offset += slen
        return torch.cat(outputs, dim=0)

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
        b, a = ba.chunk(2, dim=-1)

        # 2. Causal Conv1d on QKV
        state_indices = context.gdn_state_indices
        seq_lens = context.seq_lens
        if context.is_prefill:
            if seq_lens is None:
                # Warmup: treat entire batch as one sequence, no state
                seq_lens = [num_tokens]
                state_indices = torch.zeros(1, dtype=torch.int64, device=hidden_states.device)
            qkv = F.silu(self._causal_conv1d_prefill(qkv, seq_lens, state_indices))
        else:
            qkv = F.silu(self._causal_conv1d_decode(qkv, state_indices))

        # 3. Split into Q, K, V and reshape
        q_size = self.key_dim // self.tp_size
        k_size = self.key_dim // self.tp_size
        v_size = self.value_dim // self.tp_size
        q, k, v_flat = qkv.split([q_size, k_size, v_size], dim=-1)
        q = q.view(num_tokens, -1, self.head_k_dim)  # [N, num_k_heads/tp, head_k_dim]
        k = k.view(num_tokens, -1, self.head_k_dim)
        v = v_flat.view(num_tokens, -1, self.head_v_dim)  # [N, num_v_heads/tp, head_v_dim]
        z = z.view(num_tokens, -1, self.head_v_dim)

        # 4-5. Delta rule recurrence
        if context.is_prefill:
            # Prefill: compute g, beta explicitly for HF's chunk implementation
            g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
            beta = torch.sigmoid(b)
            attn_out = self._delta_rule_prefill(q, k, v, g, beta, seq_lens, state_indices)
        else:
            # Decode: fused Triton kernel computes g, beta internally from a, b, A_log, dt_bias
            self._decode_a = a
            self._decode_b = b
            attn_out = self._delta_rule_decode(q, k, v, None, None, state_indices)

        # 6. Output: RMSNormGated(output, z) -> out_proj
        attn_out = attn_out.reshape(-1, self.head_v_dim)
        z_flat = z.reshape(-1, self.head_v_dim)
        attn_out = self.norm(attn_out, z_flat)
        attn_out = attn_out.view(num_tokens, -1)  # [N, value_dim/tp]
        output = self.out_proj(attn_out)
        return output
