"""Fused Triton kernels for GDN decode path.

Adapted from vLLM's causal_conv1d and fla ops.
Two kernels replace ~25 PyTorch ops per GDN layer:
1. fused_conv1d_update: conv state shift + convolution + silu (1 kernel)
2. fused_delta_rule_update: g/beta computation + L2norm + state update + output (1 kernel)
"""
import torch
import triton
import triton.language as tl
from math import exp as math_exp


# ---------------------------------------------------------------------------
# Kernel 1: Fused causal conv1d update + silu
# ---------------------------------------------------------------------------

@triton.jit
def _fused_conv1d_update_kernel(
    x_ptr,              # [batch, conv_dim] input
    conv_state_ptr,     # [max_seqs, conv_dim, kernel-1] state buffer
    weight_ptr,         # [conv_dim, kernel] conv weights
    out_ptr,            # [batch, conv_dim] output (can be same as x_ptr)
    state_indices_ptr,  # [batch] indices into conv_state
    batch: tl.constexpr,
    conv_dim,
    SILU: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Single-token conv1d decode: read state → conv+silu → update state. One kernel."""
    # Each program handles one sequence, BLOCK_D channels at a time
    i_seq = tl.program_id(0)
    if i_seq >= batch:
        return
    i_d_block = tl.program_id(1)
    offs_d = i_d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < conv_dim

    state_idx = tl.load(state_indices_ptr + i_seq).to(tl.int64)

    # Load conv state [3 columns for kernel_width=4] and current input
    # conv_state layout: [max_seqs, conv_dim, kernel-1] = [*, conv_dim, 3]
    state_base = conv_state_ptr + state_idx * conv_dim * 3 + offs_d * 3
    s0 = tl.load(state_base + 0, mask=mask_d, other=0.0)
    s1 = tl.load(state_base + 1, mask=mask_d, other=0.0)
    s2 = tl.load(state_base + 2, mask=mask_d, other=0.0)

    # Current input token
    x = tl.load(x_ptr + i_seq * conv_dim + offs_d, mask=mask_d, other=0.0)

    # Load weights [conv_dim, 4]
    w_base = weight_ptr + offs_d * 4
    w0 = tl.load(w_base + 0, mask=mask_d, other=0.0)
    w1 = tl.load(w_base + 1, mask=mask_d, other=0.0)
    w2 = tl.load(w_base + 2, mask=mask_d, other=0.0)
    w3 = tl.load(w_base + 3, mask=mask_d, other=0.0)

    # Convolution: dot product of [s0, s1, s2, x] with [w0, w1, w2, w3]
    acc = s0 * w0 + s1 * w1 + s2 * w2 + x * w3

    # Silu activation
    if SILU:
        acc = acc / (1.0 + tl.exp(-acc.to(tl.float32)))

    # Write output
    tl.store(out_ptr + i_seq * conv_dim + offs_d, acc, mask=mask_d)

    # Update state: shift left, append x
    tl.store(state_base + 0, s1, mask=mask_d)
    tl.store(state_base + 1, s2, mask=mask_d)
    tl.store(state_base + 2, x, mask=mask_d)


def fused_conv1d_update(
    x: torch.Tensor,              # [batch, conv_dim]
    conv_state: torch.Tensor,     # [max_seqs, conv_dim, kernel-1]
    weight: torch.Tensor,         # [conv_dim, kernel]
    state_indices: torch.Tensor,  # [batch]
    silu: bool = True,
) -> torch.Tensor:
    """Fused causal conv1d single-step decode. Returns output same shape as x.

    NOTE: x may be a non-contiguous view (e.g. from qkvz.split()) so we must
    call .contiguous() before passing to the Triton kernel, which assumes
    stride == conv_dim on the batch dim. conv_state and weight are also
    required to be contiguous.
    """
    x = x.contiguous()
    batch, conv_dim = x.shape
    out = torch.empty_like(x)
    BLOCK_D = 256
    grid = (batch, triton.cdiv(conv_dim, BLOCK_D))
    _fused_conv1d_update_kernel[grid](
        x, conv_state, weight, out, state_indices,
        batch, conv_dim,
        SILU=silu, BLOCK_D=BLOCK_D,
    )
    return out


# ---------------------------------------------------------------------------
# Kernel 2: Fused delta rule decode (g/beta + L2norm + state update + output)
# Adapted from vLLM's fused_sigmoid_gating_delta_rule_update
# ---------------------------------------------------------------------------

@triton.jit
def _fused_delta_rule_decode_kernel(
    q_ptr, k_ptr, v_ptr,     # [batch, num_v_heads/tp, head_dim]
    a_ptr, b_ptr,             # [batch, num_v_heads/tp]
    A_log_ptr, dt_bias_ptr,   # [num_v_heads/tp]
    state_ptr,                # [max_seqs, num_v_heads/tp, head_k_dim, head_v_dim]
    state_indices_ptr,        # [batch]
    o_ptr,                    # [batch, num_v_heads/tp, head_v_dim]
    scale,
    num_k_heads_tp,           # H (Q/K heads per TP)
    num_v_heads_tp: tl.constexpr,  # HV (V heads per TP)
    head_k_dim: tl.constexpr,      # K
    head_v_dim: tl.constexpr,      # V
    stride_state_seq,         # stride for state's sequence dim
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_L2NORM: tl.constexpr,
):
    """Fused delta rule decode: one kernel per (batch, head_v, v_block)."""
    i_v = tl.program_id(0)   # V block index
    i_nh = tl.program_id(1)  # batch * num_v_heads
    i_n = i_nh // num_v_heads_tp
    i_hv = i_nh % num_v_heads_tp
    # Map v-head to k-head (GQA)
    i_h = i_hv // (num_v_heads_tp // num_k_heads_tp)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < head_k_dim
    mask_v = o_v < head_v_dim
    # State layout: [heads, K, V] (matching HF's convention)
    mask_h = mask_k[:, None] & mask_v[None, :]  # [BK, BV]

    # Load state index
    state_idx = tl.load(state_indices_ptr + i_n).to(tl.int64)

    # Load initial state as [BK, BV] — state is [heads, K_dim, V_dim]
    p_h = state_ptr + state_idx * stride_state_seq + i_hv * head_k_dim * head_v_dim
    p_h = p_h + o_k[:, None] * head_v_dim + o_v[None, :]
    b_h = tl.load(p_h, mask=mask_h, other=0).to(tl.float32)  # [BK, BV]

    # Load q, k, v
    b_q = tl.load(q_ptr + (i_n * num_v_heads_tp + i_h) * head_k_dim + o_k,
                  mask=mask_k, other=0).to(tl.float32)  # [BK]
    b_k = tl.load(k_ptr + (i_n * num_v_heads_tp + i_h) * head_k_dim + o_k,
                  mask=mask_k, other=0).to(tl.float32)  # [BK]
    b_v = tl.load(v_ptr + (i_n * num_v_heads_tp + i_hv) * head_v_dim + o_v,
                  mask=mask_v, other=0).to(tl.float32)  # [BV]

    # L2 norm on q, k
    if USE_L2NORM:
        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q = b_q * scale

    # Compute g and beta from a, b, A_log, dt_bias
    a_val = tl.load(a_ptr + i_n * num_v_heads_tp + i_hv).to(tl.float32)
    b_val = tl.load(b_ptr + i_n * num_v_heads_tp + i_hv).to(tl.float32)
    A_log_val = tl.load(A_log_ptr + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias_ptr + i_hv).to(tl.float32)

    x = a_val + dt_bias_val
    softplus_x = tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)
    g_val = -tl.exp(A_log_val) * softplus_x
    beta_val = tl.sigmoid(b_val)

    # Delta rule with state [BK, BV]:
    # state *= exp(g)
    b_h = b_h * tl.exp(g_val)
    # v -= state^T @ k  →  v[v] -= sum_k(state[k,v] * k[k])
    b_v = b_v - tl.sum(b_h * b_k[:, None], 0)
    b_v = b_v * beta_val
    # state += outer(k, v)  →  state[k,v] += k[k] * v[v]
    b_h = b_h + b_k[:, None] * b_v[None, :]

    # Output: o[v] = sum_k(state[k,v] * q[k])
    b_o = tl.sum(b_h * b_q[:, None], 0)

    # Write output
    tl.store(o_ptr + (i_n * num_v_heads_tp + i_hv) * head_v_dim + o_v,
             b_o.to(o_ptr.dtype.element_ty), mask=mask_v)

    # Write back state [BK, BV]
    tl.store(p_h, b_h.to(state_ptr.dtype.element_ty), mask=mask_h)


def fused_delta_rule_decode(
    q: torch.Tensor,           # [batch, num_k_heads/tp, head_k_dim]
    k: torch.Tensor,           # [batch, num_k_heads/tp, head_k_dim]
    v: torch.Tensor,           # [batch, num_v_heads/tp, head_v_dim]
    a: torch.Tensor,           # [batch, num_v_heads/tp]
    b: torch.Tensor,           # [batch, num_v_heads/tp]
    A_log: torch.Tensor,       # [num_v_heads/tp]
    dt_bias: torch.Tensor,     # [num_v_heads/tp]
    state: torch.Tensor,       # [max_seqs, num_v_heads/tp, head_k_dim, head_v_dim]
    state_indices: torch.Tensor,  # [batch]
    use_l2norm: bool = True,
) -> torch.Tensor:
    """Fused delta rule single-step decode. Returns output [batch, num_v_heads/tp, head_v_dim]."""
    batch = q.size(0)
    num_k_heads_tp = q.size(1)
    num_v_heads_tp = v.size(1)
    head_k_dim = q.size(2)
    head_v_dim = v.size(2)
    scale = head_k_dim ** -0.5

    # GQA: expand q, k to match v heads (done inside kernel via i_h mapping)
    # Reshape q, k to [batch, num_v_heads/tp, head_k_dim] by repeating
    n_rep = num_v_heads_tp // num_k_heads_tp
    if n_rep > 1:
        q = q.repeat_interleave(n_rep, dim=1)
        k = k.repeat_interleave(n_rep, dim=1)

    o = torch.empty_like(v)
    BK = triton.next_power_of_2(head_k_dim)
    BV = min(triton.next_power_of_2(head_v_dim), 32)
    NV = triton.cdiv(head_v_dim, BV)

    grid = (NV, batch * num_v_heads_tp)
    _fused_delta_rule_decode_kernel[grid](
        q.contiguous(), k.contiguous(), v.contiguous(),
        a.contiguous(), b.contiguous(),
        A_log, dt_bias,
        state, state_indices, o,
        scale,
        num_k_heads_tp * n_rep,  # after GQA expansion, all heads are num_v_heads_tp
        num_v_heads_tp=num_v_heads_tp,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        stride_state_seq=state.stride(0),
        BK=BK, BV=BV,
        USE_L2NORM=use_l2norm,
    )
    return o
