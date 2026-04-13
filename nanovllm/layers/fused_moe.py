"""Fused MoE: Triton grouped GEMM for Mixture of Experts.

Simplified from vLLM's fused_moe. BF16 only, no quantization.
Optimized: no .item() calls, pre-allocated buffers, CUDA Graph compatible.
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernels for token alignment (CUDA Graph compatible)
# Replaces argsort + scatter with atomic operations
# ---------------------------------------------------------------------------

@triton.jit
def _moe_count_kernel(flat_ids_ptr, counts_ptr, num_tokens, BLOCK: tl.constexpr):
    """Count tokens per expert using atomic add."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < num_tokens
    expert = tl.load(flat_ids_ptr + offs, mask=mask, other=0)
    tl.atomic_add(counts_ptr + expert, 1, mask=mask)


@triton.jit
def _moe_scatter_kernel(flat_ids_ptr, cumsum_ptr, counter_ptr, sorted_ids_ptr,
                        num_tokens, BLOCK: tl.constexpr):
    """Scatter tokens into sorted positions using atomic add for within-expert offset."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < num_tokens
    expert = tl.load(flat_ids_ptr + offs, mask=mask, other=0).to(tl.int64)
    # Atomic add to get unique position within expert group
    pos = tl.atomic_add(counter_ptr + expert, 1, mask=mask).to(tl.int64)
    # Destination = cumsum[expert] + pos
    expert_offset = tl.load(cumsum_ptr + expert, mask=mask, other=0)
    dest = expert_offset + pos
    # Write token index
    tl.store(sorted_ids_ptr + dest, offs.to(tl.int32), mask=mask)


@triton.jit
def _moe_expert_ids_kernel(cum_blocks_ptr, expert_ids_ptr, max_blocks,
                           NUM_EXPERTS: tl.constexpr, BLOCK: tl.constexpr):
    """Fill expert_ids from cumulative block counts. One program per expert."""
    expert = tl.program_id(0)
    start = tl.load(cum_blocks_ptr + expert).to(tl.int64)
    end = tl.load(cum_blocks_ptr + expert + 1).to(tl.int64)
    # Fill expert_ids[start:end] = expert
    offs = start + tl.arange(0, BLOCK).to(tl.int64)
    mask = (offs < end) & (offs < max_blocks)
    tl.store(expert_ids_ptr + offs, tl.full((BLOCK,), expert, dtype=tl.int32), mask=mask)


# ---------------------------------------------------------------------------
# Token alignment wrapper (CUDA Graph safe — no argsort, no .item())
# ---------------------------------------------------------------------------

def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    sorted_token_ids_buf: torch.Tensor | None = None,
    expert_ids_buf: torch.Tensor | None = None,
    num_tokens_post_padded_buf: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort tokens by expert, pad to block_size. Fully Triton-based, CUDA Graph safe."""
    M, top_k = topk_ids.shape
    num_tokens = M * top_k
    max_num_tokens_padded = num_tokens + num_experts * (block_size - 1)
    max_blocks = max_num_tokens_padded // block_size
    device = topk_ids.device
    flat_ids = topk_ids.view(-1).to(torch.int32)

    # Allocate or reuse buffers
    if sorted_token_ids_buf is None:
        sorted_token_ids_buf = torch.empty(max_num_tokens_padded, dtype=torch.int32, device=device)
    if expert_ids_buf is None:
        expert_ids_buf = torch.empty(max_blocks, dtype=torch.int32, device=device)
    if num_tokens_post_padded_buf is None:
        num_tokens_post_padded_buf = torch.empty(1, dtype=torch.int32, device=device)

    sorted_token_ids_buf[:max_num_tokens_padded].fill_(num_tokens)
    expert_ids_buf[:max_blocks].fill_(-1)

    # 1. Count tokens per expert (Triton atomic)
    counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
    grid1 = (triton.cdiv(num_tokens, 256),)
    _moe_count_kernel[grid1](flat_ids, counts, num_tokens, BLOCK=256)

    # 2. Compute padded cumsum (torch ops on small [num_experts] tensor — graph safe)
    padded = ((counts.to(torch.int64) + block_size - 1) // block_size * block_size)
    cumsum = torch.zeros(num_experts + 1, dtype=torch.int64, device=device)
    cumsum[1:] = padded.cumsum(0)
    num_tokens_post_padded_buf[0] = cumsum[num_experts].to(torch.int32)

    # 3. Scatter tokens to sorted positions (Triton atomic)
    counter = torch.zeros(num_experts, dtype=torch.int32, device=device)
    grid2 = (triton.cdiv(num_tokens, 256),)
    _moe_scatter_kernel[grid2](flat_ids, cumsum, counter, sorted_token_ids_buf,
                               num_tokens, BLOCK=256)

    # 4. Fill expert_ids (Triton, one program per expert)
    blocks_per_expert = padded // block_size
    cum_blocks = torch.zeros(num_experts + 1, dtype=torch.int64, device=device)
    cum_blocks[1:] = blocks_per_expert.cumsum(0)
    # Max blocks any single expert could have (for BLOCK size)
    max_expert_blocks = max(1, max_num_tokens_padded // block_size // num_experts + 2)
    # Round up to power of 2 for Triton
    triton_block = 1
    while triton_block < max_expert_blocks:
        triton_block *= 2
    triton_block = max(triton_block, 16)
    _moe_expert_ids_kernel[(num_experts,)](cum_blocks, expert_ids_buf, max_blocks,
                                           NUM_EXPERTS=num_experts, BLOCK=triton_block)

    return sorted_token_ids_buf[:max_num_tokens_padded], expert_ids_buf[:max_blocks], num_tokens_post_padded_buf


# ---------------------------------------------------------------------------
# Triton kernel (BF16-only, stripped from vLLM)
# ---------------------------------------------------------------------------

@triton.jit
def _write_zeros(c_ptr, stride_cm, stride_cn, pid_n, N, offs_token, token_mask,
                 BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, compute_type: tl.constexpr):
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, acc, mask=token_mask[:, None] & (offs_cn[None, :] < N))


@triton.jit
def fused_moe_kernel(
    a_ptr, b_ptr, c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr, expert_ids_ptr, num_tokens_post_padded_ptr,
    N, K, EM, num_valid_tokens,
    stride_am, stride_ak, stride_be, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + pid_m * BLOCK_SIZE_M + offs).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        _write_zeros(c_ptr, stride_cm, stride_cn, pid_n, N, offs_token, token_mask,
                     BLOCK_SIZE_M, BLOCK_SIZE_N, compute_type)
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + off_experts * stride_be + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator *= moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, accumulator, mask=token_mask[:, None] & (offs_cn[None, :] < N))


# ---------------------------------------------------------------------------
# Pre-allocated buffer cache for fused MoE (avoids per-step allocation)
# ---------------------------------------------------------------------------

class FusedMoECache:
    """Pre-allocated buffers for fused MoE forward. Reused across steps."""

    def __init__(self, max_tokens: int, top_k: int, gate_up_size: int,
                 inter_size: int, hidden_size: int, num_experts: int,
                 block_size: int, dtype: torch.dtype, device: torch.device):
        mt = max_tokens * top_k
        max_padded = mt + num_experts * (block_size - 1)
        max_blocks = max_padded // block_size

        self.intermediate1 = torch.zeros(mt, gate_up_size, dtype=dtype, device=device)
        self.intermediate2 = torch.zeros(mt, inter_size, dtype=dtype, device=device)
        self.output = torch.zeros(mt, hidden_size, dtype=dtype, device=device)
        self.sorted_token_ids = torch.empty(max_padded, dtype=torch.int32, device=device)
        self.expert_ids = torch.empty(max_blocks, dtype=torch.int32, device=device)
        self.num_tokens_post_padded = torch.empty(1, dtype=torch.int32, device=device)
        self.topk_weights_flat = torch.empty(mt, dtype=dtype, device=device)


_MOE_CACHE: dict[int, FusedMoECache] = {}


def _get_config(M: int, E: int, N: int, K: int, top_k: int) -> dict:
    """Mirror of vLLM's fused_moe.get_default_config for bf16 / non-quantized case.

    M here is the REQUEST-level batch size (num rows of the input A, before
    top-k expansion), not num_tokens_post_padded. vLLM tunes tile sizes based
    on request count, not the kernel's internal expansion.
    """
    # BLOCK_M grows with batch, but slower than M * top_k so we keep enough
    # parallelism for small batches and enough arithmetic intensity for large.
    if M <= 32:
        block_m = 16
    elif M <= 96:
        block_m = 32
    elif M <= 512:
        block_m = 64
    else:
        block_m = 128

    # Small batches are memory-bound → favor tall-K tiles (larger K).
    block_n = 64 if M <= 64 else 128
    block_k = 128 if M <= 64 else 64

    # GROUP_SIZE_M helps adjacent M-blocks share weight tiles in L2, but only
    # pays off when each expert has many M-blocks. In Qwen3.5-35B-A3B we have
    # 256 experts and typical decode M=64 → tokens_per_expert ≈ 2 → grouping
    # HURTS because M-blocks rarely belong to the same expert.
    tokens_per_expert = M // max(E, 1)
    group_m = 16 if tokens_per_expert > 128 else 1

    # More warps = more parallelism within a block = higher arithmetic intensity,
    # but only when blocks are big enough to saturate. M > 128 means large
    # batches where 8 warps (256 threads) fit well.
    num_warps = 4 if M <= 128 else 8

    # Pipeline depth: small M has less work to hide latency, so shallower
    # pipeline (stages=4 on tiny batches actually helps by giving the
    # scheduler more independent loads to pick from).
    num_stages = 4 if M <= 32 else 3

    return {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": block_k,
        "GROUP_SIZE_M": group_m,
        "num_warps": num_warps,
        "num_stages": num_stages,
    }


def invoke_fused_moe(A, B, C, topk_weights, sorted_token_ids, expert_ids,
                     num_tokens_post_padded, mul_routed_weight, top_k,
                     num_requests, num_experts):
    """Launch fused MoE kernel.

    num_requests: A.size(0) / top_k at the original call site (or A.size(0) if
        top_k=1 like GEMM2). Used for config selection.
    num_experts: total experts, used to compute tokens_per_expert for the
        GROUP_SIZE_M heuristic.
    """
    M = A.size(0)
    N, K = B.size(1), B.size(2)
    EM = sorted_token_ids.size(0)
    config = _get_config(num_requests, num_experts, N, K, top_k)
    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    compute_type = tl.bfloat16 if A.dtype == torch.bfloat16 else tl.float16
    fused_moe_kernel[grid](
        A, B, C, topk_weights,
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        N, K, EM, M * top_k,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(2), B.stride(1),
        C.stride(0), C.stride(1),
        MUL_ROUTED_WEIGHT=mul_routed_weight, top_k=top_k,
        compute_type=compute_type, **config,
    )


def fused_moe_forward(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    activation_fn,
) -> torch.Tensor:
    M, hidden = hidden_states.shape
    top_k = topk_ids.size(1)
    gate_up_size = w1.size(1)
    inter_size = w2.size(2)
    # Use request-level M for config selection (matches vLLM semantics).
    config = _get_config(M, num_experts, gate_up_size, hidden, top_k)
    block_size = config["BLOCK_SIZE_M"]

    # Get or create cache (keyed by device ordinal to support TP). Use the
    # MAX possible block_size for sizing cache buffers so any runtime config
    # (block_m ∈ {16, 32, 64, 128}) fits.
    MAX_BLOCK_M = 128
    device_idx = hidden_states.device.index or 0
    cache = _MOE_CACHE.get(device_idx)
    if cache is None or cache.intermediate1.size(0) < M * top_k:
        max_tokens = max(M, 512)  # pre-allocate for up to 512 tokens
        cache = FusedMoECache(max_tokens, top_k, gate_up_size, inter_size, hidden,
                              num_experts, MAX_BLOCK_M, hidden_states.dtype, hidden_states.device)
        _MOE_CACHE[device_idx] = cache

    mt = M * top_k

    # 1. Align tokens to blocks (reuses pre-allocated buffers)
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, block_size, num_experts,
        cache.sorted_token_ids, cache.expert_ids, cache.num_tokens_post_padded)

    # 2. GEMM1 into pre-allocated buffer
    cache.intermediate1[:mt].zero_()
    invoke_fused_moe(
        hidden_states, w1, cache.intermediate1[:mt],
        topk_weights=None,
        sorted_token_ids=sorted_token_ids, expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        mul_routed_weight=False, top_k=top_k,
        num_requests=M, num_experts=num_experts,
    )

    # 3. Activation in-place
    activation_fn(cache.intermediate1[:mt], cache.intermediate2[:mt])

    # 4. GEMM2 with routing weights (top_k=1 for independent intermediate rows)
    cache.output[:mt].zero_()
    cache.topk_weights_flat[:mt] = topk_weights.flatten()
    invoke_fused_moe(
        cache.intermediate2[:mt], w2, cache.output[:mt],
        topk_weights=cache.topk_weights_flat[:mt],
        sorted_token_ids=sorted_token_ids, expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        mul_routed_weight=True, top_k=1,
        num_requests=M, num_experts=num_experts,
    )

    # 5. Aggregate
    return cache.output[:mt].view(M, top_k, hidden).sum(dim=1)
