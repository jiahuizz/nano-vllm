import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.linear import (
    ReplicatedLinear, MergedColumnParallelLinear, RowParallelLinear,
)


class ExpertWeights(nn.Module):
    """Container for stacked expert weights so parameter path matches checkpoint."""

    def __init__(self, num_experts, gate_up_size, down_out_size, hidden_size, intermediate_per_tp, tp_rank):
        super().__init__()
        self.tp_rank = tp_rank
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, gate_up_size, hidden_size))
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, down_out_size, intermediate_per_tp))
        self.gate_up_proj.weight_loader = self._gate_up_weight_loader
        self.down_proj.weight_loader = self._down_weight_loader

    def _gate_up_weight_loader(self, param, loaded_weight):
        if loaded_weight.dim() == 3:
            # gate_up layout: [num_experts, gate(inter) | up(inter), hidden]
            # Must shard gate and up independently, then concatenate
            tp_size = dist.get_world_size()
            half = loaded_weight.size(1) // 2
            gate = loaded_weight[:, :half, :]
            up = loaded_weight[:, half:, :]
            shard_size = half // tp_size
            gate_shard = gate.narrow(1, self.tp_rank * shard_size, shard_size)
            up_shard = up.narrow(1, self.tp_rank * shard_size, shard_size)
            loaded_weight = torch.cat([gate_shard, up_shard], dim=1)
        param.data.copy_(loaded_weight)

    def _down_weight_loader(self, param, loaded_weight):
        if loaded_weight.dim() == 3:
            shard_size = param.data.size(2)
            start_idx = self.tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(2, start_idx, shard_size)
        param.data.copy_(loaded_weight)


class SparseMoEBlock(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_size: int,
        shared_expert_intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.tp_size = dist.get_world_size()
        self.tp_rank = dist.get_rank()
        intermediate_per_tp = moe_intermediate_size // self.tp_size

        self.gate = ReplicatedLinear(hidden_size, num_experts, bias=False)

        # Stacked expert weights in a sub-module so path = "experts.gate_up_proj"
        self.experts = ExpertWeights(
            num_experts, 2 * intermediate_per_tp, hidden_size,
            hidden_size, intermediate_per_tp, self.tp_rank)

        self.act_fn = SiluAndMul()

        # Shared expert (always applied to all tokens)
        if shared_expert_intermediate_size > 0:
            self.shared_expert = nn.Module()
            self.shared_expert.gate_up_proj = MergedColumnParallelLinear(
                hidden_size, [shared_expert_intermediate_size] * 2, bias=False)
            self.shared_expert.down_proj = RowParallelLinear(
                shared_expert_intermediate_size, hidden_size, bias=False)
            self.shared_expert_gate = ReplicatedLinear(hidden_size, 1, bias=False)
            self.has_shared_expert = True
        else:
            self.has_shared_expert = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape

        # Router: softmax → top-k
        router_logits = self.gate(hidden_states)  # [N, num_experts]
        scores = F.softmax(router_logits, dim=-1, dtype=torch.float)
        topk_scores, topk_indices = scores.topk(self.num_experts_per_tok, dim=-1)
        topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)
        topk_scores = topk_scores.to(router_logits.dtype)

        # Fused expert computation via Triton grouped GEMM
        from nanovllm.layers.fused_moe import fused_moe_forward
        output = fused_moe_forward(
            hidden_states,
            self.experts.gate_up_proj,   # [E, 2*inter/tp, hidden]
            self.experts.down_proj,      # [E, hidden, inter/tp]
            topk_scores,
            topk_indices,
            self.num_experts,
            self.act_fn,
        )

        # All-reduce across TP (down_proj is row-parallel)
        if self.tp_size > 1:
            reduce_dtype = torch.float32 if output.dtype in (torch.float16, torch.bfloat16) else output.dtype
            if reduce_dtype != output.dtype:
                output = output.to(reduce_dtype)
            dist.all_reduce(output)
            if reduce_dtype != hidden_states.dtype:
                output = output.to(hidden_states.dtype)

        # Shared expert
        if self.has_shared_expert:
            shared_gate_up = self.shared_expert.gate_up_proj(hidden_states)
            shared_act = self.act_fn(shared_gate_up)
            shared_out = self.shared_expert.down_proj(shared_act)
            shared_gate = torch.sigmoid(self.shared_expert_gate(hidden_states))
            output = output + shared_gate * shared_out

        return output
