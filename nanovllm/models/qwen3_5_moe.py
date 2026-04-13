import torch
from torch import nn
import torch.distributed as dist

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import (
    QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear,
)
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.layers.gdn import GDNAttention
from nanovllm.layers.moe import SparseMoEBlock


class Qwen3_5MoeAttention(nn.Module):
    """Full softmax attention with output gate and partial RoPE."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position: int = 262144,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000000,
        partial_rotary_factor: float = 0.25,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.tp_size = tp_size
        self.tp_rank = dist.get_rank()
        self.total_num_heads = num_heads
        self.num_heads = num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        # When KV heads < TP size, vLLM gives each rank ONE kv head
        # (replicated across multiple ranks). nano-vllm previously kept all kv
        # heads per rank which gave wrong GQA grouping (group_size = num_q/num_kv
        # per rank, but global grouping is num_q_global/num_kv_global). The fix:
        # 1 kv head per rank, sliced based on tp_rank.
        if num_kv_heads >= tp_size:
            self.num_kv_heads = num_kv_heads // tp_size
            self.num_kv_head_replicas = 1
        else:
            assert tp_size % num_kv_heads == 0
            self.num_kv_heads = 1
            # How many ranks share each kv head (e.g., tp=4, kv=2 → 2 ranks/kv)
            self.num_kv_head_replicas = tp_size // num_kv_heads
        # The kv head index this rank owns (in the global kv head space)
        self.kv_head_idx = self.tp_rank // self.num_kv_head_replicas
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        # Separate projections for Q+Gate, K, V
        from nanovllm.layers.linear import ColumnParallelLinear, ReplicatedLinear
        self.q_proj = ColumnParallelLinear(
            hidden_size, self.total_num_heads * self.head_dim * 2, bias=False)  # Q + Gate
        # When KV heads < TP, replicate KV on each rank
        if self.total_num_kv_heads >= tp_size:
            self.k_proj = ColumnParallelLinear(
                hidden_size, self.total_num_kv_heads * self.head_dim, bias=False)
            self.v_proj = ColumnParallelLinear(
                hidden_size, self.total_num_kv_heads * self.head_dim, bias=False)
        else:
            self.k_proj = ReplicatedLinear(
                hidden_size, self.total_num_kv_heads * self.head_dim, bias=False)
            self.v_proj = ReplicatedLinear(
                hidden_size, self.total_num_kv_heads * self.head_dim, bias=False)
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )

        rotary_dim = int(head_dim * partial_rotary_factor)
        self.rotary_emb = get_rope(
            head_dim,
            rotary_dim=rotary_dim,
            max_position=max_position,
            base=rope_theta,
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )

        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps, add_unit_offset=True)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps, add_unit_offset=True)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        q_gate = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Separate Q and Gate within each head
        q_gate = q_gate.view(-1, self.num_heads, self.head_dim * 2)
        q, gate = q_gate.chunk(2, dim=-1)

        # Reshape kv to [N, total_num_kv_heads, head_dim]; when KV is replicated
        # across ranks (total_num_kv_heads < tp_size), each rank only needs the
        # ONE kv head matching its q head shard — slice to that.
        k = k.view(-1, self.total_num_kv_heads, self.head_dim)
        v = v.view(-1, self.total_num_kv_heads, self.head_dim)
        if self.num_kv_head_replicas > 1:
            k = k[:, self.kv_head_idx:self.kv_head_idx + self.num_kv_heads].contiguous()
            v = v[:, self.kv_head_idx:self.kv_head_idx + self.num_kv_heads].contiguous()

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)

        o = self.attn(q, k, v)

        gate = torch.sigmoid(gate.reshape(-1, self.num_heads * self.head_dim))
        o = o.flatten(1, -1) * gate

        return self.o_proj(o)


class Qwen3_5MoeDecoderLayer(nn.Module):

    def __init__(self, config, layer_type: str) -> None:
        super().__init__()
        self.layer_type = layer_type

        if layer_type == "linear_attention":
            self.linear_attn = GDNAttention(
                hidden_size=config.hidden_size,
                num_k_heads=config.linear_num_key_heads,
                num_v_heads=config.linear_num_value_heads,
                head_k_dim=config.linear_key_head_dim,
                head_v_dim=config.linear_value_head_dim,
                conv_kernel_size=config.linear_conv_kernel_dim,
                rms_norm_eps=config.rms_norm_eps,
            )
        elif layer_type == "full_attention":
            self.self_attn = Qwen3_5MoeAttention(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=getattr(config, 'head_dim', 256),
                max_position=config.max_position_embeddings,
                rms_norm_eps=config.rms_norm_eps,
                rope_theta=getattr(config, 'rope_theta', 10000000),
                partial_rotary_factor=getattr(config, 'partial_rotary_factor', 0.25),
            )

        self.mlp = SparseMoEBlock(
            hidden_size=config.hidden_size,
            moe_intermediate_size=config.moe_intermediate_size,
            shared_expert_intermediate_size=getattr(config, 'shared_expert_intermediate_size', 0),
            num_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            hidden_act=config.hidden_act,
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add_unit_offset=True)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add_unit_offset=True)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Qwen3.5 uses post-add residual (not fused pre-norm+residual like Qwen3)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Attention (GDN or full)
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states)
        else:
            hidden_states = self.self_attn(positions, hidden_states)

        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, None


class Qwen3_5MoeModel(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)

        # Build layers with hybrid types
        layer_types = getattr(config, 'layer_types', None)
        if layer_types is None:
            interval = getattr(config, 'full_attention_interval', 4)
            layer_types = [
                "full_attention" if (i + 1) % interval == 0 else "linear_attention"
                for i in range(config.num_hidden_layers)
            ]

        self.layers = nn.ModuleList([
            Qwen3_5MoeDecoderLayer(config, layer_types[i])
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add_unit_offset=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states, _ = layer(positions, hidden_states, None)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class Qwen3_5MoeForCausalLM(nn.Module):
    packed_modules_mapping = {
        # Full attention Q/K/V are now separate ColumnParallelLinear (no packing needed)
        # GDN projections
        "in_proj_qkv": ("in_proj_qkvz", (0, 1, 2)),
        "in_proj_z": ("in_proj_qkvz", 3),
        "in_proj_b": ("in_proj_ba", 0),
        "in_proj_a": ("in_proj_ba", 1),
        # Shared expert MLP (gate_proj + up_proj → gate_up_proj)
        "shared_expert.gate_proj": ("shared_expert.gate_up_proj", 0),
        "shared_expert.up_proj": ("shared_expert.gate_up_proj", 1),
    }

    def __init__(self, config) -> None:
        super().__init__()
        self.model = Qwen3_5MoeModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if getattr(config, 'tie_word_embeddings', False):
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
