import torch
import torch.nn.functional as F
from torch import nn

# Allow more torch.compile recompilations: RMSNorm is called with several
# distinct ranks/shapes (2D for hidden states, 3D for q_norm/k_norm head dim).
# Default cache_size_limit is 8 which gets hit quickly in this model.
torch._dynamo.config.cache_size_limit = 64


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        add_unit_offset: bool = False,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.add_unit_offset = add_unit_offset
        if add_unit_offset:
            self.weight = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.weight = nn.Parameter(torch.ones(hidden_size))

    def _apply_weight(self, x):
        if self.add_unit_offset:
            return x * (1.0 + self.weight.float())
        return x * self.weight.float()

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = self._apply_weight(x).to(orig_dtype)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = self._apply_weight(x).to(orig_dtype)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)


class RMSNormGated(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        add_unit_offset: bool = False,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.add_unit_offset = add_unit_offset
        if add_unit_offset:
            self.weight = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        w = (1.0 + self.weight.float()) if self.add_unit_offset else self.weight.float()
        x = (x * w).to(orig_dtype)
        return x * F.silu(z.float()).to(orig_dtype)
