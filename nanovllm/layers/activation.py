import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def _compute(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y

    def forward(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        result = self._compute(x)
        if out is not None:
            out.copy_(result)
            return out
        return result
