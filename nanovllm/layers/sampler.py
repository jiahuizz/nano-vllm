import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # Greedy when temperature == 0, otherwise random sampling
        logits = logits.float()
        # Clamp temperature to avoid div-by-zero; greedy is handled by the same argmax path
        # (dividing by ~0 makes softmax a hard argmax)
        temps = temperatures.unsqueeze(dim=1).clamp(min=1e-10)
        logits = logits.div_(temps)
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
