import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float()
        # Greedy path: if any temperature is 0, use argmax (no noise)
        # For simplicity, if ALL temperatures are 0, do pure greedy; otherwise do random sampling
        if (temperatures == 0).all():
            return logits.argmax(dim=-1)
        # Random sampling via Gumbel-max trick
        temps = temperatures.unsqueeze(dim=1).clamp(min=1e-10)
        logits = logits.div_(temps)
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
