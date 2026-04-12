# Adapted from flash-linear-attention (MIT License)
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Via vLLM (Apache-2.0)
from .chunk import chunk_gated_delta_rule

__all__ = [
    "chunk_gated_delta_rule",
]
