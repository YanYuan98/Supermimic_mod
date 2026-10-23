"""Cosine learning-rate schedule (paper Table A.5): decay to `final_ratio` over `decay_iterations`, then constant."""

from __future__ import annotations

import math


def cosine_lr(iteration: int, base_lr: float, final_ratio: float = 0.03, decay_iterations: int = 250_000) -> float:
    decay = max(1, int(decay_iterations))
    if iteration >= decay:
        return base_lr * final_ratio
    cos = 0.5 * (1.0 + math.cos(math.pi * iteration / decay))
    return base_lr * (final_ratio + (1.0 - final_ratio) * cos)
