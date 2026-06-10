from __future__ import annotations

import math


class NoamScheduler:
    """Transformer learning-rate schedule from Attention Is All You Need."""

    def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000, factor: float = 1.0) -> None:
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.step_num = 0

    def step(self) -> float:
        self.step_num += 1
        lr = self.rate()
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.step()
        return lr

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def rate(self) -> float:
        step = max(1, self.step_num)
        return self.factor * (self.d_model ** -0.5) * min(step ** -0.5, step * self.warmup_steps ** -1.5)
