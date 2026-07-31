"""Loss functions for the transparent training stack."""

from __future__ import annotations

from autodiff import Value
from differentiable_ops import Vector


class CrossEntropyLoss:
    """Causal next-token cross-entropy over a sequence of logits."""

    def compute(self, logits: list[Vector], targets: list[int]) -> Value:
        if len(logits) != len(targets) or not logits:
            raise ValueError("logits and targets must be non-empty and equally sized")
        losses = []
        for position, target in enumerate(targets):
            row = logits[position]
            if target < 0 or target >= len(row):
                raise ValueError("target is outside the vocabulary")
            highest = max(value.data for value in row)
            normalizer = sum(((value - highest).exp() for value in row), Value(0.0)).log()
            losses.append(normalizer + highest - row[target])
        return sum(losses, Value(0.0)) / len(losses)
