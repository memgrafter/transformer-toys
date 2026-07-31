"""Configuration for the transparent toy Transformer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TransformerConfig:
    vocabulary_size: int
    model_width: int
    feedforward_width: int
    maximum_sequence_length: int

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> TransformerConfig:
        names = (
            "vocabulary_size",
            "model_width",
            "feedforward_width",
            "maximum_sequence_length",
        )
        values = {name: state[name] for name in names}
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        return cls(**values)
