"""Small deterministic datasets for toy sequence experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingExample:
    input_ids: list[int]
    target_ids: list[int]


class AlternatingSequenceDataset:
    """Both phase offsets of a repeating zero/one next-token task."""

    def __init__(self, sequence_length: int = 7) -> None:
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        self.sequence_length = sequence_length

    def examples(self) -> list[TrainingExample]:
        examples = []
        for phase in (0, 1):
            sequence = [(phase + index) % 2 for index in range(self.sequence_length + 1)]
            examples.append(
                TrainingExample(
                    input_ids=sequence[:-1],
                    target_ids=sequence[1:],
                )
            )
        return examples
