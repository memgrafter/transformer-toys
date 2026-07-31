"""JSON checkpoint persistence for the transparent Transformer."""

from __future__ import annotations

import json
from pathlib import Path

from trainable_transformer import TrainableTransformer


class JsonCheckpoint:
    """Save and load the existing human-readable checkpoint schema."""

    @staticmethod
    def save(model: TrainableTransformer, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as file:
            json.dump(model.to_state(), file, indent=2)
            file.write("\n")

    @staticmethod
    def load(path: str | Path) -> TrainableTransformer:
        return TrainableTransformer.from_json(path)
