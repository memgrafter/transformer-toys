#!/usr/bin/env python3
"""Convert a toy Transformer JSON checkpoint to a Safetensors file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file


MATRIX_FIELDS = (
    "token_embedding",
    "position_embedding",
    "query_weight",
    "key_weight",
    "value_weight",
    "output_weight",
    "first_feedforward_weight",
    "second_feedforward_weight",
)
VECTOR_FIELDS = (
    "first_feedforward_bias",
    "second_feedforward_bias",
    "output_bias",
)
REQUIRED_FIELDS = {
    "vocabulary_size",
    "model_width",
    "feedforward_width",
    "maximum_sequence_length",
    "random_state",
    *MATRIX_FIELDS,
    *VECTOR_FIELDS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert transformer.json to a Safetensors checkpoint."
    )
    parser.add_argument("input", type=Path, help="Source JSON checkpoint")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="Destination file (default: input with a .safetensors suffix)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing destination file",
    )
    return parser.parse_args()


def require_integer(state: dict[str, Any], name: str) -> int:
    value = state[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value <= 0 and name != "random_state":
        raise ValueError(f"{name} must be positive")
    return value


def validate_checkpoint(state: dict[str, Any]) -> dict[str, int]:
    missing = REQUIRED_FIELDS - state.keys()
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"checkpoint is missing fields: {names}")

    dimensions = {
        name: require_integer(state, name)
        for name in (
            "vocabulary_size",
            "model_width",
            "feedforward_width",
            "maximum_sequence_length",
            "random_state",
        )
    }
    v, d, f, context = (
        dimensions["vocabulary_size"],
        dimensions["model_width"],
        dimensions["feedforward_width"],
        dimensions["maximum_sequence_length"],
    )
    expected_shapes = {
        "token_embedding": (v, d),
        "position_embedding": (context, d),
        "query_weight": (d, d),
        "key_weight": (d, d),
        "value_weight": (d, d),
        "output_weight": (d, d),
        "first_feedforward_weight": (d, f),
        "second_feedforward_weight": (f, d),
        "first_feedforward_bias": (f,),
        "second_feedforward_bias": (d,),
        "output_bias": (v,),
    }
    for name, expected in expected_shapes.items():
        actual = torch.as_tensor(state[name]).shape
        if tuple(actual) != expected:
            raise ValueError(f"{name} has shape {tuple(actual)}, expected {expected}")

    return dimensions


def convert(input_path: Path, output_path: Path, force: bool) -> None:
    if output_path.exists() and not force:
        raise FileExistsError(f"destination exists: {output_path} (use --force)")

    with input_path.open(encoding="utf-8") as file:
        state = json.load(file)
    if not isinstance(state, dict):
        raise ValueError("checkpoint root must be a JSON object")

    dimensions = validate_checkpoint(state)
    tensors = {
        name: torch.as_tensor(state[name], dtype=torch.float32)
        for name in (*MATRIX_FIELDS, *VECTOR_FIELDS)
    }
    tensors["random_state"] = torch.tensor(
        dimensions["random_state"], dtype=torch.int64
    )
    if not all(torch.isfinite(tensor).all().item() for tensor in tensors.values()):
        raise ValueError("checkpoint contains a non-finite weight or bias")

    metadata = {
        "format": "transformer-poking",
        "dtype": "float32",
        "vocabulary_size": str(dimensions["vocabulary_size"]),
        "model_width": str(dimensions["model_width"]),
        "feedforward_width": str(dimensions["feedforward_width"]),
        "maximum_sequence_length": str(dimensions["maximum_sequence_length"]),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output_path), metadata=metadata)


def main() -> None:
    args = parse_args()
    output = args.output or args.input.with_suffix(".safetensors")
    try:
        convert(args.input, output, args.force)
    except (OSError, ValueError, TypeError) as error:
        raise SystemExit(f"conversion failed: {error}") from error
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
