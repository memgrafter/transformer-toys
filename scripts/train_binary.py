#!/usr/bin/env python3
"""Train the JSON toy Transformer to predict alternating binary tokens."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import save_file

# Allow `python scripts/train_binary.py` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transformer import ToyTransformer  # noqa: E402


class TrainableToyTransformer(nn.Module):
    """Autodiff version of the one-block model in transformer.py."""

    def __init__(self, checkpoint: ToyTransformer) -> None:
        super().__init__()
        self.vocabulary_size = checkpoint.vocabulary_size
        self.model_width = checkpoint.model_width
        self.feedforward_width = checkpoint.feedforward_width
        self.maximum_sequence_length = checkpoint.maximum_sequence_length

        self.token_embedding = nn.Parameter(
            torch.tensor(checkpoint.token_embedding, dtype=torch.float32)
        )
        self.position_embedding = nn.Parameter(
            torch.tensor(checkpoint.position_embedding, dtype=torch.float32)
        )
        self.query_weight = nn.Parameter(torch.tensor(checkpoint.query_weight))
        self.key_weight = nn.Parameter(torch.tensor(checkpoint.key_weight))
        self.value_weight = nn.Parameter(torch.tensor(checkpoint.value_weight))
        self.output_weight = nn.Parameter(torch.tensor(checkpoint.output_weight))
        self.first_feedforward_weight = nn.Parameter(
            torch.tensor(checkpoint.first_feedforward_weight)
        )
        self.second_feedforward_weight = nn.Parameter(
            torch.tensor(checkpoint.second_feedforward_weight)
        )
        self.first_feedforward_bias = nn.Parameter(
            torch.tensor(checkpoint.first_feedforward_bias)
        )
        self.second_feedforward_bias = nn.Parameter(
            torch.tensor(checkpoint.second_feedforward_bias)
        )
        self.output_bias = nn.Parameter(torch.tensor(checkpoint.output_bias))

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(values, (self.model_width,), eps=1e-5)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        sequence_length = token_ids.shape[1]
        if sequence_length > self.maximum_sequence_length:
            raise ValueError("sequence is longer than the configured limit")

        positions = self.position_embedding[:sequence_length]
        hidden = self.token_embedding[token_ids] + positions
        hidden = self.normalize(hidden)

        queries = hidden @ self.query_weight
        keys = hidden @ self.key_weight
        values = hidden @ self.value_weight
        scores = queries @ keys.transpose(-1, -2)
        scores = scores / self.model_width**0.5
        causal_mask = torch.triu(
            torch.ones(sequence_length, sequence_length, device=token_ids.device),
            diagonal=1,
        ).bool()
        scores = scores.masked_fill(causal_mask, float("-inf"))
        probabilities = F.softmax(scores, dim=-1)
        context = probabilities @ values
        attended = hidden + context @ self.output_weight

        normalized = self.normalize(attended)
        feedforward = normalized @ self.first_feedforward_weight
        feedforward = F.relu(feedforward + self.first_feedforward_bias)
        feedforward = feedforward @ self.second_feedforward_weight
        feedforward = feedforward + self.second_feedforward_bias
        block_output = self.normalize(normalized + feedforward)

        # Match transformer.py's tied input/output embeddings.
        return block_output @ self.token_embedding.transpose(0, 1) + self.output_bias


def alternating_batch(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return both phases of the repeating 0, 1 sequence."""
    inputs = torch.tensor(
        [[0, 1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0, 1]],
        dtype=torch.int64,
        device=device,
    )
    targets = 1 - inputs
    return inputs, targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("transformer.json"))
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("binary-trained.safetensors"),
        help="Output path; .json writes the human-readable checkpoint schema",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(model: TrainableToyTransformer, output: Path) -> None:
    state = {
        name: value.detach().cpu().tolist()
        for name, value in model.state_dict().items()
    }
    checkpoint = {
        "vocabulary_size": model.vocabulary_size,
        "model_width": model.model_width,
        "feedforward_width": model.feedforward_width,
        "maximum_sequence_length": model.maximum_sequence_length,
        "random_state": 0,
        **state,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(checkpoint, file, indent=2)
        file.write("\\n")


def save_safetensors_checkpoint(model: TrainableToyTransformer, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    tensors = {
        name: value.detach().cpu().contiguous()
        for name, value in model.state_dict().items()
    }
    save_file(
        tensors,
        str(output),
        metadata={
            "format": "transformer-poking-binary-training",
            "vocabulary_size": str(model.vocabulary_size),
            "model_width": str(model.model_width),
            "feedforward_width": str(model.feedforward_width),
            "maximum_sequence_length": str(model.maximum_sequence_length),
        },
    )


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive")
    device = choose_device(args.device)
    checkpoint = ToyTransformer.from_json(str(args.checkpoint))
    model = TrainableToyTransformer(checkpoint).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    inputs, targets = alternating_batch(device)

    model.train()
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, model.vocabulary_size), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % max(1, args.epochs // 10) == 0:
            print(f"epoch={epoch:4d} loss={loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        predictions = model(inputs).argmax(dim=-1)

    if args.output.suffix.lower() == ".json":
        save_checkpoint(model, args.output)
    else:
        save_safetensors_checkpoint(model, args.output)

    print("saved:", args.output)
    print("device:", device)
    print("inputs:", inputs.cpu().tolist())
    print("targets:", targets.cpu().tolist())
    print("predictions:", predictions.cpu().tolist())


if __name__ == "__main__":
    main()
