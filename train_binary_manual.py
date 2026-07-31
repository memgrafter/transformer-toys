#!/usr/bin/env python3
"""Train the transparent Transformer without PyTorch."""

from __future__ import annotations

import argparse
from pathlib import Path

from checkpoint import JsonCheckpoint
from datasets import AlternatingSequenceDataset
from gradient_descent import GradientDescent
from losses import CrossEntropyLoss
from trainer import Trainer
from trainable_transformer import TrainableTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("transformer.json"))
    parser.add_argument("--output", type=Path, default=Path("binary-trained-manual.json"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--sequence-length", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = TrainableTransformer.from_json(
        args.checkpoint,
        vocabulary_size=2,
        separate_output_embeddings=True,
    )
    optimizer = GradientDescent(model.parameters(), learning_rate=args.learning_rate)
    trainer = Trainer(model, CrossEntropyLoss(), optimizer)
    examples = AlternatingSequenceDataset(args.sequence_length).examples()
    history = trainer.fit(examples, epochs=args.epochs)

    for result in history:
        if result.epoch == 1 or result.epoch % max(1, args.epochs // 10) == 0:
            print(f"epoch={result.epoch:4d} loss={result.loss:.6f}")

    JsonCheckpoint.save(model, args.output)
    print("saved:", args.output)


if __name__ == "__main__":
    main()
