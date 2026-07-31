"""Training and overfitting diagnostics."""

from __future__ import annotations

import argparse

from core import (
    TrainingExample,
    accuracy,
    examples,
    print_prediction_histogram,
    train_with_trace,
)


def report_training(sequence_length: int) -> None:
    one_position_examples = [
        TrainingExample(input_ids=[0], target_ids=[1]),
        TrainingExample(input_ids=[1], target_ids=[0]),
    ]
    model, history = train_with_trace(one_position_examples, 100, {1, 10, 100})
    print("[one-position training]")
    print(f"initial_loss={history[0]:.6f} final_loss={history[-1]:.6f} accuracy={accuracy(model, one_position_examples):.6f}")
    print_prediction_histogram(model, one_position_examples)

    sequence_examples = examples(sequence_length)
    model, history = train_with_trace([sequence_examples[0]], 1000, {1, 10, 100, 1000})
    print("[sequence overfit]")
    print(f"initial_loss={history[0]:.6f} final_loss={history[-1]:.6f} accuracy={accuracy(model, [sequence_examples[0]]):.6f}")
    print_prediction_histogram(model, [sequence_examples[0]])

    model, history = train_with_trace([sequence_examples[1]], 100, {100})
    print("[second-phase overfit]")
    print(f"initial_loss={history[0]:.6f} final_loss={history[-1]:.6f} accuracy={accuracy(model, [sequence_examples[1]]):.6f}")
    print_prediction_histogram(model, [sequence_examples[1]])

    model, history = train_with_trace(sequence_examples, 100, {1, 10, 100})
    print("[both-phases training]")
    print(f"initial_loss={history[0]:.6f} final_loss={history[-1]:.6f} accuracy={accuracy(model, sequence_examples):.6f}")
    print_prediction_histogram(model, sequence_examples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-length", type=int, default=5)
    args = parser.parse_args()
    report_training(args.sequence_length)


if __name__ == "__main__":
    main()
