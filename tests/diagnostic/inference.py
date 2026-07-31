"""Prediction, probability, and class-collapse diagnostics."""

from __future__ import annotations

import argparse

from core import accuracy, examples, print_prediction_histogram, print_snapshot, training_model


def report_initial_predictions(sequence_length: int) -> None:
    model = training_model()
    examples_to_check = examples(sequence_length)
    print("[initial predictions]")
    print(f"accuracy={accuracy(model, examples_to_check):.6f}")
    for index, example in enumerate(examples_to_check):
        print_snapshot(model, example, f"initial phase {index}")
    print("[initial histogram]")
    print_prediction_histogram(model, examples_to_check)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-length", type=int, default=5)
    args = parser.parse_args()
    report_initial_predictions(args.sequence_length)


if __name__ == "__main__":
    main()
