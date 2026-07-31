"""Gradient-descent update diagnostics."""

from __future__ import annotations

import argparse

from core import LOSS, examples, loss_for, parameter_groups, training_model, vector_norm
from gradient_descent import GradientDescent


def report_update_direction(sequence_length: int) -> None:
    model = training_model()
    example = examples(sequence_length)[0]
    optimizer = GradientDescent(model.parameters(), learning_rate=1e-4)
    before = loss_for(model, example)
    loss = LOSS.compute(model.forward(example.input_ids), example.target_ids)
    loss.backward()
    optimizer.step()
    after = loss_for(model, example)
    print("[update direction]")
    print(f"loss_before={before:.10f} loss_after={after:.10f} decreased={after < before}")


def report_update_magnitudes(sequence_length: int) -> None:
    model = training_model()
    example = examples(sequence_length)[0]
    before = {name: vector_norm(values) for name, values in parameter_groups(model).items()}
    loss = LOSS.compute(model.forward(example.input_ids), example.target_ids)
    loss.backward()
    GradientDescent(model.parameters(), learning_rate=0.01).step()
    print("[parameter updates]")
    for name, values in parameter_groups(model).items():
        print(f"{name}: norm_before={before[name]:.6e} norm_after={vector_norm(values):.6e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-length", type=int, default=5)
    args = parser.parse_args()
    report_update_direction(args.sequence_length)
    report_update_magnitudes(args.sequence_length)


if __name__ == "__main__":
    main()
