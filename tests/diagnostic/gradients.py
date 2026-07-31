"""Gradient, graph, and numerical-stability diagnostics."""

from __future__ import annotations

import argparse
import math

from core import (
    LOSS,
    examples,
    graph_size,
    gradient_norm,
    parameter_groups,
    training_model,
)


def report_gradient_statistics(sequence_length: int) -> None:
    model = training_model()
    example = examples(sequence_length)[0]
    loss = LOSS.compute(model.forward(example.input_ids), example.target_ids)
    print("[gradient statistics]")
    print(f"loss={loss.data:.6f} graph_nodes={graph_size(loss)}")
    loss.backward()
    for name, values in parameter_groups(model).items():
        zero_count = sum(value.grad == 0.0 for value in values)
        finite = all(math.isfinite(value.grad) for value in values)
        print(
            f"{name}: norm={gradient_norm(values):.6e} "
            f"min={min(value.grad for value in values):.6e} "
            f"max={max(value.grad for value in values):.6e} "
            f"zero={zero_count}/{len(values)} finite={finite}"
        )


def report_finite_difference_gradients(sequence_length: int) -> None:
    print("[finite-difference gradients]")
    example = examples(sequence_length)[0]
    locations = (
        ("token_embedding", 0, 0),
        ("query_weight", 0, 0),
        ("key_weight", 0, 0),
        ("value_weight", 0, 0),
        ("first_feedforward_weight", 0, 0),
        ("output_bias", 0),
        ("output_embedding", 0, 0),
    )
    for location in locations:
        model = training_model()
        loss = LOSS.compute(model.forward(example.input_ids), example.target_ids)
        loss.backward()
        container = getattr(model, location[0])
        parameter = container[location[1]] if len(location) == 2 else container[location[1]][location[2]]
        analytical = parameter.grad
        original = parameter.data
        step = 1e-5
        parameter.data = original + step
        plus = LOSS.compute(model.forward(example.input_ids), example.target_ids).data
        parameter.data = original - step
        minus = LOSS.compute(model.forward(example.input_ids), example.target_ids).data
        parameter.data = original
        numerical = (plus - minus) / (2.0 * step)
        print(
            f"{'.'.join(str(item) for item in location)}: "
            f"analytical={analytical:.6e} numerical={numerical:.6e} "
            f"error={abs(analytical - numerical):.6e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-length", type=int, default=5)
    args = parser.parse_args()
    report_gradient_statistics(args.sequence_length)
    report_finite_difference_gradients(args.sequence_length)


if __name__ == "__main__":
    main()
