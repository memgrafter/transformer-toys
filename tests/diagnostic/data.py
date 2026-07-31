"""Data, baseline, and causality diagnostics."""

from __future__ import annotations

import argparse

from core import LOSS, examples, training_model


def report_lookup_baseline(sequence_length: int) -> None:
    counts: dict[int, dict[int, int]] = {}
    for example in examples(sequence_length):
        for current, target in zip(example.input_ids, example.target_ids):
            counts.setdefault(current, {})[target] = counts.setdefault(current, {}).get(target, 0) + 1
    correct = 0
    total = 0
    print("[lookup baseline]")
    for current, target_counts in sorted(counts.items()):
        prediction = max(target_counts, key=target_counts.get)
        print(f"input={current} target_counts={target_counts} prediction={prediction}")
        correct += max(target_counts.values())
        total += sum(target_counts.values())
    print(f"baseline_accuracy={correct / total:.6f}")


def report_full_model_causality() -> None:
    model = training_model()
    prefix_logits = model.forward([0])[0]
    extended_logits = model.forward([0, 1])[0]
    difference = max(
        abs(left.data - right.data)
        for left, right in zip(prefix_logits, extended_logits)
    )
    print("[full-model causality]")
    print(f"prefix_logit_difference={difference:.6e} causal={difference < 1e-10}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-length", type=int, default=5)
    args = parser.parse_args()
    report_lookup_baseline(args.sequence_length)
    report_full_model_causality()


if __name__ == "__main__":
    main()
