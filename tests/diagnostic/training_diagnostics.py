"""Run all focused training diagnostics."""

from __future__ import annotations

import argparse

from data import report_full_model_causality, report_lookup_baseline
from gradients import report_finite_difference_gradients, report_gradient_statistics
from inference import report_initial_predictions
from training import report_training
from updates import report_update_direction, report_update_magnitudes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-length", type=int, default=5)
    args = parser.parse_args()
    print("[data]")
    report_lookup_baseline(args.sequence_length)
    report_full_model_causality()
    print("[inference]")
    report_initial_predictions(args.sequence_length)
    print("[gradients]")
    report_gradient_statistics(args.sequence_length)
    report_finite_difference_gradients(args.sequence_length)
    print("[updates]")
    report_update_direction(args.sequence_length)
    report_update_magnitudes(args.sequence_length)
    print("[training]")
    report_training(args.sequence_length)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
