#!/usr/bin/env bash
set -euo pipefail

DIAGNOSTIC_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SEQUENCE_LENGTH=${1:-15}

for diagnostic in data gradients updates inference training; do
    printf '\n===== %s (sequence length %s) =====\n' "$diagnostic" "$SEQUENCE_LENGTH"
    python "$DIAGNOSTIC_DIR/$diagnostic.py" \
        --sequence-length "$SEQUENCE_LENGTH"
done
