# Training Outcomes — 2026-07-31

## Purpose

The goal was to make the toy Transformer trainable without PyTorch, then prove
that it can learn a small binary next-token rule while keeping the computation
readable and debuggable.

## Starting point

The original model was a one-block, one-head Transformer in `transformer.py`.
It used Python lists, loaded six-token JSON weights, and had no autodiff or
training loop.

The first training attempt used:

- Six output tokens, although the task was binary.
- Tied input and output embeddings.
- Sequential updates for the two training examples.
- Plain gradient descent.

That version reduced loss but collapsed to one output token. Accuracy was 0.5.

## Implemented training stack

The pure-Python training path now contains:

- `autodiff.py` — scalar reverse-mode autodiff.
- `gradient_descent.py` — plain gradient descent.
- `differentiable_ops.py` — vector, matrix, normalization, softmax, and causal
  attention operations.
- `trainable_transformer.py` — autodiff-backed Transformer.
- `losses.py` — cross-entropy loss.
- `datasets.py` — alternating binary examples.
- `trainer.py` — full-batch training.
- `checkpoint.py` — JSON checkpoint persistence.
- `train_binary_manual.py` — manual training entry point.

The binary training configuration now uses:

- Vocabulary size: 2.
- Separate input and output embeddings.
- One Transformer block.
- Learned positional embeddings.
- Full-batch updates.
- Plain gradient descent with learning rate `0.01`.
- Human-readable JSON checkpoints.

## Concrete training results

### 100-epoch manual run

Command:

```bash
python train_binary_manual.py --epochs 100 --output /tmp/binary-trained-manual.json
```

Loss progression:

```text
epoch 1:   0.799857
epoch 10:  0.720729
epoch 20:  0.698139
epoch 30:  0.689383
epoch 40:  0.673574
epoch 50:  0.624947
epoch 60:  0.517231
epoch 70:  0.384358
epoch 80:  0.279722
epoch 90:  0.210379
epoch 100: 0.165117
```

The checkpoint was written successfully.

### Sequence-length-15 diagnostic

Command:

```bash
tests/diagnostic/test.sh 15
```

Results from the training diagnostics:

| Experiment | Initial loss | Final loss | Accuracy |
| --- | ---: | ---: | ---: |
| One-position mapping | 0.837402 | 0.109612 | 1.000000 |
| First-phase sequence overfit | 0.792815 | 0.006603 | 1.000000 |
| Second-phase sequence overfit | 0.806605 | 0.097526 | 1.000000 |
| Both alternating phases | 0.799710 | 0.167167 | 1.000000 |

The both-phase prediction histogram was:

```text
predicted_tokens: {0: 15, 1: 15}
confusion target,prediction: {(0, 0): 15, (1, 1): 15}
```

## Correctness evidence

The diagnostics and tests verified:

- The lookup baseline reaches 100%.
- The autodiff forward pass matches the original reference model.
- Full-model attention is causal.
- Selected analytical gradients match finite differences.
- Gradients are finite.
- Gradient-descent updates reduce loss for a small step.
- Checkpoint save/load preserves predictions.
- The computation graph is inspectable. A length-5 example produced 9,329
  graph nodes.
- Repeated training runs are deterministic after replacing unordered autodiff
  parent storage with ordered tuples.

The earlier deterministic-run failure was caused by `frozenset` traversal order.
Floating-point additions occurred in different orders between processes. The
ordered parent tuple fixed that issue.

## Diagnostic layout

Diagnostics are now split into focused files:

```text
tests/diagnostic/core.py
 tests/diagnostic/data.py
 tests/diagnostic/gradients.py
 tests/diagnostic/updates.py
 tests/diagnostic/inference.py
 tests/diagnostic/training.py
 tests/diagnostic/training_diagnostics.py
 tests/diagnostic/test.sh
```

Run one diagnostic directly:

```bash
python tests/diagnostic/gradients.py --sequence-length 15
```

Run all focused diagnostics:

```bash
tests/diagnostic/test.sh 15
```

## Test results

The latest targeted test runs reported:

```text
17 unit tests passed
2 integration tests passed
```

The integration tests execute the real `train_binary_manual.py` process for
100 and 500 epochs, validate the generated binary checkpoint, check decreasing
loss, and check both-phase accuracy.

## Current limitations

The model can now learn the toy rule, but this does not prove general language
model capability. It trains on a tiny synthetic dataset and uses scalar Python
arithmetic. It is intentionally slow and does not implement accelerator dtype
behavior such as BF16 or FP16.

The learned positional embeddings are still present. The current length-15
result tests training accuracy, not held-out length generalization. A held-out
length experiment has not yet been run.

The PyTorch scripts remain in `scripts/` for later work, but the PyTorch launcher
was removed. The current training path is `train_binary_manual.py`.

## Next comparison

Train on shorter sequences and evaluate without updating on a longer sequence:

```text
Train:      lengths 5 and 7
Evaluate:   length 15
```

This will test whether the model learned the alternating rule or memorized
training positions. Because the model uses learned positional embeddings,
longer unseen positions may fail even if the token rule is correct.

After that comparison, test a position-encoding change such as sinusoidal
positions or RoPE before increasing model depth.
