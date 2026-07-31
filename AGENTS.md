# Transformer Poking Project Guide

## Objective

Build a small, transparent Transformer that can be changed and understood.
Use it as a learning and research instrument, not as a production language
model. The project should demonstrate modern Transformer ideas with simple
code, explicit tests, and observable training behavior.

The main priorities are:

1. Understand the equations.
2. Keep the forward pass inspectable.
3. Train small examples end to end.
4. Add one architectural idea at a time.
5. Preserve tests that prove the idea works.
6. Move to PyTorch or accelerator code only when the educational implementation
   has reached its limit.

Do not optimize for scale before the model behavior is understood.

## Current status

### Reference implementation

`transformer.py` is the readable forward-pass reference. It uses Python lists
and ordinary numbers. It has one Transformer block and can load the original
JSON checkpoint. It now accepts an optional `output_embedding` field so it can
inspect checkpoints with separate input and output embeddings.

### Pure-Python training implementation

The current trainable path is deliberately independent of PyTorch:

- `autodiff.py` contains scalar reverse-mode autodiff.
- `gradient_descent.py` contains plain gradient descent.
- `differentiable_ops.py` contains readable vector, matrix, normalization,
  softmax, and causal-attention operations.
- `trainable_transformer.py` contains the autodiff-backed model.
- `losses.py` contains cross-entropy loss.
- `datasets.py` contains the alternating binary dataset.
- `trainer.py` performs full-batch updates.
- `checkpoint.py` writes and reads JSON checkpoints.
- `train_binary_manual.py` is the executable training entry point.

The binary training model currently uses:

- Vocabulary size: 2
- Separate input and output embeddings
- One Transformer block
- Existing learned positional embeddings
- Full-batch gradient descent
- Python scalar arithmetic
- Human-readable JSON checkpoints

The original JSON checkpoint remains a six-token reference checkpoint. The
manual binary trainer loads the first two token rows and creates a separate
output embedding matrix.

### Observed behavior

The model now learns all of the current toy tasks:

- One-position mapping: 100% accuracy
- One-sequence overfit: 100% accuracy
- Both alternating phases: 100% accuracy

The model first failed because the six-token tied-output setup and sequential
example updates allowed class collapse. Reducing the vocabulary, separating
input/output embeddings, and accumulating a full batch before updating fixed
that training experiment.

### Tests and diagnostics

Unit tests are in `tests/`. Integration tests are in
`test/integration/`. Diagnostic output is in
`tests/diagnostic/training_diagnostics.py`.

Run the core tests:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

Run the complete training integration test:

```bash
python test/integration/test_binary_training.py
```

Run the diagnostic report:

```bash
python tests/diagnostic/training_diagnostics.py
```

The diagnostics cover data, baseline behavior, forward parity, causality,
finite-difference gradients, gradient statistics, update direction, parameter
updates, graph size, probability reports, confusion counts, and overfitting.

## Next steps

### Immediate

1. Keep the binary training result as a regression test.
2. Add an inference command that loads a JSON checkpoint and prints logits,
   probabilities, and predicted tokens.
3. Add explicit checkpoint validation for non-finite numbers and malformed
   dimensions.
4. Add a small training configuration object for learning rate, epochs,
   sequence length, and optimizer choice.
5. Add a gradient-accumulation test that proves one full-batch update equals the
   intended averaged loss update.
6. Compare several learning rates before adding a more complex optimizer.

### Training machinery

1. Add momentum SGD.
2. Add AdamW with visible first-moment and second-moment state.
3. Add a learning-rate schedule and warmup.
4. Add gradient clipping and report when clipping occurs.
5. Add held-out examples and report loss separately from training loss.
6. Add checkpoint resume, including optimizer state.

Keep each optimizer in its own class. Do not hide the update rule.

### Architecture foundations

1. Stack a second identical block.
2. Make block count configurable.
3. Add multiple attention heads.
4. Add learned normalization scale and bias.
5. Compare LayerNorm and RMSNorm.
6. Compare learned positions, sinusoidal positions, and RoPE.
7. Replace ReLU with GELU and then SwiGLU.
8. Add padding masks and packed examples.
9. Add key/value caching for generation.
10. Add sampling instead of greedy decoding.

Each change must have a small test and a clear comparison with the previous
model.

## Big picture

The project has two implementations with different roles:

```text
transformer.py
    Transparent reference equations

trainable_transformer.py
    Transparent autodiff training implementation

Future PyTorch implementation
    Tensor execution, accelerator support, mixed precision, and scale
```

The reference model is not obsolete when a faster implementation is added.
Use it to check shapes, intermediate values, masks, and final logits on small
inputs. A faster backend must prove parity with the reference before it becomes
the preferred training path.

The current pure-Python system is intentionally slow and normally behaves like
Python double-precision arithmetic. It is for learning and toy experiments.
It does not implement the earlier RTX 3090 BF16/FP32 policy. That policy belongs
to the later PyTorch backend.

## Learning roadmap: modern Transformer architectures

All of the following remain Transformer-based. Add them in a controlled order.

### 1. Standard decoder Transformer

Use the current model to learn the complete training path:

- Token embeddings
- Positional representations
- Causal self-attention
- Residual connections
- Layer normalization
- Feed-forward network
- Next-token cross-entropy
- Autoregressive generation

The first goal is not a good language model. The goal is to observe how each
part changes representations and loss.

### 2. Modern decoder block

Move toward common modern decoder designs:

- Pre-normalization
- RMSNorm
- RoPE
- GELU or SwiGLU
- Configurable depth and width
- Separate or tied output embeddings
- Stable residual scaling

Compare each change on the same tiny dataset.

### 3. Attention efficiency variants

Study the relationship between attention structure and memory:

- Multi-head attention
- Multi-query attention
- Grouped-query attention
- Key/value caching
- Sliding-window attention
- Block-sparse attention
- Flash-style tiled attention

Flash-style attention changes the implementation and memory pattern while
preserving the attention result. It should first be studied with a small parity
test against ordinary attention.

### 4. Context and position variants

Study how a Transformer represents order and long context:

- Learned absolute positions
- Sinusoidal positions
- RoPE
- Scaled RoPE
- ALiBi
- Long-context interpolation
- Prefix-LM masks
- Document-boundary masks

Use targeted synthetic tasks so position behavior is measurable.

### 5. Capacity and routing variants

After stacked dense blocks work, study parameter capacity:

- Wider feed-forward networks
- SwiGLU parameter balancing
- Mixture-of-Experts routing
- Top-1 and top-2 expert selection
- Expert load balancing
- Shared and routed experts

MoE is still Transformer-based, but it adds routing and capacity behavior that
must be tested separately from attention.

### 6. Efficient adaptation

Once a small pretrained checkpoint exists, study adaptation:

- LoRA
- Prefix tuning
- Prompt tuning
- Adapter blocks
- Partial-layer freezing

Measure trainable parameter count, loss, and output behavior.

### 7. Compression and serving

Only after the model and training behavior are understood:

- FP32 master parameters with mixed-precision compute
- BF16 and FP16 comparisons
- Weight-only INT8 and INT4 inference
- Safetensors checkpoints
- KV-cache memory accounting
- Batch scheduling
- Compilation and fused kernels

These are execution and serving topics, not replacements for understanding the
Transformer equations.

## Working rules

- Keep the current reference implementation readable.
- Prefer one responsibility per class.
- Use explicit names and dimensions.
- Test failure cases, not only successful training.
- Keep diagnostics when a bug is fixed; convert useful diagnostics into tests.
- Do not call a lower loss proof of understanding. Check predictions and held-out
  behavior.
- Do not introduce PyTorch merely to hide a bug in the educational model.
- When a PyTorch backend is added, compare it with the pure-Python model on tiny
  inputs before using it for larger training.
