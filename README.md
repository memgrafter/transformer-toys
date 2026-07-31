# Transformer Poking

This repository is a small, inspectable decoder-only Transformer. It is for
learning and controlled experiments, not for useful language generation or
production scale.

## Current model

`transformer.py` is the readable reference implementation. It has:

1. Token embeddings.
2. Learned positional embeddings.
3. One causal self-attention head.
4. Residual connections and layer normalization.
5. A two-layer ReLU feed-forward network.
6. Greedy next-token prediction.

The reference model loads `transformer.json`, which contains a six-token toy
checkpoint. It also understands an optional `output_embedding` field for
checkpoints with separate input and output embeddings.

## Current training experiment

The active learning path is pure Python. It does not require PyTorch:

```bash
python train_binary_manual.py --epochs 100
```

The trainer loads the reference checkpoint, then creates a binary training
model with:

- Two token classes: `0` and `1`.
- Separate input and output embeddings.
- One Transformer block.
- Learned positional embeddings.
- Full-batch gradient descent.
- Human-readable JSON output.

The model learns the alternating rule:

```text
0 1 0 1 ...  ->  1 0 1 0 ...
1 0 1 0 ...  ->  0 1 0 1 ...
```

The implementation is intentionally scalar and slow. It exists so that the
forward pass, gradients, updates, and checkpoint values remain visible.

## Tests and diagnostics

Run the unit tests:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

Run the real training integration test:

```bash
python test/integration/test_binary_training.py
```

Run all focused diagnostics for a sequence length of 15:

```bash
tests/diagnostic/test.sh 15
```

Individual diagnostics are in `tests/diagnostic/`:

- `data.py`: dataset baseline and full-model causality.
- `gradients.py`: graph size, gradient statistics, and finite differences.
- `updates.py`: update direction and parameter movement.
- `inference.py`: probabilities, predictions, and class histograms.
- `training.py`: one-position, sequence-overfit, and both-phase training.

The diagnostic sequence length is a training length. A held-out length is not
used during training and is evaluated later to test generalization beyond the
training positions.

## Checkpoint formats

The normal learning checkpoint is JSON because it is easy to inspect. Trained
binary checkpoints include `output_embedding` and have vocabulary size 2.
`transformer.py` can load these checkpoints.

Safetensors and PyTorch conversion utilities remain deferred until the
transparent training path has more experiments. When that work starts, use
FP32 master parameters and accelerator-appropriate mixed precision rather than
changing the educational model first.

# Transformer extensions

## 1. Attention width: multiple heads

The current model calculates one set of queries, keys, and values. Multi-head
attention splits the model width into smaller pieces:

```text
width = 8, heads = 2

head 0: Q/K/V width 4 -> attention result width 4
head 1: Q/K/V width 4 -> attention result width 4

concatenate results -> width 8 -> output projection
```

Each head can focus on a different relationship. Multi-query attention (MQA)
shares keys and values across heads but keeps separate queries. Grouped-query
attention (GQA) shares keys and values within groups of query heads. These
reduce key/value-cache memory during generation. Latent-attention designs,
such as multi-head latent attention (MLA), compress or reconstruct key/value
information instead of storing the ordinary cache directly.

Useful experiments:

- Start with two heads and assert that `model_width % head_count == 0`.
- Compare ordinary multi-head attention, MQA, and GQA.
- Print each head's attention probabilities.
- Remove the output projection and observe what changes.

## 2. Attention patterns and sparsity

Full attention compares every position with every earlier position. For a
sequence of length `n`, this costs roughly `n * n` score entries.

Alternatives include:

- **Sliding-window attention**: attend only to nearby tokens.
- **Local plus global attention**: local windows plus selected global tokens.
- **Block-sparse attention**: calculate only selected blocks.
- **Dilated attention**: skip at a regular interval.
- **Prefix or bidirectional attention**: special patterns for encoded context.
- **Cross-attention**: queries attend to a separate sequence of keys and values.
- **Retrieval attention**: attend to retrieved chunks or memories.
- **Linear attention**: rearrange operations to avoid an explicit score matrix.
- **Kernel or feature-map attention**: approximate the softmax kernel.
- **Flash-style attention**: compute exact attention in memory-efficient tiles.
- **Ring or sequence-parallel attention**: distribute sequence blocks across devices.

Sparse and approximate methods change either the connectivity pattern or the
way scores are computed. Tiled implementations can preserve the same math
while changing memory use.

## 3. Position information

The toy model adds a learned vector for each absolute position. Other choices
include:

- **Sinusoidal positions**: fixed sine and cosine vectors.
- **Learned absolute positions**: the current approach.
- **Relative position bias**: add a distance-dependent value to attention scores.
- **Relative key/value positions**: inject distance into attention content.
- **RoPE**: rotate query and key pairs by their positions.
- **ALiBi**: add a head-specific distance penalty to scores.
- **Interpolated or scaled RoPE**: adapt rotation frequencies for longer context.
- **YaRN-style scaling**: extend RoPE context with frequency and attention scaling.
- **Position interpolation**: map a long context into a trained position range.
- **Learned position extrapolation**: train the position rule for longer inputs.
- **No explicit position encoding**: rely on another mechanism, such as recurrence.

Position changes are often the first place where a model's stated context
length and its useful context length differ.

## 4. Transformer block layout

The toy block uses a simplified normalization and residual arrangement. Common
variants include:

- **Post-norm**: normalize after each residual addition.
- **Pre-norm**: normalize before attention and feed-forward sublayers.
- **RMSNorm**: normalize by root mean square without subtracting the mean.
- **NormFormer-style extra normalization**: normalize selected intermediate values.
- **DeepNorm-style residual scaling**: stabilize very deep stacks.
- **Gated residual paths**: learn how much of a branch to add.
- **Parallel attention and feed-forward branches**: calculate both from one input.
- **Convolutional or state-space side branches**: add local or recurrent mixing.

A normal Transformer block usually has two sublayers:

```text
x = x + attention(normalize(x))
x = x + feed_forward(normalize(x))
```

## 5. Feed-forward networks

The toy feed-forward network is `linear -> ReLU -> linear`. Common variants
include:

- **GELU** and **SwiGLU** activations.
- **GEGLU** and **ReGLU** gated activations.
- **Three-matrix gated MLPs**: one branch gates another before projection.
- **Mixture-of-Experts (MoE)**: route each token to a few expert MLPs.
- **Switch-style routing**: route each token to one expert.
- **Expert capacity limits and token dropping**.
- **Shared experts plus routed experts**.
- **Mixture-of-depths**: skip some blocks for selected tokens.
- **Low-rank or factorized MLP weights**.
- **Convolutional MLPs**: add local mixing between projections.

MoE increases parameter count without applying every parameter to every token.
It introduces routing, load balancing, capacity, and communication problems.

## 6. Depth and connectivity

The example has one block. A normal model stacks many blocks. Extensions include:

- Encoder-only stacks, such as bidirectional masked encoders.
- Decoder-only causal stacks, like this example.
- Encoder-decoder stacks with cross-attention.
- Prefix language models with mixed attention masks.
- Universal or recurrent Transformers that reuse a block.
- Layer dropping and stochastic depth.
- Skip connections across several layers.
- Early exit from the stack.
- Adaptive computation time.
- Block sharing or cross-layer parameter sharing.
- Deep equilibrium or iterative Transformer blocks.

## 7. Input and output representations

The toy model expects integer token IDs. Architecture-level alternatives include:

- Byte, character, word, subword, or byte-level token inputs.
- Learned input/output embedding tying or separate output weights.
- Continuous vectors from an encoder instead of token IDs.
- Image patches, video patches, audio frames, point sets, or graph tokens.
- A special latent bottleneck for variable-size inputs.
- Perceiver-style resampling from many input tokens into fewer latent tokens.
- Multimodal adapters and modality-specific input projections.
- Cross-attention between text and image, audio, video, or sensor streams.
- Any-resolution patch packing or token merging for visual inputs.

## 8. Memory and recurrence

A normal forward pass forgets the previous call. Extensions add memory:

- Cached keys and values during autoregressive decoding.
- Recurrent memory tokens.
- Compressed memory of old hidden states.
- Segment-level recurrence.
- External key/value memory.
- Retrieval-augmented attention.
- Persistent learned memory slots.
- State compression or recurrent state updates.

The key/value cache is not a new model layer, but it is essential to make
causal generation affordable.

## 9. Parameter and arithmetic efficiency

Ways to reduce memory or compute include:

- Weight sharing and factorized matrices.
- Low-rank adapters (LoRA) and other parameter-efficient adapters.
- Prefix tuning, prompt tuning, and soft prompts.
- Quantized weights or activations: 8-bit, 4-bit, and lower formats.
- Weight-only, activation-aware, and mixed-precision quantization.
- Pruning individual weights, heads, neurons, layers, or tokens.
- Knowledge distillation into a smaller Transformer.
- Token merging, pooling, or pruning.
- Sparse expert routing.
- Kernel fusion and tiled attention.
- Parallelism over data, tensor, pipeline, sequence, and experts.

Some items here are implementation techniques rather than changes to the
mathematical architecture. They still change how a Transformer can be used.

## 10. Attention masks

The toy model applies causality by looping only over positions at or before
the current position. Other masks support different tasks:

- Full bidirectional attention.
- Causal attention.
- Padding masks for unequal sequence lengths.
- Prefix-LM masks.
- Document or segment boundaries.
- Block-diagonal packed examples.
- Sliding windows with global tokens.
- Retrieval and memory masks.
- Structured masks for trees, graphs, or layouts.
- Multimodal masks that control which modalities can communicate.

A mask is part of the model's connectivity. It is not the same thing as
zeroing a token embedding.

## 11. Objectives and behavior-shaped variants

The block can be trained for different prediction relationships:

- Next-token prediction.
- Masked-token prediction.
- Span corruption.
- Denoising sequence-to-sequence prediction.
- Permuted or replaced-token prediction.
- Contrastive representation learning.
- Prefix completion.
- Multiple-token or speculative draft prediction.
- Classification or regression from a pooled representation.
- Preference or reward-conditioned generation.

The objective is not a Transformer layer, but it determines what the same
architecture learns to represent.

## 12. Retrieval, tools, and external actions

A Transformer can be connected to systems outside its layers:

- Retrieve documents before or during generation.
- Insert retrieved text as context tokens.
- Attend to a separate memory store.
- Call tools and feed results back as new tokens.
- Use a planner, verifier, or reranker around the model.
- Add a multimodal perception encoder.
- Use a recurrent controller around repeated model calls.

These are system extensions. They do not automatically make the attention
layers themselves different.

## 13. Hybrid and adjacent architectures

Several modern designs combine Transformer blocks with other sequence mixers:

- Convolution plus attention.
- Recurrent memory plus attention.
- State-space or selective-scan layers plus attention.
- Gated linear recurrent units plus attention.
- Hybrids that use local mixers in lower layers and attention in upper layers.
- Neural memory or external memory modules.
- Graph, diffusion, or energy-based modules around Transformer components.

A model can still be called a Transformer hybrid even when not every block is a
standard attention block. State-space-only, convolution-only, and recurrent-only
models belong in the appendix's adjacent-model category, not in the core list.

## Suggested order for modifying this toy

1. Train binary sequences at lengths 5, 15, and 31.
2. Evaluate a held-out sequence length without updating the weights.
3. Add an inference command that prints probabilities and predictions.
4. Compare plain gradient descent with momentum and AdamW.
5. Add a second block and inspect its intermediate values.
6. Replace one head with two heads.
7. Add a real padding/mask argument.
8. Replace learned positions with sinusoidal positions or RoPE.
9. Replace ReLU with GELU or a gated MLP.
10. Add key/value caching to `generate`.
11. Add configurable temperature and sampling.

# Appendix: non-Transformer concerns

These are needed to build a usable language-model system, but they are not
Transformer architecture.

## Data and tokenization

- Collect, license, filter, deduplicate, and split data.
- Normalize text and decide how to handle unusual bytes.
- Train or choose a tokenizer.
- Map text to token IDs and back to text.
- Build training examples and next-token labels.
- Handle sequence packing, padding, truncation, and document boundaries.
- Prevent evaluation or test data from leaking into training.

## Training machinery

- Cross-entropy or another loss function.
- Backpropagation and automatic differentiation.
- SGD, Adam, AdamW, or another optimizer.
- Learning-rate schedule and warmup.
- Gradient clipping and accumulation.
- Weight decay.
- Mixed precision and loss scaling.
- Checkpoint save, restore, and resume.
- Distributed training and fault recovery.
- Random seeds and experiment configuration.

The repository now contains a small autodiff engine, gradient descent,
cross-entropy training, JSON checkpointing, and integration diagnostics. More
advanced optimizers, schedules, mixed precision, and distributed training are
future exercises.

## Generation and decoding

- Temperature scaling.
- Random sampling.
- Top-k and nucleus (top-p) sampling.
- Repetition penalties.
- Stop-token handling.
- Maximum output length.
- Key/value caching.
- Batch scheduling.
- Speculative decoding.
- Constrained or grammar-based decoding.
- Streaming output.

Greedy decoding is the only decoding method in the example.

## Evaluation

- Held-out loss and perplexity.
- Exact-match or task-specific accuracy.
- Long-context retrieval tests.
- Robustness and adversarial tests.
- Calibration and uncertainty checks.
- Bias, toxicity, privacy, and memorization checks.
- Latency, throughput, memory, and cost measurements.
- Human or application-level evaluation.

## Serving and operations

- Model serialization and versioning.
- CPU/GPU/accelerator kernels.
- Quantization and compilation.
- Request batching and queueing.
- Memory limits and cache eviction.
- Authentication, rate limits, and logging.
- Monitoring, rollback, and incident handling.
- Hardware placement and network communication.

## Product behavior

- System and developer instructions.
- Prompt templates.
- Conversation history management.
- Retrieval and document ranking.
- Tool schemas and permission checks.
- Output validation.
- Safety policies and refusal behavior.
- User feedback and quality measurement.

## Adjacent model families

These can replace or complement Transformers, but they are not Transformer
extensions by themselves:

- RNNs, LSTMs, and GRUs.
- Convolutional sequence models.
- State-space models.
- Selective-scan or recurrent sequence mixers.
- Linear-attention-only models.
- Diffusion models.
- Graph neural networks.
- Classical statistical language models.

## What this toy intentionally omits

The example is missing training, batching, tokenization, checkpointing,
production numerical stability, performance optimization, and most of the
extensions listed above. That is deliberate: each omission leaves a small
piece that can be implemented and inspected by hand.
