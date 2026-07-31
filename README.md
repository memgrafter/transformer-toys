# Transformer Poking

`transformer.py` is a small decoder-only Transformer written with Python lists.
It has one layer and one attention head. It is designed to be changed, not to
be fast or useful as a language model.

There is no finite list of *all* Transformer extensions. The first part of this
document is a map of the major architecture extensions used in research and
systems through 2026. The appendix separates concerns that are important for a
working model but are not Transformer architecture.

## Baseline in this directory

The toy model contains:

1. Token embeddings.
2. Learned positional embeddings.
3. One causal self-attention head.
4. One residual path.
5. Layer normalization.
6. A two-layer ReLU feed-forward network.
7. Tied input and output embeddings.
8. Greedy next-token generation.

It has random weights and no training code. Its output is useful for inspecting
shapes and changing equations.

# Extending to 3090

The GA102 whitepaper lists tensor core acceleration for:

- TF32, BF16, FP16, INT8, and INT4 Tensor Core acceleration
- Also "binary 1-bit operations"
- Avoid FP64
    - FP64 double precision CUDA executes at 1/64 of FP32 rate; no FP64 tensor core mode
- TF32 tensors can be supplied, cuBLAS may execute with FP32 8-bit range + 10-bit mantissa, FP32 accumulation/output

## Official RTX 3090 peak figures

For the RTX 3090 Founders Edition, NVIDIA list:

FP32 CUDA cores: 35.6 TFLOPS
TF32 Tensor: 35.6 dense / 71 sparse TFLOPS
BF16 Tensor with FP32 accumulate: 71 dense / 142 sparse TFLOPS
FP16 Tensor with FP32 accumulate: 71 dense / 142 sparse TFLOPS
FP16 Tensor with FP16 accumulate: 142 dense / 284 sparse TFLOPS
INT8 Tensor: 284 dense / 568 sparse TOPS
INT4 Tensor: 568 dense / 1,136 sparse TOPS

## Dtype policy for PyTorch training

Use PyTorch with AMP and choose **BF16 mixed precision** as the baseline. This
uses the RTX 3090's BF16 Tensor Cores, has FP32-like exponent range, and avoids
the loss-scaling problems that FP16 can require. This is a dtype decision only;
it does not require changing the JSON representation of the toy model.

| Data | Dtype | Reason |
| --- | --- | --- |
| Token IDs and labels | `torch.int64` | Native `Embedding` and cross-entropy index type |
| Parameters and parameter gradients | `torch.float32` | Stable master weights and updates |
| AdamW state | `torch.float32` | Stable optimizer moments |
| Linear/attention activations | `torch.bfloat16` under `torch.autocast` | Fast Tensor Core math |
| LayerNorm, softmax, attention reductions, logits, and loss | `torch.float32` | Prevent overflow and loss of small values |

The JSON numbers are Python floating-point values after loading. When a future
PyTorch loader reads them, create parameters as `float32`; autocast should make
eligible operations use BF16 without permanently converting the master
parameters. Keep token IDs as integer tensors and never cast them to BF16.

Do not use FP64 for model data. Do not use INT8 or INT4 during training; reserve
those for later inference quantization. FP16 is a valid speed benchmark on the
3090, but use it only as an explicit alternative with `GradScaler`. BF16 is the
preferred default for a small training loop because it is simpler and more
robust. If a run uses full FP32 instead, enable Ampere TF32 matmul in PyTorch;
TF32 is a matmul compute mode, not a stored parameter dtype.


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

1. Add a second block and print the shape after every operation.
2. Replace one head with two heads.
3. Add a real padding/mask argument.
4. Replace learned positions with sinusoidal positions or RoPE.
5. Replace ReLU with GELU or a gated MLP.
6. Add key/value caching to `generate`.
7. Add a configurable temperature and sampling method.
8. Add a small training loop only after the forward pass is clear.

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

The current file has none of these. Its weights are initialized randomly.

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
