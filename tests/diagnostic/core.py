"""Shared helpers for the focused training diagnostics."""

from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from autodiff import Value  # noqa: E402
from datasets import AlternatingSequenceDataset, TrainingExample  # noqa: E402
from gradient_descent import GradientDescent  # noqa: E402
from losses import CrossEntropyLoss  # noqa: E402
from trainable_transformer import TrainableTransformer  # noqa: E402
from trainer import Trainer  # noqa: E402

CHECKPOINT = REPOSITORY_ROOT / "transformer.json"
LOSS = CrossEntropyLoss()
PARAMETER_FIELDS = (
    "token_embedding",
    "position_embedding",
    "query_weight",
    "key_weight",
    "value_weight",
    "output_weight",
    "first_feedforward_weight",
    "second_feedforward_weight",
    "first_feedforward_bias",
    "second_feedforward_bias",
    "output_bias",
    "output_embedding",
)


def training_model() -> TrainableTransformer:
    return TrainableTransformer.from_json(
        CHECKPOINT,
        vocabulary_size=2,
        separate_output_embeddings=True,
    )


def examples(sequence_length: int = 5) -> list[TrainingExample]:
    return AlternatingSequenceDataset(sequence_length).examples()


def loss_for(model: TrainableTransformer, example: TrainingExample) -> float:
    return LOSS.compute(model.forward(example.input_ids), example.target_ids).data


def probabilities(row: list[Value]) -> list[float]:
    highest = max(value.data for value in row)
    exponentials = [math.exp(value.data - highest) for value in row]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def predictions(model: TrainableTransformer, example: TrainingExample) -> list[int]:
    logits = model.forward(example.input_ids)
    return [max(range(len(row)), key=lambda index: row[index].data) for row in logits]


def accuracy(model: TrainableTransformer, examples_to_check: list[TrainingExample]) -> float:
    correct = 0
    total = 0
    for example in examples_to_check:
        predicted = predictions(model, example)
        correct += sum(actual == target for actual, target in zip(predicted, example.target_ids))
        total += len(example.target_ids)
    return correct / total


def parameter_groups(model: TrainableTransformer) -> dict[str, list[Value]]:
    groups: dict[str, list[Value]] = {}
    for field in PARAMETER_FIELDS:
        if not hasattr(model, field):
            continue
        value = getattr(model, field)
        if value and isinstance(value[0], list):
            groups[field] = [item for row in value for item in row]
        else:
            groups[field] = list(value)
    return groups


def vector_norm(values: list[Value]) -> float:
    return math.sqrt(sum(value.data**2 for value in values))


def gradient_norm(values: list[Value]) -> float:
    return math.sqrt(sum(value.grad**2 for value in values))


def graph_size(loss: Value) -> int:
    nodes: set[int] = set()

    def visit(value: Value) -> None:
        if id(value) in nodes:
            return
        nodes.add(id(value))
        for child in value.children:
            visit(child)

    visit(loss)
    return len(nodes)


def print_snapshot(model: TrainableTransformer, example: TrainingExample, label: str) -> None:
    logits = model.forward(example.input_ids)
    print(f"\n[{label}]")
    print("input:     ", example.input_ids)
    print("target:    ", example.target_ids)
    print("prediction:", [max(range(len(row)), key=lambda index: row[index].data) for row in logits])
    for position, (row, target) in enumerate(zip(logits, example.target_ids)):
        values = probabilities(row)
        print(
            f"position={position} target={target} "
            f"target_probability={values[target]:.6f} "
            f"loss={LOSS.compute([row], [target]).data:.6f} "
            f"probabilities={[round(value, 6) for value in values]}"
        )


def train_with_trace(
    examples_to_train: list[TrainingExample],
    epochs: int,
    snapshots: set[int] | None = None,
) -> tuple[TrainableTransformer, list[float]]:
    model = training_model()
    optimizer = GradientDescent(model.parameters(), learning_rate=0.01)
    trainer = Trainer(model, LOSS, optimizer)
    history: list[float] = []
    snapshots = snapshots or set()
    for epoch in range(1, epochs + 1):
        history.append(trainer.train_batch(examples_to_train))
        if epoch in snapshots:
            print(f"\nepoch={epoch} mean_loss={history[-1]:.6f}")
            for example in examples_to_train:
                print_snapshot(model, example, f"epoch {epoch}")
    return model, history


def print_prediction_histogram(
    model: TrainableTransformer, examples_to_check: list[TrainingExample]
) -> None:
    predicted_counts: Counter[int] = Counter()
    confusion: Counter[tuple[int, int]] = Counter()
    for example in examples_to_check:
        predicted = predictions(model, example)
        predicted_counts.update(predicted)
        confusion.update(zip(example.target_ids, predicted))
    print("predicted_tokens:", dict(sorted(predicted_counts.items())))
    print("confusion target,prediction:", dict(sorted(confusion.items())))
