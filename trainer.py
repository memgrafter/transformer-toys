"""Training orchestration for the transparent toy model."""

from __future__ import annotations

from dataclasses import dataclass

from gradient_descent import GradientDescent
from losses import CrossEntropyLoss
from datasets import TrainingExample
from trainable_transformer import TrainableTransformer


@dataclass(frozen=True)
class TrainingResult:
    epoch: int
    loss: float


class Trainer:
    """Run forward, loss, backward, and gradient-descent steps."""

    def __init__(
        self,
        model: TrainableTransformer,
        loss_function: CrossEntropyLoss,
        optimizer: GradientDescent,
    ) -> None:
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        expected = {id(parameter) for parameter in model.parameters()}
        actual = {id(parameter) for parameter in optimizer.parameters}
        if expected != actual:
            raise ValueError("optimizer parameters must match model parameters")

    def train_step(self, example: TrainingExample) -> float:
        return self.train_batch([example])

    def train_batch(self, examples: list[TrainingExample]) -> float:
        if not examples:
            raise ValueError("at least one training example is required")
        self.optimizer.zero_grad()
        losses = [
            self.loss_function.compute(
                self.model.forward(example.input_ids), example.target_ids
            )
            for example in examples
        ]
        loss = sum(losses[1:], losses[0]) / len(losses)
        loss.backward()
        self.optimizer.step()
        return loss.data

    def fit(self, examples: list[TrainingExample], epochs: int) -> list[TrainingResult]:
        if not examples:
            raise ValueError("at least one training example is required")
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        history: list[TrainingResult] = []
        for epoch in range(1, epochs + 1):
            loss = self.train_batch(examples)
            history.append(TrainingResult(epoch, loss))
        return history
