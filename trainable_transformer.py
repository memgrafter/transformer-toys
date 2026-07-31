"""Autodiff-backed version of the transparent one-block Transformer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from autodiff import Value
from differentiable_ops import DifferentiableOps, Matrix, Vector
from transformer_config import TransformerConfig


class TrainableTransformer:
    """A one-block Transformer whose parameters are autodiff Values."""

    MATRIX_FIELDS = (
        "token_embedding",
        "position_embedding",
        "query_weight",
        "key_weight",
        "value_weight",
        "output_weight",
        "first_feedforward_weight",
        "second_feedforward_weight",
        "output_embedding",
    )
    VECTOR_FIELDS = (
        "first_feedforward_bias",
        "second_feedforward_bias",
        "output_bias",
    )

    def __init__(
        self,
        config: TransformerConfig,
        state: dict[str, Any],
        separate_output_embeddings: bool = False,
    ) -> None:
        self.config = config
        self.random_state = state.get("random_state", 0)
        self.separate_output_embeddings = separate_output_embeddings or "output_embedding" in state
        self.token_embedding = self._matrix(state["token_embedding"], "token_embedding")
        self.position_embedding = self._matrix(state["position_embedding"], "position_embedding")
        self.query_weight = self._matrix(state["query_weight"], "query_weight")
        self.key_weight = self._matrix(state["key_weight"], "key_weight")
        self.value_weight = self._matrix(state["value_weight"], "value_weight")
        self.output_weight = self._matrix(state["output_weight"], "output_weight")
        self.first_feedforward_weight = self._matrix(
            state["first_feedforward_weight"], "first_feedforward_weight"
        )
        self.second_feedforward_weight = self._matrix(
            state["second_feedforward_weight"], "second_feedforward_weight"
        )
        self.first_feedforward_bias = self._vector(
            state["first_feedforward_bias"], "first_feedforward_bias"
        )
        self.second_feedforward_bias = self._vector(
            state["second_feedforward_bias"], "second_feedforward_bias"
        )
        self.output_bias = self._vector(state["output_bias"], "output_bias")
        self.output_embedding = (
            self._matrix(state["output_embedding"], "output_embedding")
            if self.separate_output_embeddings
            else self.token_embedding
        )
        self._validate_shapes()

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        vocabulary_size: int | None = None,
        separate_output_embeddings: bool = False,
    ) -> TrainableTransformer:
        with Path(path).open(encoding="utf-8") as file:
            source_state = json.load(file)
        if not isinstance(source_state, dict):
            raise ValueError("checkpoint root must be a JSON object")
        state = dict(source_state)
        if vocabulary_size is not None:
            if vocabulary_size <= 0 or vocabulary_size > state["vocabulary_size"]:
                raise ValueError("vocabulary_size must fit the source checkpoint")
            state["vocabulary_size"] = vocabulary_size
            state["token_embedding"] = state["token_embedding"][:vocabulary_size]
            state["output_bias"] = state["output_bias"][:vocabulary_size]
        if separate_output_embeddings:
            state["output_embedding"] = [row[:] for row in state["token_embedding"]]
        return cls(
            TransformerConfig.from_state(state),
            state,
            separate_output_embeddings=separate_output_embeddings,
        )

    @staticmethod
    def _vector(values: list[float], label: str) -> Vector:
        if not isinstance(values, list):
            raise ValueError(f"{label} must be a list")
        return [Value(value, label=f"{label}[{index}]") for index, value in enumerate(values)]

    @staticmethod
    def _matrix(values: list[list[float]], label: str) -> Matrix:
        if not isinstance(values, list) or any(not isinstance(row, list) for row in values):
            raise ValueError(f"{label} must be a matrix")
        return [
            [Value(value, label=f"{label}[{row}][{column}]")
             for column, value in enumerate(values[row])]
            for row in range(len(values))
        ]

    def _validate_shapes(self) -> None:
        config = self.config
        expected = {
            "token_embedding": (config.vocabulary_size, config.model_width),
            "position_embedding": (config.maximum_sequence_length, config.model_width),
            "query_weight": (config.model_width, config.model_width),
            "key_weight": (config.model_width, config.model_width),
            "value_weight": (config.model_width, config.model_width),
            "output_weight": (config.model_width, config.model_width),
            "first_feedforward_weight": (config.model_width, config.feedforward_width),
            "second_feedforward_weight": (config.feedforward_width, config.model_width),
            "first_feedforward_bias": (config.feedforward_width,),
            "second_feedforward_bias": (config.model_width,),
            "output_bias": (config.vocabulary_size,),
        }
        if self.separate_output_embeddings:
            expected["output_embedding"] = (config.vocabulary_size, config.model_width)
        for name, shape in expected.items():
            value = getattr(self, name)
            actual = (len(value), len(value[0])) if isinstance(value[0], list) else (len(value),)
            if actual != shape:
                raise ValueError(f"{name} has shape {actual}, expected {shape}")

    def parameters(self) -> list[Value]:
        """Return each trainable Value once, including tied embeddings once."""
        parameters: list[Value] = []
        seen: set[int] = set()
        fields = (*self.MATRIX_FIELDS, *self.VECTOR_FIELDS)
        if not self.separate_output_embeddings:
            fields = tuple(field for field in fields if field != "output_embedding")
        for field in fields:
            value = getattr(self, field)
            values = (item for row in value for item in row) if value and isinstance(value[0], list) else iter(value)
            for parameter in values:
                if id(parameter) not in seen:
                    seen.add(id(parameter))
                    parameters.append(parameter)
        return parameters

    def forward(self, token_ids: list[int]) -> list[list[Value]]:
        self._validate_tokens(token_ids)
        hidden = [
            DifferentiableOps.add_vectors(
                self.token_embedding[token_id], self.position_embedding[position]
            )
            for position, token_id in enumerate(token_ids)
        ]
        hidden = [DifferentiableOps.layer_norm(vector) for vector in hidden]

        queries = [DifferentiableOps.dot(vector, self.query_weight) for vector in hidden]
        keys = [DifferentiableOps.dot(vector, self.key_weight) for vector in hidden]
        values = [DifferentiableOps.dot(vector, self.value_weight) for vector in hidden]
        contexts = DifferentiableOps.causal_attention(
            queries, keys, values, self.config.model_width
        )

        attended = [
            DifferentiableOps.add_vectors(
                hidden[position],
                DifferentiableOps.dot(context, self.output_weight),
            )
            for position, context in enumerate(contexts)
        ]
        block_output: list[Vector] = []
        for vector in attended:
            normalized = DifferentiableOps.layer_norm(vector)
            feedforward = DifferentiableOps.dot(normalized, self.first_feedforward_weight)
            feedforward = DifferentiableOps.add_vectors(feedforward, self.first_feedforward_bias)
            feedforward = [value.relu() for value in feedforward]
            feedforward = DifferentiableOps.dot(feedforward, self.second_feedforward_weight)
            feedforward = DifferentiableOps.add_vectors(feedforward, self.second_feedforward_bias)
            block_output.append(
                DifferentiableOps.layer_norm(
                    DifferentiableOps.add_vectors(normalized, feedforward)
                )
            )

        return [
            [
                sum((vector[index] * self.output_embedding[token][index]
                     for index in range(self.config.model_width)), Value(0.0))
                + self.output_bias[token]
                for token in range(self.config.vocabulary_size)
            ]
            for vector in block_output
        ]

    def _validate_tokens(self, token_ids: list[int]) -> None:
        if not token_ids:
            raise ValueError("sequence cannot be empty")
        if len(token_ids) > self.config.maximum_sequence_length:
            raise ValueError("sequence is longer than the configured limit")
        for token_id in token_ids:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise TypeError("token IDs must be integers")
            if token_id < 0 or token_id >= self.config.vocabulary_size:
                raise ValueError("token ID is outside the vocabulary")

    def to_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "vocabulary_size": self.config.vocabulary_size,
            "model_width": self.config.model_width,
            "feedforward_width": self.config.feedforward_width,
            "maximum_sequence_length": self.config.maximum_sequence_length,
            "random_state": self.random_state,
        }
        fields = (*self.MATRIX_FIELDS, *self.VECTOR_FIELDS)
        if not self.separate_output_embeddings:
            fields = tuple(field for field in fields if field != "output_embedding")
        for field in fields:
            value = getattr(self, field)
            if value and isinstance(value[0], list):
                state[field] = [[item.data for item in row] for row in value]
            else:
                state[field] = [item.data for item in value]
        return state
