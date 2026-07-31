"""Readable vector and matrix operations over autodiff Values."""

from __future__ import annotations

import math
from collections.abc import Sequence

from autodiff import Value

Vector = list[Value]
Matrix = list[Vector]


class DifferentiableOps:
    """Stateless model operations kept separate from model parameters."""

    @staticmethod
    def add_vectors(left: Vector, right: Vector) -> Vector:
        DifferentiableOps._same_length(left, right)
        return [a + b for a, b in zip(left, right)]

    @staticmethod
    def dot(vector: Vector, matrix: Matrix) -> Vector:
        if not matrix or len(matrix) != len(vector):
            raise ValueError("matrix row count must match vector width")
        output_width = len(matrix[0])
        if any(len(row) != output_width for row in matrix):
            raise ValueError("matrix rows must have equal width")
        return [
            sum((vector[row] * matrix[row][column]
                 for row in range(len(vector))), Value(0.0))
            for column in range(output_width)
        ]

    @staticmethod
    def layer_norm(values: Vector, epsilon: float = 1e-5) -> Vector:
        if not values:
            raise ValueError("cannot normalize an empty vector")
        mean = sum(values, Value(0.0)) / len(values)
        variance = sum(((value - mean) ** 2 for value in values), Value(0.0)) / len(values)
        return [(value - mean) / (variance + epsilon) ** 0.5 for value in values]

    @staticmethod
    def softmax(values: Vector) -> Vector:
        if not values:
            raise ValueError("cannot apply softmax to an empty vector")
        highest = max(value.data for value in values)
        exponentials = [(value - highest).exp() for value in values]
        total = sum(exponentials, Value(0.0))
        return [value / total for value in exponentials]

    @staticmethod
    def causal_attention(
        queries: Matrix,
        keys: Matrix,
        values: Matrix,
        model_width: int,
    ) -> Matrix:
        if not (len(queries) == len(keys) == len(values)):
            raise ValueError("attention inputs must have equal sequence lengths")
        scale = math.sqrt(model_width)
        attended: Matrix = []
        for position, query in enumerate(queries):
            scores = [
                sum((query[index] * keys[past][index]
                     for index in range(model_width)), Value(0.0)) / scale
                for past in range(position + 1)
            ]
            probabilities = DifferentiableOps.softmax(scores)
            context = [
                sum((probabilities[past] * values[past][index]
                     for past in range(position + 1)), Value(0.0))
                for index in range(model_width)
            ]
            attended.append(context)
        return attended

    @staticmethod
    def _same_length(left: Sequence[Value], right: Sequence[Value]) -> None:
        if len(left) != len(right):
            raise ValueError("vectors must have equal lengths")
