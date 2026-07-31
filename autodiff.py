"""Small reverse-mode scalar automatic differentiation engine."""

from __future__ import annotations

import math
from collections.abc import Callable


class Value:
    """A scalar value and its reverse-mode gradient in a computation graph."""

    def __init__(
        self,
        data: float,
        children: tuple[Value, ...] = (),
        operation: str = "",
        label: str = "",
    ) -> None:
        self.data = self._finite_float(data, "data")
        self.grad = 0.0
        self.children = tuple(children)
        self.operation = operation
        self.label = label
        self._backward: Callable[[], None] = lambda: None

    @staticmethod
    def _finite_float(value: float, name: str) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{name} must be numeric") from error
        if not math.isfinite(result):
            raise ValueError(f"{name} must be finite")
        return result

    @staticmethod
    def _coerce(other: Value | float) -> Value:
        return other if isinstance(other, Value) else Value(other)

    def __add__(self, other: Value | float) -> Value:
        other_value = self._coerce(other)
        result = Value(self.data + other_value.data, (self, other_value), "+")

        def backward() -> None:
            self.grad += result.grad
            other_value.grad += result.grad

        result._backward = backward
        return result

    def __radd__(self, other: Value | float) -> Value:
        return self + other

    def __neg__(self) -> Value:
        result = Value(-self.data, (self,), "neg")

        def backward() -> None:
            self.grad -= result.grad

        result._backward = backward
        return result

    def __sub__(self, other: Value | float) -> Value:
        return self + (-self._coerce(other))

    def __rsub__(self, other: Value | float) -> Value:
        return self._coerce(other) - self

    def __mul__(self, other: Value | float) -> Value:
        other_value = self._coerce(other)
        result = Value(self.data * other_value.data, (self, other_value), "*")

        def backward() -> None:
            self.grad += other_value.data * result.grad
            other_value.grad += self.data * result.grad

        result._backward = backward
        return result

    def __rmul__(self, other: Value | float) -> Value:
        return self * other

    def __truediv__(self, other: Value | float) -> Value:
        return self * self._coerce(other) ** -1

    def __rtruediv__(self, other: Value | float) -> Value:
        return self._coerce(other) / self

    def __pow__(self, exponent: float) -> Value:
        exponent = self._finite_float(exponent, "exponent")
        if self.data == 0.0 and exponent < 0.0:
            raise ValueError("cannot raise zero to a negative exponent")
        result = Value(self.data**exponent, (self,), "pow")

        def backward() -> None:
            if exponent == 0.0:
                return
            self.grad += exponent * self.data ** (exponent - 1.0) * result.grad

        result._backward = backward
        return result

    def exp(self) -> Value:
        try:
            result = Value(math.exp(self.data), (self,), "exp")
        except OverflowError as error:
            raise ValueError("exponential overflow") from error

        def backward() -> None:
            self.grad += result.data * result.grad

        result._backward = backward
        return result

    def log(self) -> Value:
        if self.data <= 0.0:
            raise ValueError("log requires a positive value")
        result = Value(math.log(self.data), (self,), "log")

        def backward() -> None:
            self.grad += result.grad / self.data

        result._backward = backward
        return result

    def relu(self) -> Value:
        result = Value(max(0.0, self.data), (self,), "relu")

        def backward() -> None:
            if self.data > 0.0:
                self.grad += result.grad

        result._backward = backward
        return result

    def tanh(self) -> Value:
        result = Value(math.tanh(self.data), (self,), "tanh")

        def backward() -> None:
            self.grad += (1.0 - result.data**2) * result.grad

        result._backward = backward
        return result

    def backward(self, gradient: float = 1.0) -> None:
        """Accumulate gradients for every ancestor of this value."""
        gradient = self._finite_float(gradient, "gradient")
        topology: list[Value] = []
        visited: set[int] = set()

        def visit(node: Value) -> None:
            identity = id(node)
            if identity in visited:
                return
            visited.add(identity)
            for child in node.children:
                visit(child)
            topology.append(node)

        visit(self)
        self.grad += gradient
        for node in reversed(topology):
            node._backward()

    def zero_grad(self) -> None:
        self.grad = 0.0

    def __repr__(self) -> str:
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g})"
