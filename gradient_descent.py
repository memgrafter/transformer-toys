"""Plain gradient descent for the scalar autodiff Value class."""

from __future__ import annotations

import math
from collections.abc import Iterable

from autodiff import Value


class GradientDescent:
    """Update a fixed collection of parameters using one learning rate."""

    def __init__(self, parameters: Iterable[Value], learning_rate: float) -> None:
        self.parameters = tuple(parameters)
        if not self.parameters:
            raise ValueError("at least one parameter is required")
        if any(not isinstance(parameter, Value) for parameter in self.parameters):
            raise TypeError("parameters must be Value instances")
        if len({id(parameter) for parameter in self.parameters}) != len(self.parameters):
            raise ValueError("parameters must not contain duplicates")
        self.learning_rate = self._positive_finite(learning_rate, "learning_rate")

    @staticmethod
    def _positive_finite(value: float, name: str) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{name} must be numeric") from error
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
        return value

    def zero_grad(self) -> None:
        """Clear gradients before the next forward/backward pass."""
        for parameter in self.parameters:
            parameter.zero_grad()

    def step(self) -> None:
        """Apply one gradient-descent update to every parameter."""
        for parameter in self.parameters:
            if not math.isfinite(parameter.grad):
                raise ValueError("cannot update with a non-finite gradient")
            updated = parameter.data - self.learning_rate * parameter.grad
            if not math.isfinite(updated):
                raise ValueError("gradient update produced a non-finite parameter")
            parameter.data = updated
