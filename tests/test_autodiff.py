import math
import unittest

from autodiff import Value
from gradient_descent import GradientDescent


class ValueTests(unittest.TestCase):
    def test_polynomial_gradient(self):
        x = Value(2.0)
        output = x**2 + 3.0 * x + 1.0

        output.backward()

        self.assertAlmostEqual(output.data, 11.0)
        self.assertAlmostEqual(x.grad, 7.0)

    def test_composed_operations(self):
        x = Value(0.5)
        output = ((x * 2.0).exp() + 1.0).log().tanh()

        output.backward()

        # Central finite difference checks the composed backward path.
        step = 1e-5

        def function(value):
            return math.tanh(math.log(math.exp(value * 2.0) + 1.0))

        expected = (function(x.data + step) - function(x.data - step)) / (2.0 * step)
        self.assertAlmostEqual(x.grad, expected, places=5)

    def test_backward_accumulates_independent_graphs(self):
        x = Value(3.0)
        (x * x).backward()
        (x * x).backward()
        self.assertAlmostEqual(x.grad, 12.0)
        x.zero_grad()
        self.assertEqual(x.grad, 0.0)


class GradientDescentTests(unittest.TestCase):
    def test_minimizes_quadratic(self):
        x = Value(0.0)
        optimizer = GradientDescent([x], learning_rate=0.1)

        for _ in range(100):
            optimizer.zero_grad()
            loss = (x - 3.0) ** 2
            loss.backward()
            optimizer.step()

        self.assertAlmostEqual(x.data, 3.0, places=5)
        self.assertLess(loss.data, 1e-8)


if __name__ == "__main__":
    unittest.main()
