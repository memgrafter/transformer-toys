import json
import math
import tempfile
import unittest
from pathlib import Path

from autodiff import Value
from checkpoint import JsonCheckpoint
from datasets import AlternatingSequenceDataset
from differentiable_ops import DifferentiableOps
from gradient_descent import GradientDescent
from losses import CrossEntropyLoss
from trainable_transformer import TrainableTransformer
from trainer import Trainer
from transformer import ToyTransformer


CHECKPOINT = Path("transformer.json")


class DataAndPrimitiveTests(unittest.TestCase):
    def test_alternating_dataset_has_both_phases_and_valid_targets(self):
        examples = AlternatingSequenceDataset(5).examples()

        self.assertEqual(len(examples), 2)
        for example in examples:
            self.assertEqual(len(example.input_ids), len(example.target_ids))
            self.assertEqual(
                example.target_ids,
                [1 - token for token in example.input_ids],
            )
            self.assertTrue(all(token in (0, 1) for token in example.input_ids))

    def test_softmax_is_normalized_and_finite(self):
        probabilities = DifferentiableOps.softmax([Value(-2), Value(0), Value(2)])

        self.assertAlmostEqual(sum(value.data for value in probabilities), 1.0)
        self.assertTrue(all(math.isfinite(value.data) for value in probabilities))

    def test_attention_is_causal(self):
        query = [[Value(1)], [Value(1)]]
        keys = [[Value(1)], [Value(1)]]
        values = [[Value(2)], [Value(100)]]
        changed_future_values = [[Value(2)], [Value(-100)]]

        first = DifferentiableOps.causal_attention(query, keys, values, 1)
        second = DifferentiableOps.causal_attention(
            query, keys, changed_future_values, 1
        )

        self.assertAlmostEqual(first[0][0].data, second[0][0].data)


class GradientAndLossTests(unittest.TestCase):
    def test_cross_entropy_matches_known_probability(self):
        logits = [[Value(0.0), Value(0.0)]]
        loss = CrossEntropyLoss().compute(logits, [0])

        self.assertAlmostEqual(loss.data, math.log(2.0), places=10)

    def test_model_gradient_matches_finite_difference(self):
        model = TrainableTransformer.from_json(CHECKPOINT)
        inputs = [0, 1, 0]
        targets = [1, 0, 1]
        loss_function = CrossEntropyLoss()

        loss = loss_function.compute(model.forward(inputs), targets)
        loss.backward()
        parameter = model.query_weight[0][0]
        analytical = parameter.grad
        original = parameter.data
        step = 1e-5

        parameter.data = original + step
        plus = loss_function.compute(model.forward(inputs), targets).data
        parameter.data = original - step
        minus = loss_function.compute(model.forward(inputs), targets).data
        parameter.data = original

        numerical = (plus - minus) / (2.0 * step)
        self.assertAlmostEqual(analytical, numerical, places=4)

    def test_gradient_descent_update_is_exact(self):
        parameter = Value(2.0)
        optimizer = GradientDescent([parameter], learning_rate=0.1)
        loss = (parameter - 5.0) ** 2
        loss.backward()

        optimizer.step()

        self.assertAlmostEqual(parameter.data, 2.6)


class ModelAndCheckpointTests(unittest.TestCase):
    def test_forward_matches_reference_model(self):
        reference = ToyTransformer.from_json(CHECKPOINT)
        model = TrainableTransformer.from_json(CHECKPOINT)
        inputs = [0, 1, 2]

        expected = reference.forward(inputs)
        actual = model.forward(inputs)
        for expected_row, actual_row in zip(expected, actual):
            for expected_value, actual_value in zip(expected_row, actual_row):
                self.assertAlmostEqual(expected_value, actual_value.data, places=10)

    def test_model_values_and_gradients_are_finite(self):
        model = TrainableTransformer.from_json(CHECKPOINT)
        loss = CrossEntropyLoss().compute(model.forward([0, 1, 0]), [1, 0, 1])
        loss.backward()

        for parameter in model.parameters():
            self.assertTrue(math.isfinite(parameter.data))
            self.assertTrue(math.isfinite(parameter.grad))

    def test_checkpoint_round_trip_preserves_predictions(self):
        model = TrainableTransformer.from_json(CHECKPOINT)
        inputs = [0, 1, 0]
        before = [row[-1].data for row in model.forward(inputs)]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            JsonCheckpoint.save(model, path)
            restored = JsonCheckpoint.load(path)

        after = [row[-1].data for row in restored.forward(inputs)]
        self.assertEqual(before, after)


class TrainingBehaviorTests(unittest.TestCase):
    def test_training_reduces_loss(self):
        model = TrainableTransformer.from_json(
            CHECKPOINT,
            vocabulary_size=2,
            separate_output_embeddings=True,
        )
        trainer = Trainer(
            model,
            CrossEntropyLoss(),
            GradientDescent(model.parameters(), learning_rate=0.01),
        )
        history = trainer.fit(AlternatingSequenceDataset(5).examples(), epochs=20)

        self.assertLess(history[-1].loss, history[0].loss)

    def test_training_learns_both_alternating_phases(self):
        model = TrainableTransformer.from_json(
            CHECKPOINT,
            vocabulary_size=2,
            separate_output_embeddings=True,
        )
        trainer = Trainer(
            model,
            CrossEntropyLoss(),
            GradientDescent(model.parameters(), learning_rate=0.01),
        )
        examples = AlternatingSequenceDataset(5).examples()
        trainer.fit(examples, epochs=100)

        correct = 0
        total = 0
        for example in examples:
            logits = model.forward(example.input_ids)
            for row, target in zip(logits, example.target_ids):
                prediction = max(range(len(row)), key=lambda index: row[index].data)
                correct += prediction == target
                total += 1
        self.assertGreaterEqual(correct / total, 0.9)


if __name__ == "__main__":
    unittest.main()
