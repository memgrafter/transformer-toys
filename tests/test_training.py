import unittest

from losses import CrossEntropyLoss
from datasets import AlternatingSequenceDataset
from gradient_descent import GradientDescent
from trainable_transformer import TrainableTransformer
from trainer import Trainer
from transformer import ToyTransformer


class TrainableTransformerTests(unittest.TestCase):
    def test_forward_matches_reference(self):
        reference = ToyTransformer.from_json("transformer.json")
        model = TrainableTransformer.from_json("transformer.json")
        token_ids = [0, 1, 2]

        expected = reference.forward(token_ids)
        actual = model.forward(token_ids)

        for expected_row, actual_row in zip(expected, actual):
            for expected_value, actual_value in zip(expected_row, actual_row):
                self.assertAlmostEqual(expected_value, actual_value.data, places=10)

    def test_training_reduces_alternating_sequence_loss(self):
        model = TrainableTransformer.from_json("transformer.json")
        trainer = Trainer(
            model,
            CrossEntropyLoss(),
            GradientDescent(model.parameters(), learning_rate=0.01),
        )
        history = trainer.fit(AlternatingSequenceDataset(5).examples(), epochs=20)

        self.assertLess(history[-1].loss, history[0].loss)


if __name__ == "__main__":
    unittest.main()
