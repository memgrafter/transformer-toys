import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from datasets import AlternatingSequenceDataset  # noqa: E402
from trainable_transformer import TrainableTransformer  # noqa: E402
TRAINING_SCRIPT = REPOSITORY_ROOT / "train_binary_manual.py"


class BinaryTrainingIntegrationTests(unittest.TestCase):
    def test_real_manual_training_writes_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            for epochs in (100, 500):
                with self.subTest(epochs=epochs):
                    output = Path(temporary_directory) / f"binary-trained-{epochs}.json"
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(TRAINING_SCRIPT),
                            "--epochs",
                            str(epochs),
                            "--output",
                            str(output),
                        ],
                        cwd=REPOSITORY_ROOT,
                        capture_output=True,
                        text=True,
                        check=False,
                    )

                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertTrue(output.is_file())
                    self.assertIn("saved:", result.stdout)

                    losses = [
                        float(line.split("loss=")[1])
                        for line in result.stdout.splitlines()
                        if " loss=" in line
                    ]
                    self.assertGreater(len(losses), 1)
                    self.assertLess(losses[-1], losses[0])

                    with output.open(encoding="utf-8") as file:
                        checkpoint = json.load(file)
                    self.assertEqual(checkpoint["vocabulary_size"], 2)
                    self.assertEqual(len(checkpoint["token_embedding"]), 2)
                    self.assertEqual(len(checkpoint["token_embedding"][0]), 8)
                    self.assertEqual(len(checkpoint["output_embedding"]), 2)

                    model = TrainableTransformer.from_json(output)
                    correct = 0
                    total = 0
                    for example in AlternatingSequenceDataset(5).examples():
                        logits = model.forward(example.input_ids)
                        for row, target in zip(logits, example.target_ids):
                            prediction = max(
                                range(len(row)),
                                key=lambda index: row[index].data,
                            )
                            correct += prediction == target
                            total += 1
                    self.assertGreaterEqual(correct / total, 0.9)

    def test_real_training_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            outputs = []
            for index in range(2):
                output = Path(temporary_directory) / f"repeat-{index}.json"
                result = subprocess.run(
                    [
                        sys.executable,
                        str(TRAINING_SCRIPT),
                        "--epochs",
                        "10",
                        "--output",
                        str(output),
                    ],
                    cwd=REPOSITORY_ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                outputs.append(output.read_text(encoding="utf-8"))

            self.assertEqual(outputs[0], outputs[1])


if __name__ == "__main__":
    unittest.main()
