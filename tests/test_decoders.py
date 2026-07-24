import tempfile
import unittest
from pathlib import Path

import torch

from canonical_study.decoders import build_decoder
from canonical_study.mlp_experiment import _train_target_decoder


MLP_CONFIG = {
    "architecture": "mlp",
    "hidden_dim": 512,
    "activation": "gelu",
    "dropout": 0.1,
}


class DecoderTests(unittest.TestCase):
    def test_mlp_shape_and_deterministic_evaluation(self):
        model = build_decoder(MLP_CONFIG, 32, 12)
        features = torch.randn(7, 32)
        model.eval()
        first = model(features)
        second = model(features)
        self.assertEqual(first.shape, (7, 12))
        torch.testing.assert_close(first, second)

    def test_mlp_checkpoint_round_trip(self):
        model = build_decoder(MLP_CONFIG, 16, 5).eval()
        features = torch.randn(4, 16)
        expected = model(features)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "decoder.pt"
            torch.save(model.state_dict(), checkpoint)
            restored = build_decoder(MLP_CONFIG, 16, 5).eval()
            restored.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        torch.testing.assert_close(expected, restored(features))

    def test_exact_orthogonal_alignment_preserves_mlp_predictions(self):
        generator = torch.Generator().manual_seed(17)
        source = torch.randn(20, 24, generator=generator)
        orthogonal, _ = torch.linalg.qr(
            torch.randn(24, 24, generator=generator)
        )
        target = source @ orthogonal
        aligned = source @ orthogonal
        model = build_decoder(MLP_CONFIG, 24, 9).eval()
        torch.testing.assert_close(model(target), model(aligned))

    def test_linear_factory_preserves_state_dict_interface(self):
        model = build_decoder({"architecture": "linear"}, 8, 3)
        self.assertEqual(set(model.state_dict()), {"weight", "bias"})

    def test_mlp_training_smoke(self):
        generator = torch.Generator().manual_seed(23)
        features = torch.randn(24, 8, generator=generator)
        labels = torch.tensor(
            [[index % 2, (index // 2) % 2, (index // 3) % 2]
             for index in range(24)],
            dtype=torch.float32,
        )
        mask = torch.ones_like(labels, dtype=torch.bool)
        config = {
            **MLP_CONFIG,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 8,
            "max_epochs": 2,
            "patience": 2,
            "pos_weight_min": 0.25,
            "pos_weight_max": 20.0,
        }
        model, training = _train_target_decoder(
            features,
            labels,
            mask,
            torch.arange(16).numpy(),
            torch.arange(16, 24).numpy(),
            torch.ones(3, dtype=torch.bool).numpy(),
            config,
            seed=42,
            device=torch.device("cpu"),
        )
        self.assertEqual(model(features).shape, (24, 3))
        self.assertEqual(training["epochs_run"], 2)


if __name__ == "__main__":
    unittest.main()
