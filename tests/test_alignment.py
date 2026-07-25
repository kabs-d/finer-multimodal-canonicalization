import unittest
from pathlib import Path
import tempfile

import torch

from canonical_study.alignment import fit_orthogonal_alignment
from canonical_study.cub_train_q_control import _save_cub_train_alignment


class OrthogonalAlignmentTests(unittest.TestCase):
    def test_recovers_rotation_and_translation(self):
        generator = torch.Generator().manual_seed(7)
        source = torch.randn(200, 16, generator=generator, dtype=torch.float64)
        matrix = torch.randn(16, 16, generator=generator, dtype=torch.float64)
        rotation, _ = torch.linalg.qr(matrix)
        translation = torch.randn(1, 16, generator=generator, dtype=torch.float64)
        target = source @ rotation + translation

        fitted = fit_orthogonal_alignment(source, target)

        self.assertLess(fitted.orthogonality_error, 1e-10)
        self.assertTrue(
            torch.allclose(
                fitted.transform(source),
                target,
                atol=1e-10,
                rtol=1e-10,
            )
        )

    def test_rejects_unpaired_shapes(self):
        with self.assertRaises(ValueError):
            fit_orthogonal_alignment(torch.zeros(3, 4), torch.zeros(4, 4))

    def test_cub_train_alignment_uses_train_split_only(self):
        generator = torch.Generator().manual_seed(31)
        train_source = torch.randn(12, 8, generator=generator, dtype=torch.float64)
        test_source = torch.randn(5, 8, generator=generator, dtype=torch.float64)
        rotation, _ = torch.linalg.qr(
            torch.randn(8, 8, generator=generator, dtype=torch.float64)
        )
        train_translation = torch.randn(1, 8, generator=generator, dtype=torch.float64)
        train_target = train_source @ rotation + train_translation
        test_target = test_source @ rotation + train_translation
        config = {
            "source_model": {"kind": "synthetic", "name": "source"},
            "target_model": {"kind": "synthetic", "name": "target"},
        }
        train_features = {
            "source": train_source.float(),
            "target": train_target.float(),
            "image_ids": torch.arange(12),
        }
        test_features = {
            "source": test_source.float(),
            "target": test_target.float(),
            "image_ids": torch.arange(100, 105),
        }
        with tempfile.TemporaryDirectory() as temporary:
            payload = _save_cub_train_alignment(
                config,
                train_features,
                test_features,
                Path(temporary) / "alignment.pt",
            )

        fitted = fit_orthogonal_alignment(train_source.float(), train_target.float())
        aligned_test = fitted.transform(test_source.float())
        torch.testing.assert_close(aligned_test, test_target.float(), atol=1e-5, rtol=1e-5)
        self.assertEqual(payload["fit_examples"], 12)
        self.assertEqual(payload["heldout_examples"], 5)
        self.assertEqual(payload["train_test_image_id_overlap"], 0)

    def test_cub_train_alignment_rejects_leaked_test_ids(self):
        features = {
            "source": torch.randn(3, 4),
            "target": torch.randn(3, 4),
            "image_ids": torch.tensor([1, 2, 3]),
        }
        config = {
            "source_model": {"kind": "synthetic", "name": "source"},
            "target_model": {"kind": "synthetic", "name": "target"},
        }
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(RuntimeError):
                _save_cub_train_alignment(
                    config,
                    features,
                    features,
                    Path(temporary) / "alignment.pt",
                )


if __name__ == "__main__":
    unittest.main()
