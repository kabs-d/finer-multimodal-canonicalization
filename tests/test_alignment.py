import unittest

import torch

from canonical_study.alignment import fit_orthogonal_alignment


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


if __name__ == "__main__":
    unittest.main()

