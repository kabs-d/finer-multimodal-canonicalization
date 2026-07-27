import unittest

import torch

from canonical_study.decoders import build_decoder


class DecoderTests(unittest.TestCase):
    def test_linear_factory_preserves_state_dict_interface(self):
        model = build_decoder({"architecture": "linear"}, 8, 3)
        self.assertEqual(set(model.state_dict()), {"weight", "bias"})


if __name__ == "__main__":
    unittest.main()
