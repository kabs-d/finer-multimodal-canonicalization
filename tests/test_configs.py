import unittest
from pathlib import Path

from canonical_study.experiment import read_config
from canonical_study.decoder_experiment import read_decoder_config


class ConfigTests(unittest.TestCase):
    def test_all_baseline_configs_are_valid(self):
        project_root = Path(__file__).resolve().parents[1]
        configs = sorted((project_root / "configs" / "baseline").glob("*.json"))
        self.assertEqual(len(configs), 2)
        for path in configs:
            config = read_config(path)
            self.assertEqual(config["dataset"]["name"], "oxford")
            self.assertEqual(config["seeds"], [42, 43, 44])

    def test_all_frozen_decoder_configs_enforce_the_agreed_scope(self):
        project_root = Path(__file__).resolve().parents[1]
        configs = sorted(
            (project_root / "configs" / "frozen_decoder").glob("*.json")
        )
        self.assertEqual(len(configs), 2)
        for path in configs:
            config = read_decoder_config(path)
            self.assertEqual(config["dataset"]["name"], "cub")
            self.assertTrue(config["feature_extraction"]["encoders_frozen"])
            self.assertFalse(config["alignment"]["refit_rotation_on_cub"])
            self.assertTrue(config["alignment"]["recompute_means_on_cub_train"])
            self.assertFalse(config["alignment"]["raw_rotation_ablation"])
            self.assertEqual(config["decoder"]["architecture"], "linear")
            self.assertEqual(config["decoder"]["seeds"], [42, 43, 44, 45, 46])

    def test_mlp_configs_are_fixed_and_reuse_frozen_features(self):
        project_root = Path(__file__).resolve().parents[1]
        configs = sorted(
            (project_root / "configs" / "mlp_decoder").glob("*.json")
        )
        self.assertEqual(len(configs), 2)
        for path in configs:
            config = read_decoder_config(path)
            decoder = config["decoder"]
            self.assertEqual(decoder["architecture"], "mlp")
            self.assertEqual(decoder["hidden_dim"], 512)
            self.assertEqual(decoder["activation"], "gelu")
            self.assertEqual(decoder["dropout"], 0.1)
            self.assertFalse(
                config["feature_extraction"]["repeat_encoder_inference"]
            )
            self.assertIn(
                "linear", config["feature_extraction"]["reuse_run_id"]
            )


if __name__ == "__main__":
    unittest.main()
