import unittest

import numpy as np
import torch

from canonical_study.attribute_text_global_retrieval import (
    CLIP_READABLE_ATTRIBUTE_INDICES,
    attribute_only_prompt,
    evaluate_global_attribute_retrieval,
)


class AttributeTextGlobalRetrievalTests(unittest.TestCase):
    def test_attribute_only_prompt(self):
        self.assertEqual(
            attribute_only_prompt("black bill"),
            "a photo of a bird with black bill.",
        )

    def test_clip_readable_subset_is_fixed_and_nontrivial(self):
        self.assertEqual(len(CLIP_READABLE_ATTRIBUTE_INDICES), 95)
        self.assertIn(209, CLIP_READABLE_ATTRIBUTE_INDICES)
        self.assertNotIn(253, CLIP_READABLE_ATTRIBUTE_INDICES)

    def test_global_retrieval_metrics_and_random_baseline(self):
        attributes = np.asarray(
            [
                [1, 0],
                [1, 0],
                [0, 1],
                [0, 1],
            ],
            dtype=bool,
        )
        mask = np.ones_like(attributes, dtype=bool)
        attribute_rows = [
            {
                "attribute_index": 0,
                "attribute_id": 1,
                "raw_attribute_name": "has_belly_color::white",
                "attribute_phrase": "white belly",
                "attribute_prompt": "a photo of a bird with white belly.",
                "clip_readable_subset": True,
            },
            {
                "attribute_index": 1,
                "attribute_id": 2,
                "raw_attribute_name": "has_primary_color::grey",
                "attribute_phrase": "grey primary",
                "attribute_prompt": "a photo of a bird with grey primary.",
                "clip_readable_subset": False,
            },
        ]
        text = torch.eye(2)
        images = torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.1, 0.9],
                [0.0, 1.0],
            ]
        )

        aggregate, per_attribute = evaluate_global_attribute_retrieval(
            {"condition": text},
            {"condition": images},
            attributes,
            mask,
            attribute_rows,
            k_values=[1, 2],
        )

        self.assertEqual(len(per_attribute), 2)
        self.assertAlmostEqual(
            aggregate["all_312"]["condition"]["ranking_accuracy_macro"],
            1.0,
        )
        self.assertAlmostEqual(
            aggregate["all_312"]["condition"]["random_precision_at_k"],
            0.5,
        )
        self.assertAlmostEqual(
            aggregate["clip_readable"]["condition"]["precision_at_2_macro"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
