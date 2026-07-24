import unittest

import numpy as np
import torch

from canonical_study.decoder_experiment import _cluster_bootstrap_ap
from canonical_study.metrics import (
    _binary_average_precision,
    aggregate_nested,
    class_retrieval_top1,
    paired_cosine,
    multilabel_metrics,
    zero_shot_top1,
)


class MetricTests(unittest.TestCase):
    def test_identity_metrics_are_perfect(self):
        values = torch.eye(4)
        labels = torch.arange(4)
        self.assertAlmostEqual(paired_cosine(values, values), 1.0)
        self.assertAlmostEqual(
            class_retrieval_top1(values, values, labels, labels), 1.0
        )
        self.assertAlmostEqual(zero_shot_top1(values, values, labels), 1.0)

    def test_nested_aggregation_uses_population_standard_deviation(self):
        mean, std = aggregate_nested([{"x": {"y": 1.0}}, {"x": {"y": 3.0}}])
        self.assertEqual(mean, {"x": {"y": 2.0}})
        self.assertEqual(std, {"x": {"y": 1.0}})

    def test_multilabel_metrics_mask_unobserved_cells(self):
        labels = np.asarray(
            [[1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.int64
        )
        mask = np.asarray(
            [[1, 1], [1, 1], [1, 0], [1, 0]], dtype=bool
        )
        probabilities = np.asarray(
            [[0.9, 0.1], [0.1, 0.9], [0.8, 0.0], [0.2, 1.0]],
            dtype=np.float64,
        )
        aggregate, attributes = multilabel_metrics(
            labels, mask, probabilities
        )
        self.assertAlmostEqual(aggregate["macro_map"], 1.0)
        self.assertAlmostEqual(aggregate["micro_map"], 1.0)
        self.assertEqual(attributes[1]["observed"], 2)

    def test_weighted_cluster_bootstrap_ap_matches_unweighted_ap(self):
        labels = np.asarray([1, 0, 1, 0], dtype=np.int64)
        scores = np.asarray([0.9, 0.8, 0.8, 0.1], dtype=np.float64)
        species = np.asarray([0, 0, 1, 1], dtype=np.int64)
        draw_counts = np.asarray([[1, 1]], dtype=np.int64)
        expected = _binary_average_precision(labels, scores)
        actual = _cluster_bootstrap_ap(
            labels, scores, species, draw_counts
        )[0]
        self.assertAlmostEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
