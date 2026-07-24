import unittest

import numpy as np

from canonical_study.attribute_analysis import (
    per_bird_recovery,
    select_f1_thresholds,
    within_species_ranking_accuracy,
)


class AttributeAnalysisTests(unittest.TestCase):
    def test_validation_threshold_maximizes_f1(self):
        labels = np.asarray([[1], [1], [0], [0]])
        mask = np.ones_like(labels, dtype=bool)
        scores = np.asarray([[0.9], [0.6], [0.7], [0.1]])
        thresholds = select_f1_thresholds(
            labels, mask, scores, np.asarray([True])
        )
        self.assertEqual(thresholds.tolist(), [0.6])

    def test_per_bird_counts_exclude_true_negatives(self):
        labels = np.asarray([[1, 0, 1], [0, 1, 0]])
        mask = np.ones_like(labels, dtype=bool)
        scores = np.asarray([[0.9, 0.8, 0.1], [0.1, 0.7, 0.2]])
        result = per_bird_recovery(
            labels,
            mask,
            scores,
            np.asarray([0.5, 0.5, 0.5]),
            np.asarray([True, True, True]),
        )
        self.assertEqual(result["mean_correctly_recovered"], 1.0)
        self.assertEqual(result["mean_missed"], 0.5)
        self.assertEqual(result["mean_hallucinated"], 0.5)

    def test_within_species_ranking_has_interpretable_chance(self):
        labels = np.asarray([[1], [0], [1], [0]])
        mask = np.ones_like(labels, dtype=bool)
        species = np.asarray([0, 0, 1, 1])
        perfect_scores = np.asarray([[0.9], [0.1], [0.8], [0.2]])
        tied_scores = np.ones_like(perfect_scores)
        eligible = np.asarray([True])
        perfect = within_species_ranking_accuracy(
            labels, mask, perfect_scores, species, eligible
        )
        tied = within_species_ranking_accuracy(
            labels, mask, tied_scores, species, eligible
        )
        self.assertEqual(perfect["macro_pair_accuracy"], 1.0)
        self.assertEqual(tied["macro_pair_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
