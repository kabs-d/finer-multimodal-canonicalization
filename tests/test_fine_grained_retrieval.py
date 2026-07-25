import unittest

import numpy as np
import torch

from canonical_study.fine_grained_retrieval import (
    _topk_indices,
    evaluable_attributes,
    evaluate_retrieval_rankings,
    rare_attributes_bottom_quartile,
    same_species_candidate_indices,
)


class FineGrainedRetrievalTests(unittest.TestCase):
    def test_same_species_candidates_exclude_query(self):
        labels = np.asarray([0, 0, 1, 0])

        candidates = same_species_candidate_indices(labels, 1)

        np.testing.assert_array_equal(candidates, np.asarray([0, 3]))

    def test_attribute_overlap_at_k_on_synthetic_rankings(self):
        attributes = np.asarray(
            [
                [1, 1, 0],
                [1, 0, 0],
                [0, 1, 0],
            ]
        )
        mask = np.ones_like(attributes, dtype=bool)
        labels = np.asarray([0, 0, 0])
        eligible = np.asarray([True, True, False])
        rare = np.asarray([False, False, False])
        rankings = {"condition": {2: [np.asarray([1, 2]), np.asarray([0, 2]), np.asarray([0, 1])]}}

        summary, per_query, _ = evaluate_retrieval_rankings(
            rankings,
            attributes,
            mask,
            labels,
            eligible,
            rare,
        )

        # Query 0 has two positives; candidates 1 and 2 each share one, so overlap=0.5.
        query0 = [row for row in per_query if row["query_index"] == 0][0]
        self.assertAlmostEqual(query0["attribute_overlap"], 0.5)
        self.assertAlmostEqual(
            summary["condition"]["k2"]["same_species_attribute_overlap"],
            0.5,
        )

    def test_rare_attributes_use_train_prevalence_only(self):
        train_attributes = np.asarray(
            [
                [1, 1, 1, 0],
                [1, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
            ]
        )
        train_mask = np.ones_like(train_attributes, dtype=bool)
        test_attributes = np.asarray(
            [
                [1, 1, 1, 1],
                [0, 0, 0, 0],
            ]
        )
        test_mask = np.ones_like(test_attributes, dtype=bool)
        eligible = evaluable_attributes(
            train_attributes,
            train_mask,
            test_attributes,
            test_mask,
        )

        rare, prevalence = rare_attributes_bottom_quartile(
            train_attributes,
            train_mask,
            eligible,
        )

        self.assertTrue(rare[3])
        self.assertFalse(rare[0])
        self.assertAlmostEqual(prevalence[3], 0.25)

    def test_rare_recall_at_k_on_synthetic_rankings(self):
        attributes = np.asarray(
            [
                [1, 1, 0],
                [1, 0, 0],
                [0, 1, 0],
            ]
        )
        mask = np.ones_like(attributes, dtype=bool)
        labels = np.asarray([0, 0, 0])
        eligible = np.asarray([True, True, False])
        rare = np.asarray([True, True, False])
        rankings = {"condition": {1: [np.asarray([1]), np.asarray([0]), np.asarray([0])]}}

        summary, per_query, _ = evaluate_retrieval_rankings(
            rankings,
            attributes,
            mask,
            labels,
            eligible,
            rare,
        )

        query0 = [row for row in per_query if row["query_index"] == 0][0]
        self.assertAlmostEqual(query0["rare_attribute_recall"], 0.5)
        self.assertAlmostEqual(
            summary["condition"]["k1"]["rare_attribute_recall"],
            (0.5 + 1.0 + 1.0) / 3.0,
        )

    def test_exact_alignment_matches_native_target_retrieval(self):
        source = torch.eye(4)
        permutation = torch.tensor([2, 0, 3, 1])
        target = source[:, permutation]
        labels = np.asarray([0, 0, 0, 0])

        native = _topk_indices(target, target, labels, [1])
        aligned = _topk_indices(source[:, permutation], target, labels, [1])

        for native_row, aligned_row in zip(native[1], aligned[1]):
            np.testing.assert_array_equal(native_row, aligned_row)


if __name__ == "__main__":
    unittest.main()
