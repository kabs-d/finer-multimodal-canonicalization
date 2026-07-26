import unittest

import numpy as np

from canonical_study.attribute_prompt_audit import (
    build_audit_rows,
    clean_attribute_phrase,
    threshold_flags,
    train_prevalence_and_rare,
)


class AttributePromptAuditTests(unittest.TestCase):
    def test_attribute_phrase_cleaning(self):
        self.assertEqual(
            clean_attribute_phrase("has_wing_color::yellow"),
            "yellow wing",
        )
        self.assertEqual(
            clean_attribute_phrase("has_breast_pattern::striped"),
            "striped breast",
        )
        self.assertEqual(
            clean_attribute_phrase("has_bill_shape::hooked"),
            "hooked bill",
        )
        self.assertEqual(
            clean_attribute_phrase("has_bill_length::longer_than_head"),
            "bill longer than head",
        )

    def test_threshold_flags(self):
        flags = threshold_flags(positive_count=3, negative_count=2)

        self.assertTrue(flags["valid_ge1_pos_ge1_neg"])
        self.assertTrue(flags["valid_ge2_pos_ge2_neg"])
        self.assertFalse(flags["valid_ge3_pos_ge3_neg"])
        self.assertFalse(flags["valid_ge5_pos_ge5_neg"])

    def test_valid_group_detection_ignores_invisible_labels(self):
        class_names = ["bird"]
        attribute_names = ["has_wing_color::yellow", "has_bill_shape::hooked"]
        test_labels = np.asarray([0, 0, 0, 0])
        test_attributes = np.asarray(
            [
                [1, 1],
                [0, 1],
                [1, 0],
                [0, 0],
            ],
            dtype=bool,
        )
        test_mask = np.asarray(
            [
                [1, 1],
                [1, 0],
                [0, 1],
                [0, 1],
            ],
            dtype=bool,
        )
        train_prevalence = np.asarray([0.25, 0.75])
        rare = np.asarray([True, False])

        rows = build_audit_rows(
            class_names,
            attribute_names,
            test_labels,
            test_attributes,
            test_mask,
            train_prevalence,
            rare,
        )

        yellow = rows[0]
        hooked = rows[1]
        self.assertEqual(yellow["test_visible"], 2)
        self.assertEqual(yellow["test_positive"], 1)
        self.assertEqual(yellow["test_negative"], 1)
        self.assertTrue(yellow["valid_ge1_pos_ge1_neg"])
        self.assertEqual(hooked["test_visible"], 3)
        self.assertEqual(hooked["test_positive"], 1)
        self.assertEqual(hooked["test_negative"], 2)
        self.assertTrue(hooked["valid_ge1_pos_ge1_neg"])

    def test_groups_with_only_one_class_are_invalid(self):
        rows = build_audit_rows(
            ["bird"],
            ["has_wing_color::yellow", "has_wing_color::blue"],
            np.asarray([0, 0]),
            np.asarray([[1, 0], [1, 0]], dtype=bool),
            np.asarray([[1, 1], [1, 1]], dtype=bool),
            np.asarray([0.5, 0.0]),
            np.asarray([False, True]),
        )

        self.assertFalse(rows[0]["valid_ge1_pos_ge1_neg"])
        self.assertFalse(rows[1]["valid_ge1_pos_ge1_neg"])

    def test_rare_attribute_flag_uses_train_prevalence_only(self):
        train_attributes = np.asarray(
            [
                [1, 1, 1, 0],
                [1, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=bool,
        )
        train_mask = np.ones_like(train_attributes, dtype=bool)

        prevalence, rare, cutoff = train_prevalence_and_rare(
            train_attributes,
            train_mask,
        )

        self.assertAlmostEqual(prevalence[3], 0.25)
        self.assertAlmostEqual(cutoff, 0.25)
        self.assertTrue(rare[3])
        self.assertFalse(rare[0])


if __name__ == "__main__":
    unittest.main()
