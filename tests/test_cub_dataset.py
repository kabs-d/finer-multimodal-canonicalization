import tempfile
import unittest
from pathlib import Path

from PIL import Image

from canonical_study.datasets import CUB2002011


class CUBDatasetTests(unittest.TestCase):
    def test_attributes_and_not_visible_mask_are_parsed_per_cell(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "CUB_200_2011"
            (root / "attributes").mkdir(parents=True)
            (root / "images" / "001.Bird").mkdir(parents=True)
            (root / "images.txt").write_text(
                "1 001.Bird/a.jpg\n2 001.Bird/b.jpg\n", encoding="utf-8"
            )
            (root / "classes.txt").write_text("1 001.Bird\n", encoding="utf-8")
            (root / "image_class_labels.txt").write_text(
                "1 1\n2 1\n", encoding="utf-8"
            )
            (root / "train_test_split.txt").write_text(
                "1 1\n2 0\n", encoding="utf-8"
            )
            (root / "attributes.txt").write_text(
                "1 has_color::red\n2 has_color::blue\n", encoding="utf-8"
            )
            (root / "attributes" / "certainties.txt").write_text(
                "1 not visible\n2 guessing\n3 probably\n4 definitely\n",
                encoding="utf-8",
            )
            (root / "attributes" / "image_attribute_labels.txt").write_text(
                "1 1 1 4 0.0\n"
                "1 2 0 1 0.0\n"
                "2 1 0 2 0.0\n"
                "2 2 1 3 0.0\n",
                encoding="utf-8",
            )
            Image.new("RGB", (4, 4)).save(root / "images" / "001.Bird" / "a.jpg")
            Image.new("RGB", (4, 4)).save(root / "images" / "001.Bird" / "b.jpg")

            train = CUB2002011(root, "train")
            test = CUB2002011(root, "test")
            self.assertEqual(len(train), 1)
            self.assertEqual(len(test), 1)
            self.assertEqual(train[0]["attributes"].tolist(), [1.0, 0.0])
            self.assertEqual(train[0]["attribute_mask"].tolist(), [True, False])
            self.assertEqual(test[0]["attribute_mask"].tolist(), [True, True])


if __name__ == "__main__":
    unittest.main()
