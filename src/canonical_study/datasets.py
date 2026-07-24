"""Dataset preparation and paper-compatible records."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class OxfordRecord:
    stem: str
    class_name: str
    label: int


class OxfordPets(Dataset):
    """Return untransformed PIL images and paper-compatible class labels."""

    def __init__(self, root: Path | str, split: str, *, download: bool = False):
        if split not in {"trainval", "test"}:
            raise ValueError("Oxford split must be 'trainval' or 'test'")
        root = Path(root).expanduser().resolve()
        if download:
            from torchvision.datasets import OxfordIIITPet

            OxfordIIITPet(
                root=str(root),
                split=split,
                target_types="category",
                download=True,
            )

        dataset_root = root / "oxford-iiit-pet"
        annotations = dataset_root / "annotations" / f"{split}.txt"
        image_root = dataset_root / "images"
        if not annotations.is_file():
            raise FileNotFoundError(
                f"missing {annotations}; run `canonical-study prepare-data` first"
            )

        stems = [
            line.split()[0]
            for line in annotations.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        class_names = sorted(
            {stem.rsplit("_", 1)[0].lower().replace("_", " ") for stem in stems}
        )
        class_to_index = {name: index for index, name in enumerate(class_names)}
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.image_root = image_root
        self.records = []
        for stem in stems:
            class_name = stem.rsplit("_", 1)[0].lower().replace("_", " ")
            self.records.append(
                OxfordRecord(stem, class_name, class_to_index[class_name])
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        image = Image.open(self.image_root / f"{record.stem}.jpg").convert("RGB")
        return {
            "image": image,
            "label": record.label,
            "text": record.class_name,
            "class_name": record.class_name,
            "image_path": f"{record.stem}.jpg",
        }


def prepare_oxford(root: Path | str) -> dict[str, int]:
    train = OxfordPets(root, "trainval", download=True)
    test = OxfordPets(root, "test", download=True)
    return {
        "trainval_examples": len(train),
        "test_examples": len(test),
        "classes": train.num_classes,
    }


def validate_oxford(root: Path | str) -> dict[str, int]:
    train = OxfordPets(root, "trainval")
    test = OxfordPets(root, "test")
    if train.class_names != test.class_names:
        raise RuntimeError("Oxford trainval/test class vocabularies differ")
    expected = {"trainval_examples": 3680, "test_examples": 3669, "classes": 37}
    actual = {
        "trainval_examples": len(train),
        "test_examples": len(test),
        "classes": train.num_classes,
    }
    if actual != expected:
        raise RuntimeError(f"unexpected Oxford split statistics: {actual}")
    return actual


CUB_ARCHIVE_URL = (
    "https://data.caltech.edu/records/65de6-vp158/files/"
    "CUB_200_2011.tgz?download=1"
)
CUB_ARCHIVE_MD5 = "97eceeb196236b17998738112f37df78"
CUB_ARCHIVE_NAME = "CUB_200_2011.tgz"


@dataclass(frozen=True)
class CUBRecord:
    image_id: int
    relative_path: str
    class_index: int
    class_name: str
    is_train: bool


def _read_indexed_text(path: Path) -> dict[int, str]:
    records: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        index, value = line.split(maxsplit=1)
        records[int(index)] = value
    return records


def _cub_root(root: Path | str) -> Path:
    supplied = Path(root).expanduser().resolve()
    if supplied.name == "CUB_200_2011":
        return supplied
    return supplied / "CUB_200_2011"


def _cub_attribute_names_path(dataset_root: Path) -> Path:
    # The official 2011 tgz places attributes.txt at the archive root even
    # though its README documents attributes/attributes.txt.
    candidates = [
        dataset_root / "attributes" / "attributes.txt",
        dataset_root / "attributes.txt",
        dataset_root.parent / "attributes.txt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


class CUB2002011(Dataset):
    """Official CUB image split with per-image binary attribute annotations."""

    def __init__(self, root: Path | str, split: str):
        if split not in {"train", "test", "all"}:
            raise ValueError("CUB split must be 'train', 'test', or 'all'")
        dataset_root = _cub_root(root)
        attribute_names_path = _cub_attribute_names_path(dataset_root)
        required = [
            dataset_root / "images.txt",
            dataset_root / "classes.txt",
            dataset_root / "image_class_labels.txt",
            dataset_root / "train_test_split.txt",
            attribute_names_path,
            dataset_root / "attributes" / "image_attribute_labels.txt",
            dataset_root / "attributes" / "certainties.txt",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"missing CUB files: {missing}; run `canonical-study prepare-cub`"
            )

        images = _read_indexed_text(dataset_root / "images.txt")
        class_names_raw = _read_indexed_text(dataset_root / "classes.txt")
        class_labels = {
            index: int(value)
            for index, value in _read_indexed_text(
                dataset_root / "image_class_labels.txt"
            ).items()
        }
        split_flags = {
            index: bool(int(value))
            for index, value in _read_indexed_text(
                dataset_root / "train_test_split.txt"
            ).items()
        }
        self.class_names = [
            class_names_raw[index].split(".", 1)[-1].replace("_", " ").lower()
            for index in sorted(class_names_raw)
        ]
        attribute_names_raw = _read_indexed_text(attribute_names_path)
        self.attribute_names = [
            attribute_names_raw[index] for index in sorted(attribute_names_raw)
        ]
        certainties = _read_indexed_text(
            dataset_root / "attributes" / "certainties.txt"
        )
        not_visible_ids = {
            index for index, name in certainties.items() if "not visible" in name.lower()
        }
        if len(not_visible_ids) != 1:
            raise RuntimeError(
                f"expected one 'not visible' certainty code, found {certainties}"
            )

        attribute_rows = np.loadtxt(
            dataset_root / "attributes" / "image_attribute_labels.txt",
            dtype=np.int32,
            usecols=(0, 1, 2, 3),
        )
        expected_rows = len(images) * len(self.attribute_names)
        if attribute_rows.shape != (expected_rows, 4):
            raise RuntimeError(
                "unexpected CUB attribute table shape: "
                f"{attribute_rows.shape}, expected {(expected_rows, 4)}"
            )
        image_indices = attribute_rows[:, 0] - 1
        attribute_indices = attribute_rows[:, 1] - 1
        attributes = np.zeros(
            (len(images), len(self.attribute_names)), dtype=np.float32
        )
        attribute_mask = np.zeros_like(attributes, dtype=np.bool_)
        attributes[image_indices, attribute_indices] = attribute_rows[:, 2]
        attribute_mask[image_indices, attribute_indices] = ~np.isin(
            attribute_rows[:, 3], list(not_visible_ids)
        )
        self.attributes = torch.from_numpy(attributes)
        self.attribute_mask = torch.from_numpy(attribute_mask)
        self.dataset_root = dataset_root
        self.records: list[CUBRecord] = []
        for image_id in sorted(images):
            is_train = split_flags[image_id]
            if split != "all" and is_train != (split == "train"):
                continue
            class_id = class_labels[image_id]
            self.records.append(
                CUBRecord(
                    image_id=image_id,
                    relative_path=images[image_id],
                    class_index=class_id - 1,
                    class_name=self.class_names[class_id - 1],
                    is_train=is_train,
                )
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        image = Image.open(
            self.dataset_root / "images" / record.relative_path
        ).convert("RGB")
        attribute_index = record.image_id - 1
        return {
            "image": image,
            "image_id": record.image_id,
            "image_path": record.relative_path,
            "label": record.class_index,
            "class_name": record.class_name,
            "text": record.class_name,
            "attributes": self.attributes[attribute_index],
            "attribute_mask": self.attribute_mask[attribute_index],
        }


def validate_cub(root: Path | str) -> dict:
    train = CUB2002011(root, "train")
    test = CUB2002011(root, "test")
    expected = {
        "train_examples": 5994,
        "test_examples": 5794,
        "classes": 200,
        "attributes": 312,
    }
    actual = {
        "train_examples": len(train),
        "test_examples": len(test),
        "classes": len(train.class_names),
        "attributes": len(train.attribute_names),
    }
    if actual != expected:
        raise RuntimeError(f"unexpected CUB statistics: {actual}")
    train_ids = {record.image_id for record in train.records}
    test_ids = {record.image_id for record in test.records}
    if train_ids & test_ids:
        raise RuntimeError("official CUB train and test image IDs overlap")
    if train.class_names != test.class_names:
        raise RuntimeError("CUB train/test class vocabularies differ")
    valid = train.attribute_mask
    positives = (train.attributes.bool() & valid).sum(dim=0)
    negatives = ((~train.attributes.bool()) & valid).sum(dim=0)
    certainty_masked = int((~valid).sum().item())
    return {
        **actual,
        "observed_train_attribute_labels": int(valid.sum().item()),
        "masked_not_visible_train_labels": certainty_masked,
        "train_attributes_with_positive_and_negative_labels": int(
            ((positives > 0) & (negatives > 0)).sum().item()
        ),
    }


def prepare_cub(root: Path | str, *, accepted_research_terms: bool) -> dict:
    """Download and verify the official archive after explicit terms acceptance."""
    if not accepted_research_terms:
        raise ValueError(
            "CUB images may be used only under the dataset's stated terms; "
            "pass --accept-research-terms after reviewing them"
        )
    from torchvision.datasets.utils import download_and_extract_archive

    destination = Path(root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    download_and_extract_archive(
        CUB_ARCHIVE_URL,
        download_root=str(destination),
        filename=CUB_ARCHIVE_NAME,
        md5=CUB_ARCHIVE_MD5,
    )
    return validate_cub(destination)
