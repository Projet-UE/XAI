from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .utils import ensure_dir, load_json, save_json

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def scan_image_folder(data_root: Union[str, Path]) -> Tuple[List[str], List[Dict[str, Any]]]:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    class_dirs = sorted([path for path in root.iterdir() if path.is_dir()])
    if not class_dirs:
        raise ValueError(f"No class folders found under: {root}")

    classes = [directory.name for directory in class_dirs]
    class_to_idx = {name: index for index, name in enumerate(classes)}
    records: List[Dict[str, Any]] = []

    for class_name in classes:
        class_dir = root / class_name
        for file_path in sorted(class_dir.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append(
                    {
                        "path": str(file_path.relative_to(root).as_posix()),
                        "label": class_to_idx[class_name],
                        "class_name": class_name,
                    }
                )

    if not records:
        raise ValueError(f"No image files found under: {root}")

    return classes, records


def _stratify_or_none(labels: List[int]) -> Optional[List[int]]:
    counts = Counter(labels)
    if len(counts) < 2:
        return None
    if min(counts.values()) < 2:
        return None
    return labels


def create_split_manifest(
    data_root: Union[str, Path],
    output_path: Union[str, Path],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, Any]:
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be >= 0 and sum to < 1.")

    classes, records = scan_image_folder(data_root)
    labels = [record["label"] for record in records]
    stratify_labels = _stratify_or_none(labels)

    train_records, temp_records = train_test_split(
        records,
        test_size=val_ratio + test_ratio,
        random_state=seed,
        stratify=stratify_labels,
    )

    if not temp_records:
        raise ValueError("Split produced an empty validation/test subset.")

    if test_ratio == 0:
        val_records = temp_records
        test_records: List[Dict[str, Any]] = []
    else:
        temp_labels = [record["label"] for record in temp_records]
        temp_stratify = _stratify_or_none(temp_labels)
        relative_test_ratio = test_ratio / (val_ratio + test_ratio)
        val_records, test_records = train_test_split(
            temp_records,
            test_size=relative_test_ratio,
            random_state=seed,
            stratify=temp_stratify,
        )

    manifest = {
        "data_root": str(Path(data_root)),
        "seed": seed,
        "classes": classes,
        "splits": {
            "train": train_records,
            "val": val_records,
            "test": test_records,
        },
    }
    save_json(manifest, output_path)
    return manifest


def ensure_split_manifest(
    data_root: Union[str, Path],
    manifest_path: Union[str, Path],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, Any]:
    target = Path(manifest_path)
    if target.exists():
        return load_json(target)
    ensure_dir(target.parent)
    return create_split_manifest(data_root, target, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)


def load_manifest(manifest_path: Union[str, Path]) -> Dict[str, Any]:
    return load_json(manifest_path)


class BrainTumorDataset(Dataset[Dict[str, Any]]):
    def __init__(
        self,
        data_root: Union[str, Path],
        manifest: Dict[str, Any],
        split: str,
        image_size: int = 224,
        augment: bool = False,
    ) -> None:
        self.data_root = Path(data_root)
        self.classes = manifest["classes"]
        self.records = manifest["splits"][split]
        self.transform = self._build_transform(image_size=image_size, augment=augment)

    @staticmethod
    def _build_transform(image_size: int, augment: bool) -> transforms.Compose:
        transform_steps: List[Any] = [transforms.Resize((image_size, image_size))]
        if augment:
            transform_steps.extend(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(degrees=10),
                ]
            )
        transform_steps.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        return transforms.Compose(transform_steps)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        image_path = self.data_root / record["path"]
        image = Image.open(image_path).convert("RGB")
        return {
            "image": self.transform(image),
            "label": torch.tensor(record["label"], dtype=torch.float32),
            "path": record["path"],
            "class_name": record["class_name"],
        }


def build_dataloaders(
    data_root: Union[str, Path],
    manifest: Dict[str, Any],
    image_size: int = 224,
    batch_size: int = 8,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    datasets = {
        "train": BrainTumorDataset(data_root, manifest, split="train", image_size=image_size, augment=True),
        "val": BrainTumorDataset(data_root, manifest, split="val", image_size=image_size, augment=False),
        "test": BrainTumorDataset(data_root, manifest, split="test", image_size=image_size, augment=False),
    }

    return {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
        )
        for split, dataset in datasets.items()
        if len(dataset) > 0
    }
