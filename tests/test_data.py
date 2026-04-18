from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from brain_tumor_xai.data import BrainTumorDataset, create_split_manifest, ensure_split_manifest


def test_split_manifest_is_reproducible(tiny_dataset: Path, tmp_path: Path) -> None:
    manifest_a = create_split_manifest(tiny_dataset, tmp_path / "split_a.json", seed=123)
    manifest_b = create_split_manifest(tiny_dataset, tmp_path / "split_b.json", seed=123)
    assert manifest_a == manifest_b


def test_dataset_reads_images_and_labels(tiny_dataset: Path, tmp_path: Path) -> None:
    manifest = ensure_split_manifest(tiny_dataset, tmp_path / "split.json", seed=7)
    dataset = BrainTumorDataset(tiny_dataset, manifest, split="train", image_size=64, augment=False)
    sample = dataset[0]
    assert sample["image"].shape == (3, 64, 64)
    assert sample["label"].item() in {0.0, 1.0}
    assert sample["path"].endswith(".png")


def test_manifest_ignores_hidden_checkpoint_images(tiny_dataset: Path, tmp_path: Path) -> None:
    hidden_dir = tiny_dataset / "yes" / ".ipynb_checkpoints"
    hidden_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(hidden_dir / "Y_fake-checkpoint.JPG")

    manifest = create_split_manifest(tiny_dataset, tmp_path / "split_hidden.json", seed=42)
    all_paths = [record["path"] for split in manifest["splits"].values() for record in split]
    assert all(".ipynb_checkpoints" not in path for path in all_paths)
