from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk

from autopet_xai.data import build_fdg_manifest, create_versioned_fdg_splits, export_nnunet_dataset


def _write_volume(path: Path, array: np.ndarray) -> None:
    image = sitk.GetImageFromArray(array.astype(np.float32))
    image.SetSpacing((2.0, 2.0, 2.0))
    sitk.WriteImage(image, str(path))


def _build_source_case(case_dir: Path, positive: bool) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    pet = np.zeros((8, 16, 16), dtype=np.float32)
    ct = np.zeros((8, 16, 16), dtype=np.float32)
    label = np.zeros((8, 16, 16), dtype=np.uint8)
    if positive:
        pet[3:5, 6:10, 6:10] = 4.0
        ct[3:5, 6:10, 6:10] = 150.0
        label[3:5, 6:10, 6:10] = 1
    _write_volume(case_dir / "PET.nii.gz", pet)
    _write_volume(case_dir / "CT.nii.gz", ct)
    _write_volume(case_dir / "SEG.nii.gz", label)


def test_fdg_manifest_split_and_nnunet_export(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    prepared_root = tmp_path / "prepared"
    for index in range(64):
        _build_source_case(source_root / f"PETCT_{index:03d}", positive=(index % 2 == 0))

    manifest = build_fdg_manifest(source_root, prepared_root, tmp_path / "fdg_manifest.json", seed=42, link_mode="copy")
    assert manifest["case_count"] == 64
    assert manifest["positive_case_count"] == 32
    assert manifest["negative_case_count"] == 32

    splits = create_versioned_fdg_splits(manifest, tmp_path / "splits", seed=42, dev_counts=(48, 8, 8))
    fdg_dev = splits["fdg_dev"]
    assert fdg_dev["train"]["count"] == 48
    assert fdg_dev["val"]["count"] == 8
    assert fdg_dev["review"]["count"] == 8

    mapping = export_nnunet_dataset(manifest, fdg_dev, dataset_id=501, nnunet_raw_root=tmp_path / "nnunet_raw", link_mode="copy")
    dataset_root = tmp_path / "nnunet_raw" / "Dataset501_AutoPETFDG"
    assert (dataset_root / "dataset.json").exists()
    assert (dataset_root / "case_mapping.json").exists()
    assert len(list((dataset_root / "imagesTr").glob("*_0000.nii.gz"))) == 56
    assert len(list((dataset_root / "imagesTs").glob("*_0000.nii.gz"))) == 8
    assert mapping["split_name"] == "fdg_dev"
