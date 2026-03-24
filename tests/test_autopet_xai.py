from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk

from autopet_xai.xai import _select_review_case_ids


def _write_binary_volume(path: Path, array: np.ndarray) -> None:
    image = sitk.GetImageFromArray(array.astype(np.uint8))
    image.SetSpacing((2.0, 2.0, 2.0))
    sitk.WriteImage(image, str(path))


def test_select_review_case_ids_balances_positive_and_negative_cases(tmp_path: Path) -> None:
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    case_mapping = {"cases": {}, "review_case_ids": []}

    case_specs = [
        ("case_pos_big", 120, 100),
        ("case_pos_small", 30, 60),
        ("case_neg_bigfp", 0, 90),
        ("case_neg_smallfp", 0, 20),
    ]

    for index, (case_id, gt_voxels, pred_voxels) in enumerate(case_specs):
        case_dir = tmp_path / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        label = np.zeros((4, 8, 8), dtype=np.uint8)
        pred = np.zeros_like(label)
        if gt_voxels > 0:
            label.reshape(-1)[:gt_voxels] = 1
        if pred_voxels > 0:
            pred.reshape(-1)[:pred_voxels] = 1

        label_path = case_dir / "label.nii.gz"
        prediction_path = prediction_dir / f"FDG_{index:04d}.nii.gz"
        _write_binary_volume(label_path, label)
        _write_binary_volume(prediction_path, pred)

        case_mapping["cases"][case_id] = {
            "label": str(label_path),
            "nnunet_case_id": f"FDG_{index:04d}",
        }
        case_mapping["review_case_ids"].append(case_id)

    selected_case_ids, summary = _select_review_case_ids(
        case_mapping=case_mapping,
        prediction_dir=prediction_dir,
        max_cases=4,
        balance_classes=True,
    )

    assert len(selected_case_ids) == 4
    assert summary["available_positive_case_count"] == 2
    assert summary["available_negative_case_count"] == 2
    assert {"case_pos_big", "case_pos_small"}.issubset(set(selected_case_ids))
    assert {"case_neg_bigfp", "case_neg_smallfp"}.issubset(set(selected_case_ids))
