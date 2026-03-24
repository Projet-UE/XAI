from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk

from autopet_xai.metrics import compute_case_metrics, postprocess_prediction_mask


def _write_volume(path: Path, array: np.ndarray) -> None:
    image = sitk.GetImageFromArray(array.astype(np.float32))
    image.SetSpacing((2.0, 2.0, 2.0))
    sitk.WriteImage(image, str(path))


def test_segmentation_metrics_include_dice_and_false_volumes() -> None:
    ground_truth = np.zeros((4, 8, 8), dtype=np.uint8)
    prediction = np.zeros_like(ground_truth)
    ground_truth[:, 2:4, 2:4] = 1
    prediction[:, 3:5, 2:4] = 1

    metrics = compute_case_metrics(ground_truth, prediction, spacing_xyz=(2.0, 2.0, 2.0))
    assert 0.0 <= metrics["dice"] <= 1.0
    assert metrics["false_negative_voxels"] > 0
    assert metrics["false_positive_voxels"] > 0
    assert metrics["false_negative_volume_ml"] > 0
    assert metrics["false_positive_volume_ml"] > 0


def test_postprocess_prediction_mask_keeps_hottest_components(tmp_path: Path) -> None:
    prediction = np.zeros((4, 16, 16), dtype=np.uint8)
    pet = np.zeros_like(prediction, dtype=np.float32)

    prediction[:, 1:5, 1:5] = 1
    prediction[:, 9:13, 9:13] = 1
    pet[:, 1:5, 1:5] = 2.0
    pet[:, 9:13, 9:13] = 8.0

    prediction_path = tmp_path / "prediction.nii.gz"
    pet_path = tmp_path / "pet.nii.gz"
    output_path = tmp_path / "prediction_filtered.nii.gz"
    _write_volume(prediction_path, prediction)
    _write_volume(pet_path, pet)

    report = postprocess_prediction_mask(
        prediction_path=prediction_path,
        pet_path=pet_path,
        output_path=output_path,
        min_component_volume_ml=0.0,
        max_components=1,
        rank_by="mean_pet",
    )

    filtered = sitk.GetArrayFromImage(sitk.ReadImage(str(output_path)))
    assert report["component_count_before"] == 2
    assert report["component_count_after"] == 1
    assert int(filtered.sum()) == int(prediction[:, 9:13, 9:13].sum())
