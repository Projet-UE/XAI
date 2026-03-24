from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Union

import numpy as np
import SimpleITK as sitk

from brain_tumor_xai.utils import ensure_dir, save_json


def load_mask_and_spacing(path: Union[str, Path]) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)
    spacing_xyz = tuple(float(value) for value in image.GetSpacing())
    return (array > 0).astype(np.uint8), spacing_xyz


def compute_case_metrics(
    ground_truth_mask: np.ndarray,
    predicted_mask: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
) -> Dict[str, Any]:
    gt = ground_truth_mask.astype(bool)
    pred = predicted_mask.astype(bool)
    intersection = np.logical_and(gt, pred).sum()
    gt_volume_voxels = int(gt.sum())
    pred_volume_voxels = int(pred.sum())
    denom = gt_volume_voxels + pred_volume_voxels
    dice = 1.0 if denom == 0 else float((2.0 * intersection) / denom)

    voxel_volume_ml = float(np.prod(spacing_xyz) / 1000.0)
    false_negative_voxels = int(np.logical_and(gt, np.logical_not(pred)).sum())
    false_positive_voxels = int(np.logical_and(np.logical_not(gt), pred).sum())

    return {
        "dice": dice,
        "gt_voxels": gt_volume_voxels,
        "pred_voxels": pred_volume_voxels,
        "false_negative_voxels": false_negative_voxels,
        "false_positive_voxels": false_positive_voxels,
        "gt_volume_ml": gt_volume_voxels * voxel_volume_ml,
        "pred_volume_ml": pred_volume_voxels * voxel_volume_ml,
        "false_negative_volume_ml": false_negative_voxels * voxel_volume_ml,
        "false_positive_volume_ml": false_positive_voxels * voxel_volume_ml,
        "spacing_xyz": [float(value) for value in spacing_xyz],
    }


def evaluate_review_predictions(
    case_mapping: Dict[str, Any],
    prediction_dir: Union[str, Path],
) -> Dict[str, Any]:
    prediction_dir = Path(prediction_dir)
    per_case: List[Dict[str, Any]] = []
    review_cases = [case_id for case_id in case_mapping["review_case_ids"]]

    for case_id in review_cases:
        record = case_mapping["cases"][case_id]
        nnunet_case_id = record["nnunet_case_id"]
        prediction_path = prediction_dir / f"{nnunet_case_id}.nii.gz"
        if not prediction_path.exists():
            raise FileNotFoundError(f"Prediction file not found for case {case_id}: {prediction_path}")

        gt_mask, spacing_xyz = load_mask_and_spacing(record["label"])
        pred_mask, _ = load_mask_and_spacing(prediction_path)
        metrics = compute_case_metrics(gt_mask, pred_mask, spacing_xyz)
        per_case.append(
            {
                "case_id": case_id,
                "nnunet_case_id": nnunet_case_id,
                "prediction_path": str(prediction_path),
                "ground_truth_path": record["label"],
                **metrics,
            }
        )

    summary = {
        "case_count": len(per_case),
        "mean_dice": float(np.mean([case["dice"] for case in per_case])) if per_case else 0.0,
        "mean_false_negative_volume_ml": float(np.mean([case["false_negative_volume_ml"] for case in per_case]))
        if per_case
        else 0.0,
        "mean_false_positive_volume_ml": float(np.mean([case["false_positive_volume_ml"] for case in per_case]))
        if per_case
        else 0.0,
        "cases": per_case,
    }
    return summary


def save_segmentation_report(metrics: Dict[str, Any], output_dir: Union[str, Path]) -> None:
    output_dir = ensure_dir(output_dir)
    save_json(metrics, output_dir / "segmentation_metrics.json")
    save_json({"cases": metrics["cases"]}, output_dir / "per_case_metrics.json")

