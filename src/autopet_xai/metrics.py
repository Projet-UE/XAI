from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

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


def _connected_component_image(mask: np.ndarray, reference_image: sitk.Image) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    component_image = sitk.ConnectedComponent(sitk.GetImageFromArray(mask.astype(np.uint8)))
    component_image.CopyInformation(reference_image)
    component_array = sitk.GetArrayFromImage(component_image)
    spacing_xyz = tuple(float(value) for value in reference_image.GetSpacing())
    return component_array, spacing_xyz


def _component_reports(
    component_array: np.ndarray,
    pet_array: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
) -> List[Dict[str, Any]]:
    voxel_volume_ml = float(np.prod(spacing_xyz) / 1000.0)
    reports: List[Dict[str, Any]] = []
    labels = [int(label) for label in np.unique(component_array) if int(label) != 0]
    for label in labels:
        component_mask = component_array == label
        component_voxels = int(component_mask.sum())
        pet_values = pet_array[component_mask]
        reports.append(
            {
                "label": label,
                "voxels": component_voxels,
                "volume_ml": component_voxels * voxel_volume_ml,
                "mean_pet": float(pet_values.mean()) if pet_values.size else 0.0,
                "max_pet": float(pet_values.max()) if pet_values.size else 0.0,
            }
        )
    return reports


def postprocess_prediction_mask(
    prediction_path: Union[str, Path],
    pet_path: Union[str, Path],
    output_path: Union[str, Path],
    min_component_volume_ml: float = 0.0,
    max_components: Optional[int] = None,
    rank_by: str = "mean_pet",
) -> Dict[str, Any]:
    prediction_path = Path(prediction_path)
    output_path = Path(output_path)
    prediction_image = sitk.ReadImage(str(prediction_path))
    prediction_array = (sitk.GetArrayFromImage(prediction_image) > 0).astype(np.uint8)
    pet_array = sitk.GetArrayFromImage(sitk.ReadImage(str(pet_path))).astype(np.float32)
    component_array, spacing_xyz = _connected_component_image(prediction_array, prediction_image)
    reports = _component_reports(component_array, pet_array, spacing_xyz)

    if rank_by not in {"mean_pet", "max_pet", "volume_ml"}:
        raise ValueError(f"Unsupported component ranking: {rank_by}")

    kept_reports = [report for report in reports if report["volume_ml"] >= min_component_volume_ml]
    kept_reports.sort(key=lambda report: (report[rank_by], report["volume_ml"]), reverse=True)
    if max_components is not None and max_components > 0:
        kept_reports = kept_reports[:max_components]

    kept_labels = {report["label"] for report in kept_reports}
    filtered_prediction = np.isin(component_array, list(kept_labels)).astype(np.uint8)
    filtered_image = sitk.GetImageFromArray(filtered_prediction)
    filtered_image.CopyInformation(prediction_image)
    ensure_dir(output_path.parent)
    sitk.WriteImage(filtered_image, str(output_path))

    voxel_volume_ml = float(np.prod(spacing_xyz) / 1000.0)
    return {
        "prediction_path": str(prediction_path),
        "output_path": str(output_path),
        "component_count_before": len(reports),
        "component_count_after": len(kept_reports),
        "rank_by": rank_by,
        "min_component_volume_ml": float(min_component_volume_ml),
        "max_components": int(max_components) if max_components is not None else None,
        "pred_voxels_before": int(prediction_array.sum()),
        "pred_voxels_after": int(filtered_prediction.sum()),
        "pred_volume_ml_before": float(prediction_array.sum() * voxel_volume_ml),
        "pred_volume_ml_after": float(filtered_prediction.sum() * voxel_volume_ml),
        "components_before": reports,
        "components_after": kept_reports,
    }


def postprocess_prediction_dir(
    case_mapping: Dict[str, Any],
    prediction_dir: Union[str, Path],
    output_dir: Union[str, Path],
    min_component_volume_ml: float = 0.0,
    max_components: Optional[int] = None,
    rank_by: str = "mean_pet",
) -> Dict[str, Any]:
    prediction_dir = Path(prediction_dir)
    output_dir = ensure_dir(output_dir)
    reports: List[Dict[str, Any]] = []

    for case_id in case_mapping["review_case_ids"]:
        record = case_mapping["cases"][case_id]
        nnunet_case_id = record["nnunet_case_id"]
        prediction_path = prediction_dir / f"{nnunet_case_id}.nii.gz"
        if not prediction_path.exists():
            raise FileNotFoundError(f"Prediction file not found for case {case_id}: {prediction_path}")

        report = postprocess_prediction_mask(
            prediction_path=prediction_path,
            pet_path=record["pet"],
            output_path=output_dir / prediction_path.name,
            min_component_volume_ml=min_component_volume_ml,
            max_components=max_components,
            rank_by=rank_by,
        )
        report.update({"case_id": case_id, "nnunet_case_id": nnunet_case_id})
        reports.append(report)

    summary = {
        "case_count": len(reports),
        "rank_by": rank_by,
        "min_component_volume_ml": float(min_component_volume_ml),
        "max_components": int(max_components) if max_components is not None else None,
        "cases": reports,
    }
    return summary
