from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

os.environ["MPLBACKEND"] = "Agg"

from captum.attr import IntegratedGradients, Occlusion, Saliency
import matplotlib
import numpy as np
import SimpleITK as sitk
import torch

from brain_tumor_xai.utils import ensure_dir, save_json

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:  # pragma: no cover - depends on optional remote dependency wiring
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
except ImportError:  # pragma: no cover - handled at runtime
    nnUNetPredictor = None  # type: ignore[assignment]


def _read_volume(path: Union[str, Path]) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)


def _count_positive_voxels(path: Union[str, Path]) -> int:
    return int((_read_volume(path) > 0).sum())


def _normalize_channel(volume: np.ndarray) -> np.ndarray:
    finite = np.isfinite(volume)
    if not finite.any():
        return np.zeros_like(volume, dtype=np.float32)
    values = volume[finite]
    lower, upper = np.percentile(values, [1, 99])
    clipped = np.clip(volume, lower, upper)
    std = float(clipped.std())
    if std == 0:
        return clipped - float(clipped.mean())
    return (clipped - float(clipped.mean())) / std


def _compute_bbox(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.argwhere(mask > 0)
    if indices.size == 0:
        center = np.array(mask.shape) // 2
        return center, center
    return indices.min(axis=0), indices.max(axis=0)


def _crop_with_padding(volume: np.ndarray, center_zyx: Sequence[int], crop_size: Tuple[int, int, int]) -> np.ndarray:
    slices = []
    pad_width = []
    for dim, center, size in zip(volume.shape, center_zyx, crop_size):
        half = size // 2
        start = int(center - half)
        end = start + size
        pad_before = max(0, -start)
        pad_after = max(0, end - dim)
        start = max(0, start)
        end = min(dim, end)
        slices.append(slice(start, end))
        pad_width.append((pad_before, pad_after))
    cropped = volume[tuple(slices)]
    if any(before or after for before, after in pad_width):
        cropped = np.pad(cropped, pad_width, mode="constant")
    return cropped.astype(np.float32)


def _select_target_mask(label: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    if np.count_nonzero(label) > 0:
        return label
    if np.count_nonzero(prediction) > 0:
        return prediction
    return np.zeros_like(label, dtype=np.uint8)


def _choose_representative_slices(mask_crop: np.ndarray, count: int = 3) -> List[int]:
    positive_slices = np.where(mask_crop.reshape(mask_crop.shape[0], -1).sum(axis=1) > 0)[0]
    if positive_slices.size == 0:
        center = mask_crop.shape[0] // 2
        return [max(0, center - 1), center, min(mask_crop.shape[0] - 1, center + 1)]
    if positive_slices.size <= count:
        return [int(index) for index in positive_slices.tolist()]
    positions = np.linspace(0, len(positive_slices) - 1, num=count)
    return [int(positive_slices[int(round(position))]) for position in positions]


def _select_review_case_ids(
    case_mapping: Dict[str, Any],
    prediction_dir: Union[str, Path],
    max_cases: int,
    balance_classes: bool = True,
) -> Tuple[List[str], Dict[str, Any]]:
    prediction_dir = Path(prediction_dir)
    candidates: List[Dict[str, Any]] = []

    for case_id in case_mapping["review_case_ids"]:
        record = case_mapping["cases"][case_id]
        prediction_path = prediction_dir / f"{record['nnunet_case_id']}.nii.gz"
        if not prediction_path.exists():
            continue
        gt_voxels = _count_positive_voxels(record["label"])
        pred_voxels = _count_positive_voxels(prediction_path)
        candidates.append(
            {
                "case_id": case_id,
                "positive": gt_voxels > 0,
                "gt_voxels": gt_voxels,
                "pred_voxels": pred_voxels,
            }
        )

    positives = sorted(
        [candidate for candidate in candidates if candidate["positive"]],
        key=lambda candidate: (candidate["gt_voxels"], candidate["pred_voxels"]),
        reverse=True,
    )
    negatives = sorted(
        [candidate for candidate in candidates if not candidate["positive"]],
        key=lambda candidate: candidate["pred_voxels"],
        reverse=True,
    )

    if not balance_classes:
        ordered = positives + negatives
        ordered = sorted(
            ordered,
            key=lambda candidate: (candidate["positive"], candidate["gt_voxels"], candidate["pred_voxels"]),
            reverse=True,
        )
        selected = ordered[:max_cases]
    else:
        positive_target = min(len(positives), max_cases // 2 if max_cases % 2 == 0 else (max_cases + 1) // 2)
        negative_target = min(len(negatives), max_cases - positive_target)
        selected = positives[:positive_target] + negatives[:negative_target]

        remaining = positives[positive_target:] + negatives[negative_target:]
        remaining = sorted(
            remaining,
            key=lambda candidate: (candidate["positive"], candidate["gt_voxels"], candidate["pred_voxels"]),
            reverse=True,
        )
        if len(selected) < max_cases:
            selected.extend(remaining[: max_cases - len(selected)])

    selected_positive_count = int(sum(1 for candidate in selected if candidate["positive"]))
    selected_negative_count = int(sum(1 for candidate in selected if not candidate["positive"]))

    return [candidate["case_id"] for candidate in selected], {
        "balanced_classes": balance_classes,
        "available_case_count": len(candidates),
        "available_positive_case_count": len(positives),
        "available_negative_case_count": len(negatives),
        "selected_case_count": len(selected),
        "selected_positive_case_count": selected_positive_count,
        "selected_negative_case_count": selected_negative_count,
        "selected_case_ids": [candidate["case_id"] for candidate in selected],
    }


def _foreground_score(output: Any) -> torch.Tensor:
    if isinstance(output, (list, tuple)):
        output = output[0]
    if output.ndim != 5:
        raise ValueError(f"Expected 5D segmentation logits, got shape {tuple(output.shape)}")
    if output.shape[1] == 1:
        foreground = output[:, 0]
    else:
        foreground = output[:, 1]
    return foreground.reshape(foreground.shape[0], -1).sum(dim=1)


def _compute_attribution(network: torch.nn.Module, input_tensor: torch.Tensor, method: str) -> np.ndarray:
    input_tensor = input_tensor.clone().detach().requires_grad_(True)
    forward = lambda tensor: _foreground_score(network(tensor))
    if method == "saliency":
        attr = Saliency(forward).attribute(input_tensor)
    elif method == "integrated_gradients":
        attr = IntegratedGradients(forward).attribute(input_tensor, baselines=torch.zeros_like(input_tensor), n_steps=16)
    elif method == "occlusion":
        attr = Occlusion(forward).attribute(
            input_tensor,
            baselines=0,
            strides=(1, 1, 8, 8),
            sliding_window_shapes=(1, 2, 16, 16),
        )
    else:  # pragma: no cover - CLI restricts choices
        raise ValueError(f"Unsupported XAI method: {method}")

    volume = attr.detach().cpu().abs().mean(dim=1).squeeze(0).numpy()
    volume -= volume.min()
    max_value = volume.max()
    if max_value > 0:
        volume /= max_value
    return volume


def _summarize_attribution(
    attribution_crop: np.ndarray,
    label_crop: np.ndarray,
    pred_crop: np.ndarray,
) -> Dict[str, Any]:
    attr = np.clip(attribution_crop.astype(np.float32), a_min=0.0, a_max=None)
    gt_mask = label_crop.astype(bool)
    pred_mask = pred_crop.astype(bool)
    union_mask = np.logical_or(gt_mask, pred_mask)
    total_mass = float(attr.sum())
    top_threshold = float(np.percentile(attr, 90)) if attr.size else 0.0
    top_mask = attr >= top_threshold if top_threshold > 0 else attr > 0

    def _mass_ratio(mask: np.ndarray) -> Optional[float]:
        if total_mass <= 0:
            return None
        return float(attr[mask].sum() / total_mass)

    def _mean_attr(mask: np.ndarray) -> Optional[float]:
        if not np.any(mask):
            return None
        return float(attr[mask].mean())

    def _top_ratio(mask: np.ndarray) -> Optional[float]:
        top_count = int(top_mask.sum())
        if top_count == 0:
            return None
        return float(np.logical_and(top_mask, mask).sum() / top_count)

    return {
        "total_positive_attribution_mass": total_mass,
        "gt_positive_voxels": int(gt_mask.sum()),
        "pred_positive_voxels": int(pred_mask.sum()),
        "mass_ratio_inside_gt": _mass_ratio(gt_mask),
        "mass_ratio_inside_prediction": _mass_ratio(pred_mask),
        "mass_ratio_inside_union": _mass_ratio(union_mask),
        "mean_attr_inside_gt": _mean_attr(gt_mask),
        "mean_attr_outside_gt": _mean_attr(~gt_mask),
        "mean_attr_inside_prediction": _mean_attr(pred_mask),
        "mean_attr_outside_prediction": _mean_attr(~pred_mask),
        "top10_ratio_inside_gt": _top_ratio(gt_mask),
        "top10_ratio_inside_prediction": _top_ratio(pred_mask),
    }


def _draw_mask_contour(axis: Any, mask_slice: np.ndarray, color: str) -> None:
    binary = (mask_slice > 0).astype(np.uint8)
    if np.count_nonzero(binary) == 0:
        return
    if np.count_nonzero(binary == 0) == 0:
        return
    axis.contour(binary, levels=[0.5], colors=[color], linewidths=1.2, alpha=0.9)


def _render_binary_panel(
    axis: Any,
    mask_slice: np.ndarray,
    *,
    title: str,
    cmap: str,
    empty_text: str,
) -> None:
    axis.set_facecolor("#eef1f3")
    axis.imshow(np.zeros_like(mask_slice), cmap="gray", vmin=0, vmax=1, alpha=0.06)
    masked = np.ma.masked_where(mask_slice <= 0, mask_slice)
    axis.imshow(masked, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    axis.set_title(title)
    if np.count_nonzero(mask_slice) == 0:
        axis.text(
            0.5,
            0.5,
            empty_text,
            ha="center",
            va="center",
            fontsize=9,
            color="#6b7280",
            transform=axis.transAxes,
        )


def _build_error_map(gt_slice: np.ndarray, pred_slice: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
    gt = gt_slice > 0
    pred = pred_slice > 0

    tp = np.logical_and(gt, pred)
    fn = np.logical_and(gt, np.logical_not(pred))
    fp = np.logical_and(np.logical_not(gt), pred)

    rgb = np.full((*gt.shape, 3), 0.93, dtype=np.float32)
    rgb[tp] = np.array([0.12, 0.69, 0.22], dtype=np.float32)
    rgb[fn] = np.array([0.86, 0.18, 0.18], dtype=np.float32)
    rgb[fp] = np.array([0.20, 0.43, 0.95], dtype=np.float32)

    return rgb, {
        "tp": int(tp.sum()),
        "fn": int(fn.sum()),
        "fp": int(fp.sum()),
    }


def _save_case_panel(
    pet_crop: np.ndarray,
    ct_crop: np.ndarray,
    label_crop: np.ndarray,
    pred_crop: np.ndarray,
    attribution_crop: np.ndarray,
    slice_indices: Sequence[int],
    output_path: Union[str, Path],
    title: str,
    case_summary: Optional[str] = None,
) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    rows = len(slice_indices)
    fig, axes = plt.subplots(rows, 5, figsize=(18, 3.6 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_index, slice_index in enumerate(slice_indices):
        pet_slice = pet_crop[slice_index]
        ct_slice = ct_crop[slice_index]
        gt_slice = label_crop[slice_index]
        pred_slice = pred_crop[slice_index]
        attr_slice = attribution_crop[slice_index]
        error_map, error_counts = _build_error_map(gt_slice, pred_slice)

        axes[row_index, 0].imshow(pet_slice, cmap="inferno")
        axes[row_index, 0].imshow(attr_slice, cmap="magma", alpha=0.45)
        _draw_mask_contour(axes[row_index, 0], gt_slice, color="#22c55e")
        _draw_mask_contour(axes[row_index, 0], pred_slice, color="#3b82f6")
        axes[row_index, 0].set_title(f"PET + {title}\nSlice {slice_index}")

        axes[row_index, 1].imshow(ct_slice, cmap="gray")
        axes[row_index, 1].imshow(attr_slice, cmap="magma", alpha=0.45)
        _draw_mask_contour(axes[row_index, 1], gt_slice, color="#22c55e")
        _draw_mask_contour(axes[row_index, 1], pred_slice, color="#3b82f6")
        axes[row_index, 1].set_title("CT + attribution")

        _render_binary_panel(
            axes[row_index, 2],
            gt_slice,
            title="Ground truth",
            cmap="Greens",
            empty_text="No GT voxels\non this slice",
        )
        _render_binary_panel(
            axes[row_index, 3],
            pred_slice,
            title="Prediction",
            cmap="Blues",
            empty_text="No predicted voxels\non this slice",
        )

        axes[row_index, 4].imshow(error_map, interpolation="nearest")
        axes[row_index, 4].set_title("TP / FN / FP")
        axes[row_index, 4].text(
            0.5,
            0.04,
            f"TP {error_counts['tp']}  FN {error_counts['fn']}  FP {error_counts['fp']}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#111827",
            transform=axes[row_index, 4].transAxes,
        )

        for column in range(5):
            axes[row_index, column].axis("off")

    if case_summary:
        fig.suptitle(case_summary, fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
    else:
        fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _load_predictor(
    training_output_dir: Union[str, Path],
    fold: Union[int, str],
    checkpoint_name: str,
    device: Union[str, torch.device],
) -> Any:
    if nnUNetPredictor is None:
        raise ImportError("nnUNetv2 is required for autoPET XAI generation.")

    torch_device = torch.device(device)
    if torch_device.type == "cpu":
        torch.backends.mkldnn.enabled = False
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=(torch_device.type == "cuda"),
        device=torch_device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    normalized_fold: Union[int, str]
    fold_text = str(fold)
    normalized_fold = int(fold_text) if fold_text.isdigit() else fold_text
    predictor.initialize_from_trained_model_folder(
        str(training_output_dir),
        use_folds=(normalized_fold,),
        checkpoint_name=checkpoint_name,
    )
    return predictor


def generate_review_xai(
    case_mapping: Dict[str, Any],
    training_output_dir: Union[str, Path],
    prediction_dir: Union[str, Path],
    output_dir: Union[str, Path],
    fold: Union[int, str] = 0,
    checkpoint_name: str = "checkpoint_best.pth",
    device: Union[str, torch.device] = "cpu",
    methods: Sequence[str] = ("saliency", "integrated_gradients", "occlusion"),
    max_cases: int = 4,
    crop_size: Tuple[int, int, int] = (32, 128, 128),
    balance_classes: bool = True,
) -> Dict[str, Any]:
    torch_device = torch.device(device)
    if torch_device.type == "cpu":
        torch.backends.mkldnn.enabled = False

    predictor = _load_predictor(training_output_dir, fold=fold, checkpoint_name=checkpoint_name, device=device)
    network = predictor.network
    network.eval()
    network.to(torch_device)

    output_dir = ensure_dir(output_dir)
    exported: List[Dict[str, Any]] = []
    review_case_ids, selection_summary = _select_review_case_ids(
        case_mapping=case_mapping,
        prediction_dir=prediction_dir,
        max_cases=max_cases,
        balance_classes=balance_classes,
    )

    for case_id in review_case_ids:
        record = case_mapping["cases"][case_id]
        nnunet_case_id = record["nnunet_case_id"]
        prediction_path = Path(prediction_dir) / f"{nnunet_case_id}.nii.gz"
        if not prediction_path.exists():
            continue

        pet = _read_volume(record["pet"])
        ct = _read_volume(record["ct"])
        label = (_read_volume(record["label"]) > 0).astype(np.uint8)
        prediction = (_read_volume(prediction_path) > 0).astype(np.uint8)
        target_mask = _select_target_mask(label, prediction)
        min_corner, max_corner = _compute_bbox(target_mask)
        center = ((min_corner + max_corner) / 2.0).round().astype(int)

        pet_crop = _crop_with_padding(pet, center, crop_size)
        ct_crop = _crop_with_padding(ct, center, crop_size)
        label_crop = _crop_with_padding(label.astype(np.float32), center, crop_size).astype(np.uint8)
        pred_crop = _crop_with_padding(prediction.astype(np.float32), center, crop_size).astype(np.uint8)

        input_tensor = np.stack([_normalize_channel(pet_crop), _normalize_channel(ct_crop)], axis=0)
        input_tensor_torch = torch.from_numpy(input_tensor).unsqueeze(0).to(torch_device, dtype=torch.float32)
        union_crop = np.logical_or(label_crop > 0, pred_crop > 0).astype(np.uint8)
        slice_indices = _choose_representative_slices(union_crop, count=3)
        case_summary = (
            f"{case_id} | GT voxels (crop): {int(np.count_nonzero(label_crop))} | "
            f"Pred voxels (crop): {int(np.count_nonzero(pred_crop))} | "
            f"Pred positive overall: {bool(np.count_nonzero(prediction) > 0)}"
        )

        case_dir = ensure_dir(output_dir / case_id)
        method_reports: List[Dict[str, Any]] = []
        for method in methods:
            attribution = _compute_attribution(network, input_tensor_torch, method)
            figure_path = case_dir / f"{method}.png"
            _save_case_panel(
                pet_crop=pet_crop,
                ct_crop=ct_crop,
                label_crop=label_crop,
                pred_crop=pred_crop,
                attribution_crop=attribution,
                slice_indices=slice_indices,
                output_path=figure_path,
                title=method,
                case_summary=case_summary,
            )
            method_reports.append(
                {
                    "method": method,
                    "figure": str(figure_path),
                    "attribution_summary": _summarize_attribution(attribution, label_crop, pred_crop),
                }
            )

        exported.append(
            {
                "case_id": case_id,
                "nnunet_case_id": nnunet_case_id,
                "prediction_path": str(prediction_path),
                "ground_truth_positive": bool(np.count_nonzero(label) > 0),
                "prediction_positive": bool(np.count_nonzero(prediction) > 0),
                "ground_truth_voxels": int(np.count_nonzero(label)),
                "prediction_voxels": int(np.count_nonzero(prediction)),
                "ground_truth_voxels_in_crop": int(np.count_nonzero(label_crop)),
                "prediction_voxels_in_crop": int(np.count_nonzero(pred_crop)),
                "slice_indices": [int(index) for index in slice_indices],
                "methods": method_reports,
            }
        )

    report = {"cases": exported, "selection": selection_summary}
    save_json(report, output_dir / "review_cases.json")
    return report
