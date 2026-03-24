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


def _foreground_score(output: Any) -> torch.Tensor:
    if isinstance(output, (list, tuple)):
        output = output[0]
    if output.ndim != 5:
        raise ValueError(f"Expected 5D segmentation logits, got shape {tuple(output.shape)}")
    if output.shape[1] == 1:
        return output[:, 0].sum()
    return output[:, 1].sum()


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


def _save_case_panel(
    pet_crop: np.ndarray,
    ct_crop: np.ndarray,
    label_crop: np.ndarray,
    pred_crop: np.ndarray,
    attribution_crop: np.ndarray,
    slice_indices: Sequence[int],
    output_path: Union[str, Path],
    title: str,
) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    rows = len(slice_indices)
    fig, axes = plt.subplots(rows, 4, figsize=(14, 3.5 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_index, slice_index in enumerate(slice_indices):
        pet_slice = pet_crop[slice_index]
        ct_slice = ct_crop[slice_index]
        gt_slice = label_crop[slice_index]
        pred_slice = pred_crop[slice_index]
        attr_slice = attribution_crop[slice_index]

        axes[row_index, 0].imshow(pet_slice, cmap="inferno")
        axes[row_index, 0].imshow(attr_slice, cmap="magma", alpha=0.45)
        axes[row_index, 0].set_title(f"PET + {title}\nSlice {slice_index}")
        axes[row_index, 1].imshow(ct_slice, cmap="gray")
        axes[row_index, 1].imshow(attr_slice, cmap="magma", alpha=0.45)
        axes[row_index, 1].set_title("CT + attribution")
        axes[row_index, 2].imshow(gt_slice, cmap="Greens")
        axes[row_index, 2].set_title("Ground truth")
        axes[row_index, 3].imshow(pred_slice, cmap="Blues")
        axes[row_index, 3].set_title("Prediction")

        for column in range(4):
            axes[row_index, column].axis("off")

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
) -> Dict[str, Any]:
    predictor = _load_predictor(training_output_dir, fold=fold, checkpoint_name=checkpoint_name, device=device)
    network = predictor.network
    network.eval()
    network.to(torch.device(device))

    output_dir = ensure_dir(output_dir)
    exported: List[Dict[str, Any]] = []
    review_case_ids = case_mapping["review_case_ids"][:max_cases]

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
        input_tensor_torch = torch.from_numpy(input_tensor).unsqueeze(0).to(torch.device(device), dtype=torch.float32)
        slice_indices = _choose_representative_slices(label_crop if np.count_nonzero(label_crop) else pred_crop, count=3)

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
            )
            method_reports.append({"method": method, "figure": str(figure_path)})

        exported.append(
            {
                "case_id": case_id,
                "nnunet_case_id": nnunet_case_id,
                "prediction_path": str(prediction_path),
                "slice_indices": [int(index) for index in slice_indices],
                "methods": method_reports,
            }
        )

    report = {"cases": exported}
    save_json(report, output_dir / "review_cases.json")
    return report
