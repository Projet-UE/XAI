#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

from brain_tumor_xai.data import BrainTumorDataset, load_manifest
from brain_tumor_xai.model import build_resnet18_binary, load_checkpoint
from brain_tumor_xai.utils import ensure_dir, select_device
from brain_tumor_xai.xai import compute_attribution


def _disable_nnpack_if_needed(device: torch.device) -> None:
    # On some Grid'5000 CPU frontends, NNPACK initialization fails repeatedly and
    # floods stderr. Disabling it keeps benchmark logs readable and avoids warning overhead.
    if device.type == "cpu" and hasattr(torch.backends, "nnpack"):
        try:
            torch.backends.nnpack.enabled = False  # type: ignore[attr-defined]
        except Exception:
            # Fallback: keep default behavior if this backend flag is unavailable.
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Brain MRI XAI methods using a protocol-level faithfulness metric: "
            "predicted-class confidence drop after masking top-attribution pixels."
        )
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/brain_mri_xai_benchmark"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-samples-per-class", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=[0.1, 0.2, 0.3],
        help="Top-attribution area fractions to mask when computing confidence-drop curves.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["gradcam", "integrated_gradients", "occlusion"],
        choices=["gradcam", "integrated_gradients", "occlusion"],
    )
    return parser.parse_args()


def _safe_mean(values: Iterable[float]) -> Optional[float]:
    values = [float(value) for value in values]
    if not values:
        return None
    return float(mean(values))


def _safe_std(values: Iterable[float]) -> Optional[float]:
    values = [float(value) for value in values]
    if len(values) < 2:
        return None
    avg = float(mean(values))
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    return float(math.sqrt(variance))


def _bootstrap_ci(values: Sequence[float], *, iterations: int, seed: int, alpha: float = 0.05) -> Optional[Dict[str, float]]:
    if not values:
        return None
    observed_mean = float(mean(values))
    if len(values) == 1 or iterations <= 1:
        return {
            "mean": observed_mean,
            "low": observed_mean,
            "high": observed_mean,
            "sample_count": float(len(values)),
        }

    rng = random.Random(seed)
    sample_count = len(values)
    bootstrap_means: List[float] = []
    for _ in range(iterations):
        total = 0.0
        for _ in range(sample_count):
            total += values[rng.randrange(sample_count)]
        bootstrap_means.append(total / sample_count)

    bootstrap_means.sort()
    low_index = int((alpha / 2.0) * len(bootstrap_means))
    high_index = int((1.0 - alpha / 2.0) * len(bootstrap_means))
    high_index = min(max(high_index, 0), len(bootstrap_means) - 1)
    return {
        "mean": observed_mean,
        "low": float(bootstrap_means[low_index]),
        "high": float(bootstrap_means[high_index]),
        "sample_count": float(sample_count),
    }


def _top_fraction_mask(heatmap: np.ndarray, fraction: float) -> np.ndarray:
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    flat = heatmap.reshape(-1)
    top_count = max(1, int(round(flat.size * fraction)))
    indices = np.argpartition(flat, -top_count)[-top_count:]
    mask = np.zeros_like(flat, dtype=bool)
    mask[indices] = True
    return mask.reshape(heatmap.shape)


def _apply_mask(image: torch.Tensor, mask_hw: np.ndarray, baseline: float = 0.0) -> torch.Tensor:
    if image.ndim != 3:
        raise ValueError(f"Expected image tensor shape (C, H, W), got {tuple(image.shape)}")
    if mask_hw.shape != tuple(image.shape[-2:]):
        raise ValueError(f"Mask shape {mask_hw.shape} does not match image spatial shape {tuple(image.shape[-2:])}")
    masked = image.clone()
    mask_tensor = torch.from_numpy(mask_hw).to(dtype=torch.bool)
    masked[:, mask_tensor] = float(baseline)
    return masked


def _tumor_probability(model: torch.nn.Module, image: torch.Tensor, device: torch.device) -> float:
    with torch.no_grad():
        logits = model(image.unsqueeze(0).to(device)).squeeze(1)
        return float(torch.sigmoid(logits)[0].detach().cpu().item())


def _predicted_class_confidence(tumor_probability: float) -> tuple[int, float]:
    predicted_label = 1 if tumor_probability >= 0.5 else 0
    confidence = tumor_probability if predicted_label == 1 else (1.0 - tumor_probability)
    return predicted_label, float(confidence)


def _sample_indices_per_class(records: Sequence[Dict[str, Any]], max_samples_per_class: int, seed: int) -> List[int]:
    class_to_indices: Dict[str, List[int]] = {}
    for index, record in enumerate(records):
        class_name = str(record["class_name"])
        class_to_indices.setdefault(class_name, []).append(index)

    rng = random.Random(seed)
    selected: List[int] = []
    for class_name, indices in sorted(class_to_indices.items()):
        take = min(max_samples_per_class, len(indices))
        selected.extend(rng.sample(indices, take))
    return sorted(selected)


def main() -> None:
    args = parse_args()
    if any(not (0.0 < value <= 1.0) for value in args.fractions):
        raise ValueError("All `--fractions` values must be in (0, 1].")

    output_dir = ensure_dir(args.output_dir)
    device = select_device(args.device)
    _disable_nnpack_if_needed(device)
    manifest = load_manifest(args.manifest_path)

    dataset = BrainTumorDataset(args.data_root, manifest, split="test", image_size=args.image_size, augment=False)
    selected_indices = _sample_indices_per_class(dataset.records, args.max_samples_per_class, args.seed)
    if not selected_indices:
        raise ValueError("No test samples found for the selected split.")

    model = build_resnet18_binary(pretrained=False)
    load_checkpoint(model, str(args.checkpoint_path), map_location=device)
    model.to(device)
    model.eval()

    case_rows: List[Dict[str, Any]] = []
    method_rows: Dict[str, List[Dict[str, Any]]] = {method: [] for method in args.methods}

    for index in selected_indices:
        sample = dataset[index]
        image = sample["image"].detach().cpu()
        label = int(sample["label"].item())
        class_name = str(sample["class_name"])
        path = str(sample["path"])

        base_probability = _tumor_probability(model, image, device)
        predicted_label, base_confidence = _predicted_class_confidence(base_probability)

        case_entry: Dict[str, Any] = {
            "index": int(index),
            "path": path,
            "class_name": class_name,
            "label": label,
            "base_tumor_probability": base_probability,
            "predicted_label": predicted_label,
            "base_predicted_class_confidence": base_confidence,
            "methods": {},
        }

        for method in args.methods:
            heatmap = compute_attribution(model, image.to(device), method)
            drops: Dict[str, float] = {}
            for fraction in args.fractions:
                mask = _top_fraction_mask(heatmap, fraction)
                masked_image = _apply_mask(image, mask, baseline=0.0)
                masked_probability = _tumor_probability(model, masked_image, device)
                masked_confidence = (
                    masked_probability if predicted_label == 1 else (1.0 - masked_probability)
                )
                drop = base_confidence - masked_confidence
                drops[f"{fraction:.2f}"] = float(drop)

            mean_drop = _safe_mean(drops.values()) or 0.0
            method_case_row = {
                "index": int(index),
                "path": path,
                "class_name": class_name,
                "label": label,
                "predicted_label": predicted_label,
                "base_tumor_probability": base_probability,
                "base_predicted_class_confidence": base_confidence,
                "confidence_drop_by_fraction": drops,
                "mean_confidence_drop": float(mean_drop),
            }
            method_rows[method].append(method_case_row)
            case_entry["methods"][method] = method_case_row

        case_rows.append(case_entry)

    method_summary: Dict[str, Any] = {}
    for method, rows in method_rows.items():
        mean_drop_values = [float(row["mean_confidence_drop"]) for row in rows]
        by_fraction: Dict[str, List[float]] = {f"{fraction:.2f}": [] for fraction in args.fractions}
        for row in rows:
            for fraction_key, drop in row["confidence_drop_by_fraction"].items():
                by_fraction[fraction_key].append(float(drop))

        class_names = sorted({str(row["class_name"]) for row in rows})
        by_class: Dict[str, Any] = {}
        for class_name in class_names:
            class_rows = [row for row in rows if str(row["class_name"]) == class_name]
            class_drops = [float(row["mean_confidence_drop"]) for row in class_rows]
            by_class[class_name] = {
                "case_count": len(class_rows),
                "mean_confidence_drop": _safe_mean(class_drops),
                "std_confidence_drop": _safe_std(class_drops),
                "ci95_mean_confidence_drop": _bootstrap_ci(
                    class_drops,
                    iterations=args.bootstrap_iterations,
                    seed=args.seed + sum(ord(char) for char in f"{method}:{class_name}"),
                ),
            }

        method_summary[method] = {
            "case_count": len(rows),
            "mean_confidence_drop": _safe_mean(mean_drop_values),
            "std_confidence_drop": _safe_std(mean_drop_values),
            "ci95_mean_confidence_drop": _bootstrap_ci(
                mean_drop_values,
                iterations=args.bootstrap_iterations,
                seed=args.seed + sum(ord(char) for char in method),
            ),
            "mean_confidence_drop_by_fraction": {
                fraction: _safe_mean(values) for fraction, values in by_fraction.items()
            },
            "by_class": by_class,
        }

    ranking = [
        {
            "method": method,
            "mean_confidence_drop": summary.get("mean_confidence_drop"),
            "ci95_mean_confidence_drop": summary.get("ci95_mean_confidence_drop"),
        }
        for method, summary in method_summary.items()
    ]
    ranking.sort(
        key=lambda entry: entry["mean_confidence_drop"] if entry["mean_confidence_drop"] is not None else -1.0,
        reverse=True,
    )
    for rank, entry in enumerate(ranking, start=1):
        entry["rank"] = rank

    benchmark = {
        "title": "Brain MRI XAI method benchmark",
        "protocol": {
            "description": (
                "For each sample, compute predicted-class confidence drop after masking top-attribution pixels. "
                "Higher confidence drop means stronger faithfulness of the attribution map."
            ),
            "fractions": [float(value) for value in args.fractions],
            "max_samples_per_class": int(args.max_samples_per_class),
            "bootstrap_iterations": int(args.bootstrap_iterations),
            "seed": int(args.seed),
        },
        "dataset": {
            "data_root": str(args.data_root),
            "manifest_path": str(args.manifest_path),
            "selected_case_count": len(selected_indices),
            "selected_indices": selected_indices,
        },
        "model": {
            "checkpoint_path": str(args.checkpoint_path),
            "device": str(device),
        },
        "methods": method_summary,
        "ranking": ranking,
        "cases": case_rows,
    }

    (output_dir / "xai_method_benchmark.json").write_text(json.dumps(benchmark, indent=2), encoding="utf-8")

    lines = [
        "# Brain MRI XAI method benchmark",
        "",
        "This file compares XAI methods with a protocol-level faithfulness metric.",
        "",
        "Protocol:",
        f"- Selected test samples: `{len(selected_indices)}`",
        f"- Max samples per class: `{args.max_samples_per_class}`",
        f"- Fractions masked: `{', '.join(f'{value:.2f}' for value in args.fractions)}`",
        f"- Bootstrap iterations: `{args.bootstrap_iterations}`",
        "",
        "Interpretation rule:",
        "- `mean_confidence_drop` higher is better for this metric (attribution identifies pixels that matter more for model confidence).",
        "",
        "## Ranking",
        "",
    ]
    if ranking:
        for entry in ranking:
            score = entry["mean_confidence_drop"]
            ci = entry.get("ci95_mean_confidence_drop") or {}
            if ci:
                lines.append(
                    f"- Rank {entry['rank']}: `{entry['method']}` -> `{score:.4f}` "
                    f"(95% CI `{ci['low']:.4f}` to `{ci['high']:.4f}`)"
                )
            else:
                lines.append(f"- Rank {entry['rank']}: `{entry['method']}` -> `{score:.4f}`")
    else:
        lines.append("- No ranking could be computed.")

    lines.extend(
        [
            "",
            "## Notes for project interpretation",
            "",
            "- This metric compares methods on model-faithfulness, not on medical truth by itself.",
            "- Keep using lesion-level metrics and confusion matrix as the primary model-performance evidence.",
            "- Use this benchmark to justify why one XAI method is preferred for discussion in the report.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
