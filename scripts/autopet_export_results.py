#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Dict, List

from brain_tumor_xai.utils import ensure_dir, load_json, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a lightweight autoPET FDG snapshot to the tracked results/ folder.")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/autopet_fdg_poc"))
    parser.add_argument("--run-id", type=str, required=True, help="Suffix used under results/, for example autopet_fdg_20260324")
    parser.add_argument("--split-name", type=str, default="fdg_dev", choices=["fdg_dev", "fdg_full"])
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--max-figures", type=int, default=4)
    return parser.parse_args()


def _copy_selected_figures(xai_dir: Path, target_dir: Path, max_figures: int) -> List[str]:
    copied: List[str] = []
    figure_paths: List[Path] = []

    case_dirs = [path for path in sorted(xai_dir.iterdir()) if path.is_dir()]
    preferred_names = ["integrated_gradients.png", "occlusion.png", "saliency.png"]
    for case_dir in case_dirs:
        for preferred_name in preferred_names:
            candidate = case_dir / preferred_name
            if candidate.exists():
                figure_paths.append(candidate)
                break
        if len(figure_paths) >= max_figures:
            break

    if len(figure_paths) < max_figures:
        remaining_candidates = [path for path in sorted(xai_dir.rglob("*.png")) if path not in figure_paths]
        figure_paths.extend(remaining_candidates[: max_figures - len(figure_paths)])

    for figure_path in figure_paths:
        relative = figure_path.relative_to(xai_dir)
        destination = target_dir / relative
        ensure_dir(destination.parent)
        shutil.copy2(figure_path, destination)
        copied.append(str(relative))
    return copied


def main() -> None:
    args = parse_args()
    split_root = args.artifacts_dir / args.split_name if (args.artifacts_dir / args.split_name).exists() else args.artifacts_dir
    metrics_path = split_root / "review_metrics" / "segmentation_metrics.json"
    review_cases_path = split_root / "xai" / "review_cases.json"
    run_config_path = split_root / "training_run_config.json"
    predict_run_config_path = split_root / "review_metrics" / "predict_run_config.json"
    xai_run_config_path = split_root / "xai" / "xai_run_config.json"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing segmentation metrics at {metrics_path}")

    target_dir = args.results_root / args.run_id
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir = ensure_dir(target_dir)
    metrics = load_json(metrics_path)
    if run_config_path.exists():
        run_config = load_json(run_config_path)
    else:
        run_config = {}
        if predict_run_config_path.exists():
            run_config.update(load_json(predict_run_config_path))
        if xai_run_config_path.exists():
            run_config.update(load_json(xai_run_config_path))
    review_cases = load_json(review_cases_path) if review_cases_path.exists() else {"cases": []}

    save_json(metrics, target_dir / "segmentation_metrics.json")
    save_json(run_config, target_dir / "run_config.json")
    save_json(review_cases, target_dir / "review_cases.json")

    copied_figures = []
    xai_dir = split_root / "xai"
    if xai_dir.exists():
        copied_figures = _copy_selected_figures(xai_dir, target_dir / "figures", args.max_figures)

    readme = f"""# autoPET FDG run snapshot

This folder tracks a lightweight snapshot of an autoPET FDG nnUNet run.

- Split: `{args.split_name}`
- Mean Dice: `{metrics.get('mean_dice', 0.0):.4f}`
- Mean false negative volume (mL): `{metrics.get('mean_false_negative_volume_ml', 0.0):.4f}`
- Mean false positive volume (mL): `{metrics.get('mean_false_positive_volume_ml', 0.0):.4f}`
- Copied figures: `{len(copied_figures)}`
"""
    (target_dir / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
