#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

from autopet_xai.metrics import evaluate_review_predictions, postprocess_prediction_dir
from brain_tumor_xai.utils import ensure_dir, load_json, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep lightweight post-processing settings on existing autoPET review predictions.")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/autopet_fdg_poc"))
    parser.add_argument("--split-name", type=str, default="fdg_full", choices=["fdg_dev", "fdg_full"])
    parser.add_argument("--predictions-dir", type=Path, default=None)
    parser.add_argument("--metrics-dir", type=Path, default=None)
    parser.add_argument("--sweep-id", type=str, required=True)
    parser.add_argument("--rank-by", nargs="+", default=["mean_pet"], choices=["mean_pet", "max_pet", "volume_ml"])
    parser.add_argument("--min-component-ml", nargs="+", type=float, default=[0.0, 5.0, 10.0, 20.0, 30.0, 50.0])
    parser.add_argument("--max-components", nargs="+", type=int, default=[0, 1, 2, 3])
    return parser.parse_args()


def _case_group_summary(cases: List[Dict[str, Any]]) -> Dict[str, float]:
    positives = [case for case in cases if float(case.get("gt_volume_ml", 0.0)) > 0.0]
    negatives = [case for case in cases if float(case.get("gt_volume_ml", 0.0)) == 0.0]
    return {
        "positive_case_count": float(len(positives)),
        "negative_case_count": float(len(negatives)),
        "positive_mean_dice": mean(float(case.get("dice", 0.0)) for case in positives) if positives else 0.0,
        "negative_mean_dice": mean(float(case.get("dice", 0.0)) for case in negatives) if negatives else 0.0,
        "positive_mean_false_negative_volume_ml": mean(float(case.get("false_negative_volume_ml", 0.0)) for case in positives)
        if positives
        else 0.0,
        "positive_mean_false_positive_volume_ml": mean(float(case.get("false_positive_volume_ml", 0.0)) for case in positives)
        if positives
        else 0.0,
        "negative_mean_false_positive_volume_ml": mean(float(case.get("false_positive_volume_ml", 0.0)) for case in negatives)
        if negatives
        else 0.0,
    }


def _config_slug(rank_by: str, min_component_ml: float, max_components: int) -> str:
    min_slug = str(min_component_ml).replace(".", "p")
    max_slug = "all" if max_components <= 0 else str(max_components)
    return f"rank-{rank_by}__minml-{min_slug}__max-{max_slug}"


def _write_summary(
    *,
    split_name: str,
    raw_predictions_dir: Path,
    sweep_id: str,
    results: List[Dict[str, Any]],
    sweep_root: Path,
) -> None:
    ranked = sorted(
        results,
        key=lambda item: (
            float(item["metrics"]["mean_dice"]),
            -float(item["metrics"]["mean_false_positive_volume_ml"]),
            -float(item["metrics"]["mean_false_negative_volume_ml"]),
        ),
        reverse=True,
    )

    summary = {
        "split_name": split_name,
        "raw_predictions_dir": str(raw_predictions_dir),
        "sweep_id": sweep_id,
        "evaluated_count": len(results),
        "ranking": [
            {
                "label": item["label"],
                "rank_by": item["rank_by"],
                "min_component_volume_ml": item["min_component_volume_ml"],
                "max_components": item["max_components"],
                "mean_dice": item["metrics"]["mean_dice"],
                "mean_false_negative_volume_ml": item["metrics"]["mean_false_negative_volume_ml"],
                "mean_false_positive_volume_ml": item["metrics"]["mean_false_positive_volume_ml"],
                "positive_mean_dice": item["positive_mean_dice"],
                "negative_mean_false_positive_volume_ml": item["negative_mean_false_positive_volume_ml"],
            }
            for item in ranked
        ],
    }
    save_json(summary, sweep_root / "summary.json")

    top = ranked[0]
    readme = f"""# autoPET post-processing sweep

This folder tracks a sweep of lightweight connected-component filtering on existing review predictions.

- Split: `{split_name}`
- Sweep id: `{sweep_id}`
- Evaluated settings so far: `{max(len(results) - 1, 0)}` post-processed variants + raw baseline

## Best configuration by mean Dice

- Label: `{top['label']}`
- Mean Dice: `{top['metrics']['mean_dice']:.4f}`
- Mean false negative volume (mL): `{top['metrics']['mean_false_negative_volume_ml']:.4f}`
- Mean false positive volume (mL): `{top['metrics']['mean_false_positive_volume_ml']:.4f}`
- Positive-case mean Dice: `{top['positive_mean_dice']:.4f}`
- Negative-case mean false positive volume (mL): `{top['negative_mean_false_positive_volume_ml']:.4f}`

See `summary.json` for the full ranking.
"""
    (sweep_root / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    args = parse_args()
    split_root = args.artifacts_dir / args.split_name if (args.artifacts_dir / args.split_name).exists() else args.artifacts_dir
    mapping = load_json(split_root / "manifests" / f"case_mapping_{args.split_name}.json")
    raw_predictions_dir = Path(args.predictions_dir or split_root / "review_predictions")
    if not raw_predictions_dir.exists():
        raise FileNotFoundError(f"Prediction directory does not exist: {raw_predictions_dir}")

    sweep_root = ensure_dir(split_root / "postprocess_sweeps" / args.sweep_id)
    metrics_root = ensure_dir(args.metrics_dir or sweep_root / "metrics")

    results: List[Dict[str, Any]] = []

    baseline_metrics = evaluate_review_predictions(mapping, raw_predictions_dir)
    baseline_entry = {
        "label": "raw_predictions",
        "rank_by": None,
        "min_component_volume_ml": 0.0,
        "max_components": None,
        "metrics": baseline_metrics,
        **_case_group_summary(baseline_metrics["cases"]),
    }
    results.append(baseline_entry)
    save_json(baseline_entry, metrics_root / "raw_predictions.json")
    _write_summary(
        split_name=args.split_name,
        raw_predictions_dir=raw_predictions_dir,
        sweep_id=args.sweep_id,
        results=results,
        sweep_root=sweep_root,
    )
    print("[sweep] raw baseline saved", flush=True)

    for rank_by in args.rank_by:
        for min_component_ml in args.min_component_ml:
            for max_components in args.max_components:
                slug = _config_slug(rank_by, min_component_ml, max_components)
                metric_path = metrics_root / f"{slug}.json"
                if metric_path.exists():
                    entry = load_json(metric_path)
                    results.append(entry)
                    print(f"[sweep] skip existing {slug}", flush=True)
                    _write_summary(
                        split_name=args.split_name,
                        raw_predictions_dir=raw_predictions_dir,
                        sweep_id=args.sweep_id,
                        results=results,
                        sweep_root=sweep_root,
                    )
                    continue
                processed_dir = ensure_dir(sweep_root / "predictions" / slug)
                print(f"[sweep] evaluating {slug}", flush=True)
                report = postprocess_prediction_dir(
                    case_mapping=mapping,
                    prediction_dir=raw_predictions_dir,
                    output_dir=processed_dir,
                    min_component_volume_ml=min_component_ml,
                    max_components=max_components if max_components > 0 else None,
                    rank_by=rank_by,
                )
                metrics = evaluate_review_predictions(mapping, processed_dir)
                entry = {
                    "label": slug,
                    "rank_by": rank_by,
                    "min_component_volume_ml": float(min_component_ml),
                    "max_components": int(max_components) if max_components > 0 else None,
                    "metrics": metrics,
                    "postprocess_report": report,
                    **_case_group_summary(metrics["cases"]),
                }
                results.append(entry)
                save_json(entry, metric_path)
                _write_summary(
                    split_name=args.split_name,
                    raw_predictions_dir=raw_predictions_dir,
                    sweep_id=args.sweep_id,
                    results=results,
                    sweep_root=sweep_root,
                )
                print(
                    "[sweep] done "
                    f"{slug} dice={metrics['mean_dice']:.4f} "
                    f"fp={metrics['mean_false_positive_volume_ml']:.4f} "
                    f"fn={metrics['mean_false_negative_volume_ml']:.4f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
