#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(payload: Dict[str, object], path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two tracked autoPET FDG run snapshots and export a lightweight summary.")
    parser.add_argument("--baseline-dir", type=Path, required=True, help="Tracked results directory for the reference run.")
    parser.add_argument("--candidate-dir", type=Path, required=True, help="Tracked results directory for the stronger/newer run.")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--run-id", type=str, required=True, help="Comparison folder name created under results/.")
    parser.add_argument("--baseline-label", type=str, default="baseline")
    parser.add_argument("--candidate-label", type=str, default="candidate")
    return parser.parse_args()


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return mean(values) if values else 0.0


def _case_table(metrics: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    return {str(case["case_id"]): case for case in metrics.get("cases", [])}


def _summarize_cases(cases: List[Dict[str, object]]) -> Dict[str, float]:
    positive_cases = [case for case in cases if float(case.get("gt_volume_ml", 0.0)) > 0.0]
    negative_cases = [case for case in cases if float(case.get("gt_volume_ml", 0.0)) == 0.0]
    return {
        "case_count": float(len(cases)),
        "positive_case_count": float(len(positive_cases)),
        "negative_case_count": float(len(negative_cases)),
        "positive_mean_dice": _mean(float(case.get("dice", 0.0)) for case in positive_cases),
        "negative_mean_dice": _mean(float(case.get("dice", 0.0)) for case in negative_cases),
        "positive_mean_false_positive_volume_ml": _mean(float(case.get("false_positive_volume_ml", 0.0)) for case in positive_cases),
        "negative_mean_false_positive_volume_ml": _mean(float(case.get("false_positive_volume_ml", 0.0)) for case in negative_cases),
        "positive_mean_false_negative_volume_ml": _mean(float(case.get("false_negative_volume_ml", 0.0)) for case in positive_cases),
        "negative_mean_false_negative_volume_ml": _mean(float(case.get("false_negative_volume_ml", 0.0)) for case in negative_cases),
    }


def _collect_case_deltas(
    baseline_cases: Dict[str, Dict[str, object]],
    candidate_cases: Dict[str, Dict[str, object]],
) -> List[Dict[str, object]]:
    deltas: List[Dict[str, object]] = []
    shared_case_ids = sorted(set(baseline_cases) & set(candidate_cases))
    for case_id in shared_case_ids:
        baseline = baseline_cases[case_id]
        candidate = candidate_cases[case_id]
        deltas.append(
            {
                "case_id": case_id,
                "gt_volume_ml": float(candidate.get("gt_volume_ml", 0.0)),
                "baseline_dice": float(baseline.get("dice", 0.0)),
                "candidate_dice": float(candidate.get("dice", 0.0)),
                "dice_delta": float(candidate.get("dice", 0.0)) - float(baseline.get("dice", 0.0)),
                "baseline_false_negative_volume_ml": float(baseline.get("false_negative_volume_ml", 0.0)),
                "candidate_false_negative_volume_ml": float(candidate.get("false_negative_volume_ml", 0.0)),
                "false_negative_volume_delta_ml": float(candidate.get("false_negative_volume_ml", 0.0))
                - float(baseline.get("false_negative_volume_ml", 0.0)),
                "baseline_false_positive_volume_ml": float(baseline.get("false_positive_volume_ml", 0.0)),
                "candidate_false_positive_volume_ml": float(candidate.get("false_positive_volume_ml", 0.0)),
                "false_positive_volume_delta_ml": float(candidate.get("false_positive_volume_ml", 0.0))
                - float(baseline.get("false_positive_volume_ml", 0.0)),
            }
        )
    return deltas


def main() -> None:
    args = parse_args()
    baseline_metrics = load_json(args.baseline_dir / "segmentation_metrics.json")
    candidate_metrics = load_json(args.candidate_dir / "segmentation_metrics.json")

    baseline_cases = _case_table(baseline_metrics)
    candidate_cases = _case_table(candidate_metrics)
    case_deltas = _collect_case_deltas(baseline_cases, candidate_cases)

    baseline_summary = {
        "mean_dice": float(baseline_metrics.get("mean_dice", 0.0)),
        "mean_false_negative_volume_ml": float(baseline_metrics.get("mean_false_negative_volume_ml", 0.0)),
        "mean_false_positive_volume_ml": float(baseline_metrics.get("mean_false_positive_volume_ml", 0.0)),
        **_summarize_cases(list(baseline_cases.values())),
    }
    candidate_summary = {
        "mean_dice": float(candidate_metrics.get("mean_dice", 0.0)),
        "mean_false_negative_volume_ml": float(candidate_metrics.get("mean_false_negative_volume_ml", 0.0)),
        "mean_false_positive_volume_ml": float(candidate_metrics.get("mean_false_positive_volume_ml", 0.0)),
        **_summarize_cases(list(candidate_cases.values())),
    }

    delta_summary = {
        key: float(candidate_summary.get(key, 0.0)) - float(baseline_summary.get(key, 0.0))
        for key in [
            "mean_dice",
            "mean_false_negative_volume_ml",
            "mean_false_positive_volume_ml",
            "positive_mean_dice",
            "negative_mean_dice",
            "positive_mean_false_positive_volume_ml",
            "negative_mean_false_positive_volume_ml",
            "positive_mean_false_negative_volume_ml",
            "negative_mean_false_negative_volume_ml",
        ]
    }

    comparison = {
        "baseline": {
            "label": args.baseline_label,
            "path": str(args.baseline_dir),
            **baseline_summary,
        },
        "candidate": {
            "label": args.candidate_label,
            "path": str(args.candidate_dir),
            **candidate_summary,
        },
        "delta_candidate_minus_baseline": delta_summary,
        "shared_case_count": len(case_deltas),
        "per_case_deltas": case_deltas,
    }

    target_dir = ensure_dir(args.results_root / args.run_id)
    save_json(comparison, target_dir / "comparison.json")

    readme = f"""# autoPET FDG comparison snapshot

This folder compares two tracked autoPET FDG nnUNet runs on the same review split.

- Baseline: `{args.baseline_label}` -> `{args.baseline_dir.name}`
- Candidate: `{args.candidate_label}` -> `{args.candidate_dir.name}`
- Shared review cases: `{len(case_deltas)}`

## Aggregate comparison

- Mean Dice: `{baseline_summary['mean_dice']:.4f} -> {candidate_summary['mean_dice']:.4f}` (`{delta_summary['mean_dice']:+.4f}`)
- Mean false negative volume (mL): `{baseline_summary['mean_false_negative_volume_ml']:.4f} -> {candidate_summary['mean_false_negative_volume_ml']:.4f}` (`{delta_summary['mean_false_negative_volume_ml']:+.4f}`)
- Mean false positive volume (mL): `{baseline_summary['mean_false_positive_volume_ml']:.4f} -> {candidate_summary['mean_false_positive_volume_ml']:.4f}` (`{delta_summary['mean_false_positive_volume_ml']:+.4f}`)
- Positive-case mean Dice: `{baseline_summary['positive_mean_dice']:.4f} -> {candidate_summary['positive_mean_dice']:.4f}` (`{delta_summary['positive_mean_dice']:+.4f}`)
- Negative-case mean false positive volume (mL): `{baseline_summary['negative_mean_false_positive_volume_ml']:.4f} -> {candidate_summary['negative_mean_false_positive_volume_ml']:.4f}` (`{delta_summary['negative_mean_false_positive_volume_ml']:+.4f}`)

See `comparison.json` for the per-case deltas.
"""
    (target_dir / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
