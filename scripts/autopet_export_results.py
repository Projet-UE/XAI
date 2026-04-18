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
    parser.add_argument("--metrics-path", type=Path, default=None)
    parser.add_argument("--review-cases-path", type=Path, default=None)
    parser.add_argument("--xai-dir", type=Path, default=None)
    parser.add_argument("--run-config-path", type=Path, default=None)
    parser.add_argument("--analysis-summary-path", type=Path, default=None)
    parser.add_argument("--method-benchmark-path", type=Path, default=None)
    parser.add_argument("--require-review-cases", action="store_true")
    parser.add_argument("--require-xai-dir", action="store_true")
    parser.add_argument("--require-analysis-summary", action="store_true")
    parser.add_argument("--require-method-benchmark", action="store_true")
    parser.add_argument(
        "--require-protocol-benchmark",
        action="store_true",
        help=(
            "Fail if neither analysis summary with method benchmark nor standalone "
            "method benchmark file is available."
        ),
    )
    parser.add_argument("--snapshot-title", type=str, default="autoPET FDG run snapshot")
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


def _normalize_metrics_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    candidate = payload.get("metrics")
    if isinstance(candidate, dict) and "mean_dice" in candidate:
        return candidate
    return payload


def main() -> None:
    args = parse_args()
    split_root = args.artifacts_dir / args.split_name if (args.artifacts_dir / args.split_name).exists() else args.artifacts_dir
    metrics_path = args.metrics_path or (split_root / "review_metrics" / "segmentation_metrics.json")
    review_cases_path = args.review_cases_path or (split_root / "xai" / "review_cases.json")
    run_config_path = split_root / "training_run_config.json"
    predict_run_config_path = split_root / "review_metrics" / "predict_run_config.json"
    xai_run_config_path = split_root / "xai" / "xai_run_config.json"
    default_analysis_summary_path = split_root / "xai_analysis" / "xai_analysis_summary.json"
    default_method_benchmark_path = split_root / "xai_analysis" / "method_benchmark.json"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing segmentation metrics at {metrics_path}")
    if args.require_review_cases and not review_cases_path.exists():
        raise FileNotFoundError(f"Missing required review_cases.json at {review_cases_path}")

    target_dir = args.results_root / args.run_id
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir = ensure_dir(target_dir)
    metrics_payload = load_json(metrics_path)
    metrics = _normalize_metrics_payload(metrics_payload)
    if args.run_config_path is not None:
        run_config = load_json(args.run_config_path)
    elif run_config_path.exists():
        run_config = load_json(run_config_path)
    else:
        run_config = {}
        if predict_run_config_path.exists():
            run_config.update(load_json(predict_run_config_path))
        if xai_run_config_path.exists():
            run_config.update(load_json(xai_run_config_path))
    review_cases = load_json(review_cases_path) if review_cases_path.exists() else {"cases": []}
    analysis_summary_path = args.analysis_summary_path or default_analysis_summary_path
    method_benchmark_path = args.method_benchmark_path or default_method_benchmark_path
    if args.require_analysis_summary and not analysis_summary_path.exists():
        raise FileNotFoundError(f"Missing required xai_analysis_summary.json at {analysis_summary_path}")
    if args.require_method_benchmark and not method_benchmark_path.exists():
        raise FileNotFoundError(f"Missing required method_benchmark.json at {method_benchmark_path}")
    analysis_summary = load_json(analysis_summary_path) if analysis_summary_path.exists() else None
    method_benchmark = load_json(method_benchmark_path) if method_benchmark_path.exists() else None
    if args.require_protocol_benchmark:
        has_summary_ranking = bool(
            analysis_summary and "ranking" in analysis_summary.get("method_benchmark", {})
        )
        has_benchmark_ranking = bool(method_benchmark and "ranking" in method_benchmark)
        if not has_summary_ranking and not has_benchmark_ranking:
            raise ValueError(
                "Protocol benchmark required but ranking data is missing from both "
                "analysis summary and standalone method benchmark."
            )

    save_json(metrics, target_dir / "segmentation_metrics.json")
    save_json(run_config, target_dir / "run_config.json")
    save_json(review_cases, target_dir / "review_cases.json")
    if analysis_summary is not None:
        save_json(analysis_summary, target_dir / "xai_analysis_summary.json")
    if method_benchmark is not None:
        save_json(method_benchmark, target_dir / "method_benchmark.json")

    copied_figures = []
    xai_dir = args.xai_dir or (split_root / "xai")
    if args.require_xai_dir and not xai_dir.exists():
        raise FileNotFoundError(f"Missing required XAI directory at {xai_dir}")
    if xai_dir.exists():
        copied_figures = _copy_selected_figures(xai_dir, target_dir / "figures", args.max_figures)

    top_method = None
    top_score = None
    ranking_from_summary = []
    if analysis_summary is not None:
        ranking_from_summary = analysis_summary.get("method_benchmark", {}).get("ranking", [])
    ranking_from_file = method_benchmark.get("ranking", []) if method_benchmark else []
    ranking = ranking_from_file if ranking_from_file else ranking_from_summary
    if ranking:
        top_method = ranking[0].get("method")
        top_score = ranking[0].get("composite_protocol_score") or ranking[0].get("protocol_score")

    top_method_line = ""
    if top_method is not None and top_score is not None:
        top_method_line = f"- Top XAI method by protocol score: `{top_method}` (`{float(top_score):.4f}`)\n"
    elif top_method is not None:
        top_method_line = f"- Top XAI method by protocol score: `{top_method}`\n"

    readme = f"""# {args.snapshot_title}

This folder tracks a lightweight snapshot of an autoPET FDG nnUNet run.

- Split: `{args.split_name}`
- Mean Dice: `{metrics.get('mean_dice', 0.0):.4f}`
- Mean false negative volume (mL): `{metrics.get('mean_false_negative_volume_ml', 0.0):.4f}`
- Mean false positive volume (mL): `{metrics.get('mean_false_positive_volume_ml', 0.0):.4f}`
- Includes XAI analysis summary: `{"yes" if analysis_summary is not None else "no"}`
- Includes standalone XAI method benchmark: `{"yes" if method_benchmark is not None else "no"}`
{top_method_line}- Copied figures: `{len(copied_figures)}`
"""
    (target_dir / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
