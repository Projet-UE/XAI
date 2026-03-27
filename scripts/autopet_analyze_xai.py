#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize qualitative autoPET XAI exports into paper-ready JSON/Markdown.")
    parser.add_argument("--review-cases-path", type=Path, required=True)
    parser.add_argument("--metrics-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--state-name", type=str, required=True)
    parser.add_argument("--title", type=str, default="autoPET FDG XAI analysis")
    return parser.parse_args()


def _safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return float(mean(filtered))


def _safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return float(numerator / denominator)


def _case_category(ground_truth_positive: bool, prediction_positive: bool, dice: float) -> str:
    if ground_truth_positive and prediction_positive:
        return "positive_detected" if dice > 0 else "positive_missed"
    if ground_truth_positive and not prediction_positive:
        return "positive_missed"
    if not ground_truth_positive and prediction_positive:
        return "false_positive"
    return "true_negative"


def main() -> None:
    args = parse_args()
    review_cases = json.loads(args.review_cases_path.read_text(encoding="utf-8"))
    metrics = json.loads(args.metrics_path.read_text(encoding="utf-8"))
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_by_case = {case["case_id"]: case for case in metrics["cases"]}
    enriched_cases: List[Dict[str, Any]] = []

    for case in review_cases["cases"]:
        case_metrics = metric_by_case.get(case["case_id"])
        if case_metrics is None:
            continue
        ground_truth_positive = bool(case.get("ground_truth_positive", case_metrics.get("gt_volume_ml", 0.0) > 0))
        prediction_positive = bool(case.get("prediction_positive", case_metrics.get("pred_volume_ml", 0.0) > 0))
        enriched_methods: List[Dict[str, Any]] = []
        for method in case["methods"]:
            summary = method.get("attribution_summary", {})
            enriched_methods.append({"method": method["method"], "figure": method["figure"], "attribution_summary": summary})
        enriched_cases.append(
            {
                "case_id": case["case_id"],
                "nnunet_case_id": case["nnunet_case_id"],
                "ground_truth_positive": ground_truth_positive,
                "prediction_positive": prediction_positive,
                "dice": float(case_metrics["dice"]),
                "false_negative_volume_ml": float(case_metrics["false_negative_volume_ml"]),
                "false_positive_volume_ml": float(case_metrics["false_positive_volume_ml"]),
                "gt_volume_ml": float(case_metrics["gt_volume_ml"]),
                "pred_volume_ml": float(case_metrics["pred_volume_ml"]),
                "category": _case_category(ground_truth_positive, prediction_positive, float(case_metrics["dice"])),
                "methods": enriched_methods,
            }
        )

    method_names = sorted({method["method"] for case in enriched_cases for method in case["methods"]})
    method_summaries: Dict[str, Any] = {}

    for method_name in method_names:
        method_cases = []
        positive_cases = []
        negative_cases = []
        false_positive_cases = []
        missed_positive_cases = []
        for case in enriched_cases:
            method = next((entry for entry in case["methods"] if entry["method"] == method_name), None)
            if method is None:
                continue
            joined = {**case, "attribution_summary": method["attribution_summary"], "figure": method["figure"]}
            method_cases.append(joined)
            if case["ground_truth_positive"]:
                positive_cases.append(joined)
            else:
                negative_cases.append(joined)
            if case["category"] == "false_positive":
                false_positive_cases.append(joined)
            if case["category"] == "positive_missed":
                missed_positive_cases.append(joined)

        method_summaries[method_name] = {
            "case_count": len(method_cases),
            "all_cases": {
                "mean_prediction_focus_ratio": _safe_mean(
                    case["attribution_summary"].get("mass_ratio_inside_prediction") for case in method_cases
                ),
                "mean_union_focus_ratio": _safe_mean(
                    case["attribution_summary"].get("mass_ratio_inside_union") for case in method_cases
                ),
            },
            "positive_cases": {
                "count": len(positive_cases),
                "mean_dice": _safe_mean(case["dice"] for case in positive_cases),
                "mean_gt_focus_ratio": _safe_mean(
                    case["attribution_summary"].get("mass_ratio_inside_gt") for case in positive_cases
                ),
                "mean_prediction_focus_ratio": _safe_mean(
                    case["attribution_summary"].get("mass_ratio_inside_prediction") for case in positive_cases
                ),
                "mean_gt_focus_enrichment": _safe_ratio(
                    _safe_mean(case["attribution_summary"].get("mean_attr_inside_gt") for case in positive_cases),
                    _safe_mean(case["attribution_summary"].get("mean_attr_outside_gt") for case in positive_cases),
                ),
            },
            "negative_cases": {
                "count": len(negative_cases),
                "mean_false_positive_volume_ml": _safe_mean(case["false_positive_volume_ml"] for case in negative_cases),
                "mean_prediction_focus_ratio": _safe_mean(
                    case["attribution_summary"].get("mass_ratio_inside_prediction") for case in negative_cases
                ),
                "mean_prediction_focus_enrichment": _safe_ratio(
                    _safe_mean(case["attribution_summary"].get("mean_attr_inside_prediction") for case in negative_cases),
                    _safe_mean(case["attribution_summary"].get("mean_attr_outside_prediction") for case in negative_cases),
                ),
            },
            "false_positive_cases": {
                "count": len(false_positive_cases),
                "mean_prediction_focus_ratio": _safe_mean(
                    case["attribution_summary"].get("mass_ratio_inside_prediction") for case in false_positive_cases
                ),
                "mean_prediction_focus_enrichment": _safe_ratio(
                    _safe_mean(case["attribution_summary"].get("mean_attr_inside_prediction") for case in false_positive_cases),
                    _safe_mean(case["attribution_summary"].get("mean_attr_outside_prediction") for case in false_positive_cases),
                ),
            },
            "positive_missed_cases": {
                "count": len(missed_positive_cases),
                "mean_gt_focus_ratio": _safe_mean(
                    case["attribution_summary"].get("mass_ratio_inside_gt") for case in missed_positive_cases
                ),
                "mean_gt_focus_enrichment": _safe_ratio(
                    _safe_mean(case["attribution_summary"].get("mean_attr_inside_gt") for case in missed_positive_cases),
                    _safe_mean(case["attribution_summary"].get("mean_attr_outside_gt") for case in missed_positive_cases),
                ),
            },
        }

    preferred_method = "integrated_gradients" if "integrated_gradients" in method_summaries else (method_names[0] if method_names else None)
    preferred = method_summaries.get(preferred_method, {})

    summary = {
        "state_name": args.state_name,
        "title": args.title,
        "selection": review_cases.get("selection", {}),
        "metric_summary": {
            "case_count": int(metrics["case_count"]),
            "mean_dice": float(metrics["mean_dice"]),
            "mean_false_negative_volume_ml": float(metrics["mean_false_negative_volume_ml"]),
            "mean_false_positive_volume_ml": float(metrics["mean_false_positive_volume_ml"]),
        },
        "case_category_counts": {
            category: sum(1 for case in enriched_cases if case["category"] == category)
            for category in ["positive_detected", "positive_missed", "false_positive", "true_negative"]
        },
        "preferred_method": preferred_method,
        "method_summaries": method_summaries,
        "cases": enriched_cases,
    }
    (output_dir / "xai_analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    observations: List[str] = []
    if preferred_method and preferred:
        positive_gt_focus = preferred.get("positive_cases", {}).get("mean_gt_focus_ratio")
        positive_gt_enrichment = preferred.get("positive_cases", {}).get("mean_gt_focus_enrichment")
        negative_pred_focus = preferred.get("negative_cases", {}).get("mean_prediction_focus_ratio")
        negative_pred_enrichment = preferred.get("negative_cases", {}).get("mean_prediction_focus_enrichment")
        fp_focus = preferred.get("false_positive_cases", {}).get("mean_prediction_focus_ratio")
        fp_enrichment = preferred.get("false_positive_cases", {}).get("mean_prediction_focus_enrichment")
        missed_gt_focus = preferred.get("positive_missed_cases", {}).get("mean_gt_focus_ratio")
        missed_gt_enrichment = preferred.get("positive_missed_cases", {}).get("mean_gt_focus_enrichment")

        if positive_gt_focus is not None:
            observations.append(
                f"On positive cases, `{preferred_method}` concentrates on average {positive_gt_focus:.3f} of attribution mass inside the ground-truth lesion."
            )
        if positive_gt_enrichment is not None:
            observations.append(
                f"On positive cases, attribution intensity is on average {positive_gt_enrichment:.2f}x higher inside the ground-truth lesion than outside it."
            )
        if negative_pred_focus is not None:
            observations.append(
                f"On negative cases, `{preferred_method}` still allocates on average {negative_pred_focus:.3f} of attribution mass inside predicted foreground, which helps explain residual false positives."
            )
        if negative_pred_enrichment is not None:
            observations.append(
                f"On negative cases, attribution intensity inside predicted foreground is {negative_pred_enrichment:.2f}x the outside level on average."
            )
        if fp_focus is not None:
            observations.append(
                f"On explicitly false-positive cases, the same method allocates on average {fp_focus:.3f} of attribution mass inside predicted foreground rather than empty background."
            )
        if fp_enrichment is not None:
            observations.append(
                f"On false-positive cases, attribution intensity inside predicted foreground is {fp_enrichment:.2f}x the outside level on average."
            )
        if missed_gt_focus is not None:
            observations.append(
                f"When lesions are missed or only partially detected, attribution inside the ground-truth region drops to {missed_gt_focus:.3f} on average."
            )
        if missed_gt_enrichment is not None:
            observations.append(
                f"For missed-positive cases, attribution intensity inside the ground-truth lesion falls to {missed_gt_enrichment:.2f}x the outside level on average."
            )

    lines = [
        f"# {args.title}",
        "",
        f"State analyzed: `{args.state_name}`",
        "",
        "Reminder: XAI highlights which regions influenced the model, not which regions are automatically cancerous.",
        "",
        "## Quantitative context",
        "",
        f"- Review case count: `{metrics['case_count']}`",
        f"- Mean Dice: `{metrics['mean_dice']:.4f}`",
        f"- Mean false negative volume (mL): `{metrics['mean_false_negative_volume_ml']:.4f}`",
        f"- Mean false positive volume (mL): `{metrics['mean_false_positive_volume_ml']:.4f}`",
        "",
        "## Review selection",
        "",
        f"- Available review cases: `{review_cases.get('selection', {}).get('available_case_count', len(enriched_cases))}`",
        f"- Selected review cases: `{review_cases.get('selection', {}).get('selected_case_count', len(enriched_cases))}`",
        f"- Selected positives: `{review_cases.get('selection', {}).get('selected_positive_case_count', 'n/a')}`",
        f"- Selected negatives: `{review_cases.get('selection', {}).get('selected_negative_case_count', 'n/a')}`",
        "",
        "## Case categories",
        "",
        f"- Positive detected: `{summary['case_category_counts']['positive_detected']}`",
        f"- Positive missed: `{summary['case_category_counts']['positive_missed']}`",
        f"- False positive: `{summary['case_category_counts']['false_positive']}`",
        f"- True negative: `{summary['case_category_counts']['true_negative']}`",
        "",
        f"## Preferred method summary: `{preferred_method}`",
        "",
    ]
    if observations:
        lines.extend([f"- {line}" for line in observations])
    else:
        lines.append("- No attribution summary was available.")

    lines.extend(
        [
            "",
            "## Interpretation guidance",
            "",
            "- In successful positive cases, we expect attribution to overlap lesion-related PET uptake and predicted lesion regions.",
            "- In false positives, attribution can still focus on high-uptake but non-lesion regions, which helps explain over-segmentation.",
            "- In missed positive cases, attribution may stay diffuse or shift away from part of the true lesion, which is consistent with false negatives.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
