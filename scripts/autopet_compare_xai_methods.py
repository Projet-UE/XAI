#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


FAILURE_METRICS_BY_CATEGORY: Dict[str, List[str]] = {
    "positive_detected": ["mass_ratio_inside_gt", "top10_ratio_inside_gt"],
    "positive_missed": ["mass_ratio_inside_gt", "top10_ratio_inside_gt"],
    "false_positive": ["mass_ratio_inside_prediction"],
    "true_negative": ["mass_ratio_inside_prediction"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare autoPET XAI methods on a common protocol and export pairwise "
            "bootstrap deltas with confidence intervals."
        )
    )
    parser.add_argument("--review-cases-path", type=Path, required=True)
    parser.add_argument("--metrics-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--state-name", type=str, default="autopet_state")
    return parser.parse_args()


def _safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return float(mean(filtered))


def _bootstrap_ci_mean(values: Sequence[float], *, iterations: int, seed: int, alpha: float = 0.05) -> Optional[Dict[str, float]]:
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


def _bootstrap_ci_delta(
    values_a: Sequence[float],
    values_b: Sequence[float],
    *,
    iterations: int,
    seed: int,
    alpha: float = 0.05,
) -> Optional[Dict[str, float]]:
    if not values_a or not values_b:
        return None
    if len(values_a) != len(values_b):
        raise ValueError("Paired bootstrap requires equal-length vectors.")

    paired = [float(a) - float(b) for a, b in zip(values_a, values_b)]
    return _bootstrap_ci_mean(paired, iterations=iterations, seed=seed, alpha=alpha)


def _case_category(ground_truth_positive: bool, prediction_positive: bool, dice: float) -> str:
    if ground_truth_positive and prediction_positive:
        return "positive_detected" if dice > 0 else "positive_missed"
    if ground_truth_positive and not prediction_positive:
        return "positive_missed"
    if not ground_truth_positive and prediction_positive:
        return "false_positive"
    return "true_negative"


def _collect_method_case_rows(review_cases: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    metric_by_case = {case["case_id"]: case for case in metrics["cases"]}
    method_rows: Dict[str, List[Dict[str, Any]]] = {}

    for case in review_cases["cases"]:
        case_metrics = metric_by_case.get(case["case_id"])
        if case_metrics is None:
            continue

        ground_truth_positive = bool(case.get("ground_truth_positive", case_metrics.get("gt_volume_ml", 0.0) > 0))
        prediction_positive = bool(case.get("prediction_positive", case_metrics.get("pred_volume_ml", 0.0) > 0))
        category = _case_category(ground_truth_positive, prediction_positive, float(case_metrics["dice"]))

        for method_entry in case.get("methods", []):
            method_name = method_entry.get("method")
            if not method_name:
                continue
            summary = method_entry.get("attribution_summary", {})
            row = {
                "case_id": case["case_id"],
                "category": category,
                "ground_truth_positive": ground_truth_positive,
                "prediction_positive": prediction_positive,
                "dice": float(case_metrics["dice"]),
                "mass_ratio_inside_gt": summary.get("mass_ratio_inside_gt"),
                "top10_ratio_inside_gt": summary.get("top10_ratio_inside_gt"),
                "mass_ratio_inside_prediction": summary.get("mass_ratio_inside_prediction"),
            }
            method_rows.setdefault(method_name, []).append(row)

    for method_name, rows in method_rows.items():
        rows.sort(key=lambda row: row["case_id"])
        method_rows[method_name] = rows

    return method_rows


def _method_summary(rows: Sequence[Dict[str, Any]], *, iterations: int, seed: int) -> Dict[str, Any]:
    positive_rows = [row for row in rows if row["ground_truth_positive"]]
    false_positive_rows = [row for row in rows if row["category"] == "false_positive"]

    localization = [row["mass_ratio_inside_gt"] for row in positive_rows if row["mass_ratio_inside_gt"] is not None]
    top10 = [row["top10_ratio_inside_gt"] for row in positive_rows if row["top10_ratio_inside_gt"] is not None]
    fp_focus = [
        row["mass_ratio_inside_prediction"]
        for row in false_positive_rows
        if row["mass_ratio_inside_prediction"] is not None
    ]

    localization_mean = _safe_mean(localization)
    fp_focus_mean = _safe_mean(fp_focus)
    score = None
    if localization_mean is not None:
        score = float(0.75 * localization_mean + 0.25 * (fp_focus_mean if fp_focus_mean is not None else 0.0))

    return {
        "case_count": len(rows),
        "positive_case_count": len(positive_rows),
        "false_positive_case_count": len(false_positive_rows),
        "mean_mass_ratio_inside_gt": localization_mean,
        "mean_top10_ratio_inside_gt": _safe_mean(top10),
        "mean_mass_ratio_inside_prediction_on_false_positive": fp_focus_mean,
        "ci95_mass_ratio_inside_gt": _bootstrap_ci_mean(localization, iterations=iterations, seed=seed),
        "ci95_top10_ratio_inside_gt": _bootstrap_ci_mean(top10, iterations=iterations, seed=seed + 1),
        "ci95_false_positive_prediction_focus": _bootstrap_ci_mean(fp_focus, iterations=iterations, seed=seed + 2),
        "protocol_score": score,
    }


def _paired_vectors(
    rows_a: Sequence[Dict[str, Any]],
    rows_b: Sequence[Dict[str, Any]],
    metric_name: str,
    *,
    filter_key: Optional[str] = None,
    filter_value: Optional[Any] = None,
) -> Tuple[List[float], List[float], List[str]]:
    map_a = {row["case_id"]: row for row in rows_a}
    map_b = {row["case_id"]: row for row in rows_b}
    shared_case_ids = sorted(set(map_a) & set(map_b))
    values_a: List[float] = []
    values_b: List[float] = []
    used_case_ids: List[str] = []
    for case_id in shared_case_ids:
        row_a = map_a[case_id]
        row_b = map_b[case_id]
        if filter_key is not None and row_a.get(filter_key) != filter_value:
            continue
        value_a = row_a.get(metric_name)
        value_b = row_b.get(metric_name)
        if value_a is None or value_b is None:
            continue
        values_a.append(float(value_a))
        values_b.append(float(value_b))
        used_case_ids.append(case_id)
    return values_a, values_b, used_case_ids


def _case_level_score(row: Dict[str, Any]) -> Optional[float]:
    category = row.get("category")
    if category in {"positive_detected", "positive_missed"}:
        inside_gt = row.get("mass_ratio_inside_gt")
        top10_inside_gt = row.get("top10_ratio_inside_gt")
        if inside_gt is None and top10_inside_gt is None:
            return None
        if inside_gt is None:
            return float(top10_inside_gt)
        if top10_inside_gt is None:
            return float(inside_gt)
        return float(0.7 * float(inside_gt) + 0.3 * float(top10_inside_gt))
    if category in {"false_positive", "true_negative"}:
        focus = row.get("mass_ratio_inside_prediction")
        if focus is None:
            return None
        return float(focus)
    return None


def _build_failure_taxonomy_and_agreement(
    method_rows: Dict[str, List[Dict[str, Any]]],
    method_names: Sequence[str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    row_lookup: Dict[str, Dict[str, Dict[str, Any]]] = {
        method_name: {row["case_id"]: row for row in rows} for method_name, rows in method_rows.items()
    }

    case_catalog: Dict[str, Dict[str, Any]] = {}
    for rows in method_rows.values():
        for row in rows:
            case_catalog[row["case_id"]] = {
                "case_id": row["case_id"],
                "category": row["category"],
                "ground_truth_positive": bool(row["ground_truth_positive"]),
                "prediction_positive": bool(row["prediction_positive"]),
                "dice": float(row["dice"]),
            }

    taxonomy_by_category: Dict[str, Any] = {}
    agreement_by_category: Dict[str, Any] = {}
    taxonomy_cases: List[Dict[str, Any]] = []

    for category in ["positive_detected", "positive_missed", "false_positive", "true_negative"]:
        category_case_ids = sorted(
            case_id for case_id, entry in case_catalog.items() if str(entry.get("category")) == category
        )
        method_behavior: Dict[str, Any] = {}
        case_winner_share: Dict[str, float] = {method_name: 0.0 for method_name in method_names}
        case_level_rows: List[Dict[str, Any]] = []
        comparable_case_count = 0
        unique_winner_count = 0

        for method_name in method_names:
            rows = [
                row_lookup[method_name][case_id]
                for case_id in category_case_ids
                if case_id in row_lookup[method_name]
            ]
            scores = [_case_level_score(row) for row in rows]
            method_behavior[method_name] = {
                "case_count": len(rows),
                "mean_case_score": _safe_mean(scores),
                "mean_mass_ratio_inside_gt": _safe_mean(row.get("mass_ratio_inside_gt") for row in rows),
                "mean_top10_ratio_inside_gt": _safe_mean(row.get("top10_ratio_inside_gt") for row in rows),
                "mean_mass_ratio_inside_prediction": _safe_mean(
                    row.get("mass_ratio_inside_prediction") for row in rows
                ),
            }

        for case_id in category_case_ids:
            score_by_method: Dict[str, float] = {}
            metric_by_method: Dict[str, Dict[str, Optional[float]]] = {}
            for method_name in method_names:
                row = row_lookup.get(method_name, {}).get(case_id)
                if row is None:
                    continue
                metric_by_method[method_name] = {
                    "mass_ratio_inside_gt": row.get("mass_ratio_inside_gt"),
                    "top10_ratio_inside_gt": row.get("top10_ratio_inside_gt"),
                    "mass_ratio_inside_prediction": row.get("mass_ratio_inside_prediction"),
                }
                score = _case_level_score(row)
                if score is not None:
                    score_by_method[method_name] = float(score)

            winners: List[str] = []
            if score_by_method:
                comparable_case_count += 1
                max_score = max(score_by_method.values())
                winners = sorted(
                    method_name for method_name, score in score_by_method.items() if score == max_score
                )
                if len(winners) == 1:
                    unique_winner_count += 1
                tie_weight = 1.0 / float(len(winners))
                for method_name in winners:
                    case_winner_share[method_name] += tie_weight

            case_level_rows.append(
                {
                    "case_id": case_id,
                    "category": category,
                    "score_by_method": score_by_method,
                    "winners": winners,
                    "metrics_by_method": metric_by_method,
                }
            )

        pairwise_dominance: List[Dict[str, Any]] = []
        for method_a, method_b in itertools.combinations(method_names, 2):
            a_better = 0
            b_better = 0
            ties = 0
            comparable = 0
            for case_row in case_level_rows:
                scores = case_row.get("score_by_method", {})
                score_a = scores.get(method_a)
                score_b = scores.get(method_b)
                if score_a is None or score_b is None:
                    continue
                comparable += 1
                if score_a > score_b:
                    a_better += 1
                elif score_b > score_a:
                    b_better += 1
                else:
                    ties += 1
            pairwise_dominance.append(
                {
                    "method_a": method_a,
                    "method_b": method_b,
                    "comparable_case_count": comparable,
                    "a_better_count": a_better,
                    "b_better_count": b_better,
                    "tie_count": ties,
                }
            )

        taxonomy_by_category[category] = {
            "case_count": len(category_case_ids),
            "case_ids": category_case_ids,
            "method_behavior": method_behavior,
        }
        agreement_by_category[category] = {
            "comparable_case_count": comparable_case_count,
            "unique_winner_case_count": unique_winner_count,
            "unique_winner_rate": (
                float(unique_winner_count / comparable_case_count) if comparable_case_count > 0 else None
            ),
            "winner_share": {
                method_name: (
                    float(case_winner_share[method_name] / comparable_case_count)
                    if comparable_case_count > 0
                    else None
                )
                for method_name in method_names
            },
            "pairwise_score_dominance": pairwise_dominance,
            "case_rows": case_level_rows,
        }
        taxonomy_cases.extend(case_level_rows)

    failure_taxonomy = {
        "description": (
            "Case-level taxonomy on review set: positive_detected, positive_missed, "
            "false_positive, true_negative, with method-specific attribution behavior."
        ),
        "metrics_by_category": FAILURE_METRICS_BY_CATEGORY,
        "by_category": taxonomy_by_category,
        "cases": sorted(taxonomy_cases, key=lambda row: (row["category"], row["case_id"])),
    }
    agreement_report = {
        "description": (
            "Compact cross-method agreement report by failure category. "
            "Winner share uses case-level score ties split evenly."
        ),
        "score_definition": (
            "positive categories: 0.7 * mass_ratio_inside_gt + 0.3 * top10_ratio_inside_gt; "
            "negative/false-positive categories: mass_ratio_inside_prediction"
        ),
        "by_category": agreement_by_category,
    }
    return failure_taxonomy, agreement_report


def main() -> None:
    args = parse_args()
    review_cases = json.loads(args.review_cases_path.read_text(encoding="utf-8"))
    metrics = json.loads(args.metrics_path.read_text(encoding="utf-8"))
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    method_rows = _collect_method_case_rows(review_cases, metrics)
    method_names = sorted(method_rows.keys())
    if not method_names:
        raise ValueError("No method entries found in review cases.")

    method_summaries: Dict[str, Any] = {}
    for method_name in method_names:
        seed = args.bootstrap_seed + sum((index + 1) * ord(char) for index, char in enumerate(method_name))
        method_summaries[method_name] = _method_summary(
            method_rows[method_name],
            iterations=args.bootstrap_iterations,
            seed=seed,
        )

    ranking = [
        {
            "method": method_name,
            "protocol_score": method_summaries[method_name].get("protocol_score"),
            "mean_mass_ratio_inside_gt": method_summaries[method_name].get("mean_mass_ratio_inside_gt"),
            "mean_top10_ratio_inside_gt": method_summaries[method_name].get("mean_top10_ratio_inside_gt"),
            "mean_mass_ratio_inside_prediction_on_false_positive": method_summaries[method_name].get(
                "mean_mass_ratio_inside_prediction_on_false_positive"
            ),
        }
        for method_name in method_names
    ]
    ranking.sort(
        key=lambda entry: (
            entry["protocol_score"] if entry["protocol_score"] is not None else -1.0,
            entry["mean_mass_ratio_inside_gt"] if entry["mean_mass_ratio_inside_gt"] is not None else -1.0,
        ),
        reverse=True,
    )
    for index, item in enumerate(ranking, start=1):
        item["rank"] = index

    metric_specs = [
        (
            "mass_ratio_inside_gt",
            "positive_only",
            "ground_truth_positive",
            True,
        ),
        (
            "top10_ratio_inside_gt",
            "positive_only",
            "ground_truth_positive",
            True,
        ),
        (
            "mass_ratio_inside_prediction",
            "false_positive_only",
            "category",
            "false_positive",
        ),
    ]
    paired_deltas: Dict[str, List[Dict[str, Any]]] = {
        f"{metric_name}__{scope}": [] for metric_name, scope, _, _ in metric_specs
    }

    for method_a, method_b in itertools.combinations(method_names, 2):
        rows_a = method_rows[method_a]
        rows_b = method_rows[method_b]
        for metric_name, scope, filter_key, filter_value in metric_specs:
            values_a, values_b, case_ids = _paired_vectors(
                rows_a,
                rows_b,
                metric_name,
                filter_key=filter_key,
                filter_value=filter_value,
            )
            if not values_a:
                continue
            delta_ci = _bootstrap_ci_delta(
                values_a,
                values_b,
                iterations=args.bootstrap_iterations,
                seed=args.bootstrap_seed
                + sum(ord(char) for char in metric_name)
                + 1000 * len(case_ids)
                + sum(ord(char) for char in method_a + method_b),
            )
            paired_deltas[f"{metric_name}__{scope}"].append(
                {
                    "method_a": method_a,
                    "method_b": method_b,
                    "paired_case_count": len(case_ids),
                    "paired_case_ids": case_ids,
                    "mean_a": float(mean(values_a)),
                    "mean_b": float(mean(values_b)),
                    "delta_a_minus_b": float(mean(values_a) - mean(values_b)),
                    "ci95_delta_a_minus_b": delta_ci,
                }
            )

    failure_taxonomy, agreement_report = _build_failure_taxonomy_and_agreement(method_rows, method_names)

    benchmark = {
        "state_name": args.state_name,
        "bootstrap_iterations": int(args.bootstrap_iterations),
        "bootstrap_seed": int(args.bootstrap_seed),
        "protocol": {
            "score_formula": (
                "0.75 * mean_mass_ratio_inside_gt + "
                "0.25 * mean_mass_ratio_inside_prediction_on_false_positive"
            ),
            "paired_delta_metrics": [
                "mass_ratio_inside_gt__positive_only",
                "top10_ratio_inside_gt__positive_only",
                "mass_ratio_inside_prediction__false_positive_only",
            ],
        },
        "method_summaries": method_summaries,
        "ranking": ranking,
        "paired_delta_ci": paired_deltas,
        "failure_taxonomy": failure_taxonomy,
        "cross_method_agreement": agreement_report,
    }
    (output_dir / "method_benchmark.json").write_text(json.dumps(benchmark, indent=2), encoding="utf-8")

    lines = [
        "# autoPET XAI method benchmark",
        "",
        f"State analyzed: `{args.state_name}`",
        "",
        "## Method ranking",
        "",
    ]
    for item in ranking:
        score = item["protocol_score"]
        score_text = "n/a" if score is None else f"{float(score):.4f}"
        lines.append(f"- Rank {item['rank']}: `{item['method']}` (protocol score `{score_text}`)")

    lines.extend(
        [
            "",
            "## Paired bootstrap delta confidence intervals",
            "",
            "Interpretation: positive `delta_a_minus_b` means method A is higher than method B for the given metric.",
            "",
        ]
    )
    for metric_key, rows in paired_deltas.items():
        lines.append(f"### `{metric_key}`")
        if not rows:
            lines.append("- No valid paired comparisons available.")
            lines.append("")
            continue
        for row in rows:
            ci = row.get("ci95_delta_a_minus_b")
            if ci:
                ci_text = f"[{ci['low']:.4f}, {ci['high']:.4f}]"
            else:
                ci_text = "n/a"
            lines.append(
                f"- `{row['method_a']}` - `{row['method_b']}`: "
                f"delta `{row['delta_a_minus_b']:.4f}` with 95% CI {ci_text} "
                f"on `{row['paired_case_count']}` paired cases."
            )
        lines.append("")

    lines.extend(
        [
            "## Failure taxonomy and cross-method agreement",
            "",
            "This section summarizes method behavior per case group and winner agreement rates.",
            "",
        ]
    )
    for category, summary in agreement_report["by_category"].items():
        comparable = summary.get("comparable_case_count", 0)
        unique_rate = summary.get("unique_winner_rate")
        unique_text = "n/a" if unique_rate is None else f"{float(unique_rate):.4f}"
        lines.append(
            f"- `{category}`: comparable cases `{comparable}`, unique-winner rate `{unique_text}`."
        )
        winner_share = summary.get("winner_share", {})
        winner_bits = []
        for method_name in method_names:
            share = winner_share.get(method_name)
            if share is None:
                winner_bits.append(f"{method_name}=n/a")
            else:
                winner_bits.append(f"{method_name}={float(share):.4f}")
        lines.append(f"  winner share: {', '.join(winner_bits)}")

    (output_dir / "method_benchmark.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
