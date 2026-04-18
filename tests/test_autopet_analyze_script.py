from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script_module(script_name: str):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


autopet_analyze = _load_script_module("autopet_analyze_xai.py")


def test_bootstrap_ci_is_well_formed() -> None:
    ci = autopet_analyze._bootstrap_ci([0.2, 0.3, 0.4, 0.5], iterations=500, seed=42)
    assert ci is not None
    assert ci["low"] <= ci["mean"] <= ci["high"]
    assert ci["sample_count"] == 4.0


def test_method_benchmark_prefers_better_localization() -> None:
    strong_cases = [
        {
            "ground_truth_positive": True,
            "category": "positive_detected",
            "attribution_summary": {
                "mass_ratio_inside_gt": 0.80,
                "top10_ratio_inside_gt": 0.85,
                "mean_attr_inside_gt": 0.80,
                "mean_attr_outside_gt": 0.20,
                "mass_ratio_inside_prediction": 0.70,
            },
        },
        {
            "ground_truth_positive": True,
            "category": "positive_detected",
            "attribution_summary": {
                "mass_ratio_inside_gt": 0.75,
                "top10_ratio_inside_gt": 0.80,
                "mean_attr_inside_gt": 0.75,
                "mean_attr_outside_gt": 0.25,
                "mass_ratio_inside_prediction": 0.65,
            },
        },
        {
            "ground_truth_positive": False,
            "category": "false_positive",
            "attribution_summary": {
                "mass_ratio_inside_gt": None,
                "top10_ratio_inside_gt": None,
                "mean_attr_inside_gt": None,
                "mean_attr_outside_gt": None,
                "mass_ratio_inside_prediction": 0.60,
            },
        },
    ]
    weak_cases = [
        {
            "ground_truth_positive": True,
            "category": "positive_detected",
            "attribution_summary": {
                "mass_ratio_inside_gt": 0.25,
                "top10_ratio_inside_gt": 0.30,
                "mean_attr_inside_gt": 0.30,
                "mean_attr_outside_gt": 0.20,
                "mass_ratio_inside_prediction": 0.20,
            },
        },
        {
            "ground_truth_positive": True,
            "category": "positive_detected",
            "attribution_summary": {
                "mass_ratio_inside_gt": 0.20,
                "top10_ratio_inside_gt": 0.25,
                "mean_attr_inside_gt": 0.25,
                "mean_attr_outside_gt": 0.20,
                "mass_ratio_inside_prediction": 0.20,
            },
        },
        {
            "ground_truth_positive": False,
            "category": "false_positive",
            "attribution_summary": {
                "mass_ratio_inside_gt": None,
                "top10_ratio_inside_gt": None,
                "mean_attr_inside_gt": None,
                "mean_attr_outside_gt": None,
                "mass_ratio_inside_prediction": 0.15,
            },
        },
    ]

    strong = autopet_analyze._build_method_benchmark(
        method_name="integrated_gradients",
        method_cases=strong_cases,
        bootstrap_iterations=300,
        bootstrap_seed=123,
    )
    weak = autopet_analyze._build_method_benchmark(
        method_name="occlusion",
        method_cases=weak_cases,
        bootstrap_iterations=300,
        bootstrap_seed=123,
    )

    ranking = autopet_analyze._rank_method_benchmarks(
        {
            "integrated_gradients": strong,
            "occlusion": weak,
        }
    )

    assert ranking[0]["method"] == "integrated_gradients"
    assert ranking[0]["composite_protocol_score"] > ranking[1]["composite_protocol_score"]
