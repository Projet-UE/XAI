from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_autopet_compare_xai_methods_generates_outputs(tmp_path: Path) -> None:
    review_cases = {
        "cases": [
            {
                "case_id": "case_a",
                "ground_truth_positive": True,
                "prediction_positive": True,
                "methods": [
                    {
                        "method": "integrated_gradients",
                        "attribution_summary": {
                            "mass_ratio_inside_gt": 0.8,
                            "top10_ratio_inside_gt": 0.9,
                            "mass_ratio_inside_prediction": 0.7,
                        },
                    },
                    {
                        "method": "occlusion",
                        "attribution_summary": {
                            "mass_ratio_inside_gt": 0.4,
                            "top10_ratio_inside_gt": 0.5,
                            "mass_ratio_inside_prediction": 0.3,
                        },
                    },
                ],
            },
            {
                "case_id": "case_b",
                "ground_truth_positive": False,
                "prediction_positive": True,
                "methods": [
                    {
                        "method": "integrated_gradients",
                        "attribution_summary": {
                            "mass_ratio_inside_gt": None,
                            "top10_ratio_inside_gt": None,
                            "mass_ratio_inside_prediction": 0.6,
                        },
                    },
                    {
                        "method": "occlusion",
                        "attribution_summary": {
                            "mass_ratio_inside_gt": None,
                            "top10_ratio_inside_gt": None,
                            "mass_ratio_inside_prediction": 0.2,
                        },
                    },
                ],
            },
        ]
    }
    metrics = {
        "case_count": 2,
        "mean_dice": 0.4,
        "mean_false_negative_volume_ml": 1.0,
        "mean_false_positive_volume_ml": 2.0,
        "cases": [
            {
                "case_id": "case_a",
                "dice": 0.8,
                "false_negative_volume_ml": 0.5,
                "false_positive_volume_ml": 1.0,
                "gt_volume_ml": 10.0,
                "pred_volume_ml": 8.0,
            },
            {
                "case_id": "case_b",
                "dice": 0.0,
                "false_negative_volume_ml": 0.0,
                "false_positive_volume_ml": 4.0,
                "gt_volume_ml": 0.0,
                "pred_volume_ml": 3.0,
            },
        ],
    }

    review_path = tmp_path / "review_cases.json"
    metrics_path = tmp_path / "segmentation_metrics.json"
    output_dir = tmp_path / "out"
    _write_json(review_path, review_cases)
    _write_json(metrics_path, metrics)

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "autopet_compare_xai_methods.py"
    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--review-cases-path",
            str(review_path),
            "--metrics-path",
            str(metrics_path),
            "--output-dir",
            str(output_dir),
            "--bootstrap-iterations",
            "200",
            "--bootstrap-seed",
            "7",
        ],
        check=True,
    )

    benchmark_path = output_dir / "method_benchmark.json"
    markdown_path = output_dir / "method_benchmark.md"
    assert benchmark_path.exists()
    assert markdown_path.exists()

    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    assert benchmark["ranking"][0]["method"] == "integrated_gradients"
    assert "paired_delta_ci" in benchmark
    assert "failure_taxonomy" in benchmark
    assert "cross_method_agreement" in benchmark
    assert benchmark["failure_taxonomy"]["by_category"]["false_positive"]["case_count"] == 1
