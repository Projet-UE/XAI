from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_autopet_xai_analyze_compare_export_pipeline(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    review_cases_path = tmp_path / "review_cases.json"
    metrics_path = tmp_path / "segmentation_metrics.json"
    analyze_out = tmp_path / "analyze_out"
    compare_out = tmp_path / "compare_out"
    export_root = tmp_path / "export_root"
    xai_dir = tmp_path / "xai"
    (xai_dir / "case_a").mkdir(parents=True, exist_ok=True)
    (xai_dir / "case_a" / "integrated_gradients.png").write_bytes(b"png")

    review_cases = {
        "cases": [
            {
                "case_id": "case_a",
                "nnunet_case_id": "FDG_0001",
                "ground_truth_positive": True,
                "prediction_positive": True,
                "methods": [
                    {
                        "method": "integrated_gradients",
                        "figure": str(xai_dir / "case_a" / "integrated_gradients.png"),
                        "attribution_summary": {
                            "mass_ratio_inside_gt": 0.8,
                            "top10_ratio_inside_gt": 0.9,
                            "mass_ratio_inside_prediction": 0.7,
                            "mean_attr_inside_gt": 0.8,
                            "mean_attr_outside_gt": 0.2,
                            "mean_attr_inside_prediction": 0.7,
                            "mean_attr_outside_prediction": 0.2,
                        },
                    },
                    {
                        "method": "occlusion",
                        "figure": str(xai_dir / "case_a" / "integrated_gradients.png"),
                        "attribution_summary": {
                            "mass_ratio_inside_gt": 0.4,
                            "top10_ratio_inside_gt": 0.5,
                            "mass_ratio_inside_prediction": 0.3,
                            "mean_attr_inside_gt": 0.4,
                            "mean_attr_outside_gt": 0.2,
                            "mean_attr_inside_prediction": 0.3,
                            "mean_attr_outside_prediction": 0.2,
                        },
                    },
                ],
            },
            {
                "case_id": "case_b",
                "nnunet_case_id": "FDG_0002",
                "ground_truth_positive": False,
                "prediction_positive": True,
                "methods": [
                    {
                        "method": "integrated_gradients",
                        "figure": str(xai_dir / "case_a" / "integrated_gradients.png"),
                        "attribution_summary": {
                            "mass_ratio_inside_gt": None,
                            "top10_ratio_inside_gt": None,
                            "mass_ratio_inside_prediction": 0.6,
                            "mean_attr_inside_gt": None,
                            "mean_attr_outside_gt": None,
                            "mean_attr_inside_prediction": 0.6,
                            "mean_attr_outside_prediction": 0.2,
                        },
                    },
                    {
                        "method": "occlusion",
                        "figure": str(xai_dir / "case_a" / "integrated_gradients.png"),
                        "attribution_summary": {
                            "mass_ratio_inside_gt": None,
                            "top10_ratio_inside_gt": None,
                            "mass_ratio_inside_prediction": 0.2,
                            "mean_attr_inside_gt": None,
                            "mean_attr_outside_gt": None,
                            "mean_attr_inside_prediction": 0.2,
                            "mean_attr_outside_prediction": 0.2,
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
    _write_json(review_cases_path, review_cases)
    _write_json(metrics_path, metrics)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "autopet_analyze_xai.py"),
            "--review-cases-path",
            str(review_cases_path),
            "--metrics-path",
            str(metrics_path),
            "--output-dir",
            str(analyze_out),
            "--state-name",
            "integration_state",
            "--bootstrap-iterations",
            "200",
        ],
        check=True,
        env=env,
    )

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "autopet_compare_xai_methods.py"),
            "--review-cases-path",
            str(review_cases_path),
            "--metrics-path",
            str(metrics_path),
            "--output-dir",
            str(compare_out),
            "--state-name",
            "integration_state",
            "--bootstrap-iterations",
            "200",
        ],
        check=True,
        env=env,
    )

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "autopet_export_results.py"),
            "--run-id",
            "integration_snapshot",
            "--results-root",
            str(export_root),
            "--metrics-path",
            str(metrics_path),
            "--review-cases-path",
            str(review_cases_path),
            "--xai-dir",
            str(xai_dir),
            "--analysis-summary-path",
            str(analyze_out / "xai_analysis_summary.json"),
            "--require-review-cases",
            "--require-xai-dir",
            "--require-analysis-summary",
            "--require-protocol-benchmark",
        ],
        check=True,
        env=env,
    )

    assert (analyze_out / "xai_analysis_summary.json").exists()
    assert (compare_out / "method_benchmark.json").exists()
    assert (compare_out / "method_benchmark.md").exists()
    assert (export_root / "integration_snapshot" / "xai_analysis_summary.json").exists()
