from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_export(args: list[str], repo_root: Path) -> subprocess.CompletedProcess[str]:
    script_path = repo_root / "scripts" / "autopet_export_results.py"
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = str(repo_root / "src")
    return subprocess.run(
        [sys.executable, str(script_path), *args],
        text=True,
        capture_output=True,
        env=env,
    )


def test_export_fails_when_required_analysis_summary_is_missing(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    metrics_path = tmp_path / "segmentation_metrics.json"
    _write_json(
        metrics_path,
        {
            "mean_dice": 0.5,
            "mean_false_negative_volume_ml": 1.0,
            "mean_false_positive_volume_ml": 2.0,
            "cases": [],
        },
    )
    review_cases_path = tmp_path / "review_cases.json"
    _write_json(review_cases_path, {"cases": []})

    result = _run_export(
        [
            "--run-id",
            "tmp_export_missing_summary",
            "--results-root",
            str(tmp_path / "results"),
            "--metrics-path",
            str(metrics_path),
            "--review-cases-path",
            str(review_cases_path),
            "--require-analysis-summary",
            "--analysis-summary-path",
            str(tmp_path / "missing_summary.json"),
        ],
        repo_root,
    )
    assert result.returncode != 0
    assert "Missing required xai_analysis_summary.json" in result.stderr


def test_export_fails_when_protocol_benchmark_section_is_missing(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    metrics_path = tmp_path / "segmentation_metrics.json"
    _write_json(
        metrics_path,
        {
            "mean_dice": 0.5,
            "mean_false_negative_volume_ml": 1.0,
            "mean_false_positive_volume_ml": 2.0,
            "cases": [],
        },
    )
    review_cases_path = tmp_path / "review_cases.json"
    _write_json(review_cases_path, {"cases": []})
    summary_path = tmp_path / "summary.json"
    _write_json(summary_path, {"preferred_method": "integrated_gradients"})

    result = _run_export(
        [
            "--run-id",
            "tmp_export_missing_protocol",
            "--results-root",
            str(tmp_path / "results"),
            "--metrics-path",
            str(metrics_path),
            "--review-cases-path",
            str(review_cases_path),
            "--analysis-summary-path",
            str(summary_path),
            "--require-protocol-benchmark",
        ],
        repo_root,
    )
    assert result.returncode != 0
    assert "Protocol benchmark required" in result.stderr


def test_export_accepts_standalone_method_benchmark_for_protocol_requirement(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    metrics_path = tmp_path / "segmentation_metrics.json"
    _write_json(
        metrics_path,
        {
            "mean_dice": 0.5,
            "mean_false_negative_volume_ml": 1.0,
            "mean_false_positive_volume_ml": 2.0,
            "cases": [],
        },
    )
    review_cases_path = tmp_path / "review_cases.json"
    _write_json(review_cases_path, {"cases": []})
    benchmark_path = tmp_path / "method_benchmark.json"
    _write_json(benchmark_path, {"ranking": [{"method": "integrated_gradients", "rank": 1}]})

    result = _run_export(
        [
            "--run-id",
            "tmp_export_with_method_benchmark",
            "--results-root",
            str(tmp_path / "results"),
            "--metrics-path",
            str(metrics_path),
            "--review-cases-path",
            str(review_cases_path),
            "--method-benchmark-path",
            str(benchmark_path),
            "--require-protocol-benchmark",
        ],
        repo_root,
    )
    assert result.returncode == 0
    exported_benchmark = tmp_path / "results" / "tmp_export_with_method_benchmark" / "method_benchmark.json"
    assert exported_benchmark.exists()


def test_export_copies_all_method_panels_for_selected_cases_and_rewrites_paths(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    metrics_path = tmp_path / "segmentation_metrics.json"
    _write_json(
        metrics_path,
        {
            "mean_dice": 0.5,
            "mean_false_negative_volume_ml": 1.0,
            "mean_false_positive_volume_ml": 2.0,
            "cases": [],
        },
    )
    xai_dir = tmp_path / "xai"
    case_dir = xai_dir / "PETCT_case"
    case_dir.mkdir(parents=True, exist_ok=True)
    for name in ["integrated_gradients.png", "occlusion.png", "saliency.png"]:
        (case_dir / name).write_bytes(b"png")

    review_cases_path = tmp_path / "review_cases.json"
    _write_json(
        review_cases_path,
        {
            "cases": [
                {
                    "case_id": "PETCT_case",
                    "methods": [
                        {
                            "method": "integrated_gradients",
                            "figure": str(case_dir / "integrated_gradients.png"),
                        },
                        {
                            "method": "occlusion",
                            "figure": str(case_dir / "occlusion.png"),
                        },
                        {
                            "method": "saliency",
                            "figure": str(case_dir / "saliency.png"),
                        },
                    ],
                }
            ]
        },
    )

    result = _run_export(
        [
            "--run-id",
            "tmp_export_rewrite_paths",
            "--results-root",
            str(tmp_path / "results"),
            "--metrics-path",
            str(metrics_path),
            "--review-cases-path",
            str(review_cases_path),
            "--xai-dir",
            str(xai_dir),
            "--require-review-cases",
            "--require-xai-dir",
            "--max-figures",
            "1",
        ],
        repo_root,
    )
    assert result.returncode == 0

    exported_dir = tmp_path / "results" / "tmp_export_rewrite_paths"
    for name in ["integrated_gradients.png", "occlusion.png", "saliency.png"]:
        assert (exported_dir / "figures" / "PETCT_case" / name).exists()

    exported_review_cases = json.loads((exported_dir / "review_cases.json").read_text(encoding="utf-8"))
    exported_methods = {
        entry["method"]: entry["figure"] for entry in exported_review_cases["cases"][0]["methods"]
    }
    assert exported_methods["integrated_gradients"] == "figures/PETCT_case/integrated_gradients.png"
    assert exported_methods["occlusion"] == "figures/PETCT_case/occlusion.png"
    assert exported_methods["saliency"] == "figures/PETCT_case/saliency.png"
