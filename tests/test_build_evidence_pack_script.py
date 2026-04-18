from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_build_project_evidence_pack_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results_root = tmp_path / "results"
    autopet_main = results_root / "autopet_main"
    autopet_cmp = results_root / "autopet_cmp"
    brain = results_root / "brain_backup"
    autopet_xai = results_root / "autopet_xai"

    _write_json(
        autopet_main / "segmentation_metrics.json",
        {
            "mean_dice": 0.5,
            "mean_false_negative_volume_ml": 1.0,
            "mean_false_positive_volume_ml": 2.0,
        },
    )
    _write_json(autopet_cmp / "comparison.json", {"delta_candidate_minus_baseline": {"mean_dice": 0.1}})
    _write_json(brain / "metrics.json", {"accuracy": 0.9, "f1": 0.88, "roc_auc": 0.95})
    _write_json(autopet_xai / "xai_analysis_summary.json", {"preferred_method": "integrated_gradients"})
    _write_json(
        autopet_xai / "method_benchmark.json",
        {"ranking": [{"method": "integrated_gradients", "rank": 1}]},
    )
    _write_json(results_root / "index.json", {"schema_version": "1.0.0"})

    # Minimal fake figures.
    (autopet_main / "figures").mkdir(parents=True, exist_ok=True)
    (brain / "figures").mkdir(parents=True, exist_ok=True)
    (autopet_main / "figures" / "a.png").write_bytes(b"png")
    (brain / "figures" / "b.png").write_bytes(b"png")

    script_path = repo_root / "scripts" / "build_project_evidence_pack.py"
    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--results-root",
            str(results_root),
            "--run-index-path",
            str(results_root / "index.json"),
            "--autopet-main-run-id",
            "autopet_main",
            "--autopet-comparison-run-id",
            "autopet_cmp",
            "--brain-mri-run-id",
            "brain_backup",
            "--autopet-xai-analysis-run-id",
            "autopet_xai",
            "--output-run-id",
            "evidence_pack_test",
        ],
        check=True,
    )

    output = results_root / "evidence_pack_test"
    assert (output / "README.md").exists()
    assert (output / "autopet" / "segmentation_metrics.json").exists()
    assert (output / "brain_mri" / "metrics.json").exists()
    assert (output / "traceability" / "requirement_traceability.json").exists()
    assert (output / "evidence_manifest.json").exists()
    assert (output / "INTERPRETATION.md").exists()
    assert (output / "EVALUATION_ALIGNMENT.md").exists()
