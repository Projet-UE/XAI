from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_run_readiness_gate_pass(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results_root = tmp_path / "results"

    autopet_id = "autopet_ok"
    brain_id = "brain_ok"
    evidence_id = "evidence_ok"

    autopet_dir = results_root / autopet_id
    brain_dir = results_root / brain_id
    evidence_dir = results_root / evidence_id

    for name in ["segmentation_metrics.json", "run_config.json", "review_cases.json", "README.md"]:
        (autopet_dir / name).parent.mkdir(parents=True, exist_ok=True)
        if name.endswith(".json"):
            _write_json(autopet_dir / name, {"ok": True})
        else:
            (autopet_dir / name).write_text("ok", encoding="utf-8")

    for name in ["metrics.json", "run_config.json", "README.md"]:
        (brain_dir / name).parent.mkdir(parents=True, exist_ok=True)
        if name.endswith(".json"):
            _write_json(brain_dir / name, {"ok": True})
        else:
            (brain_dir / name).write_text("ok", encoding="utf-8")

    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "README.md").write_text("ok", encoding="utf-8")
    _write_json(
        evidence_dir / "evidence_manifest.json",
        {
            "run_ids": {
                "autopet_main": autopet_id,
                "brain_mri_backup": brain_id,
            }
        },
    )

    mapping_path = tmp_path / "mapping.json"
    _write_json(
        mapping_path,
        {
            "schema_version": "1.0.0",
            "rubrics": [
                {
                    "rubric_id": "demo",
                    "rubric_title": "Demo",
                    "source": "new/Materials/demo.xlsx",
                    "criteria": [
                        {
                            "criterion_id": "C1",
                            "label": "readme present",
                            "required_evidence": ["README.md"],
                        }
                    ],
                }
            ],
        },
    )

    script_path = repo_root / "scripts" / "run_readiness_gate.py"
    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--results-root",
            str(results_root),
            "--autopet-run-id",
            autopet_id,
            "--brain-mri-run-id",
            brain_id,
            "--evidence-pack-run-id",
            evidence_id,
            "--mapping-path",
            str(mapping_path),
        ],
        check=True,
    )

    report = json.loads((evidence_dir / "READINESS_GATE.json").read_text(encoding="utf-8"))
    assert report["overall_status"] == "pass"
    assert report["checks_failed"] == 0


def test_run_readiness_gate_fail_on_manifest_mismatch(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results_root = tmp_path / "results"

    autopet_id = "autopet_ok"
    brain_id = "brain_ok"
    evidence_id = "evidence_fail"

    autopet_dir = results_root / autopet_id
    brain_dir = results_root / brain_id
    evidence_dir = results_root / evidence_id

    for name in ["segmentation_metrics.json", "run_config.json", "review_cases.json", "README.md"]:
        if name.endswith(".json"):
            _write_json(autopet_dir / name, {"ok": True})
        else:
            (autopet_dir / name).parent.mkdir(parents=True, exist_ok=True)
            (autopet_dir / name).write_text("ok", encoding="utf-8")

    for name in ["metrics.json", "run_config.json", "README.md"]:
        if name.endswith(".json"):
            _write_json(brain_dir / name, {"ok": True})
        else:
            (brain_dir / name).parent.mkdir(parents=True, exist_ok=True)
            (brain_dir / name).write_text("ok", encoding="utf-8")

    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "README.md").write_text("ok", encoding="utf-8")
    _write_json(
        evidence_dir / "evidence_manifest.json",
        {
            "run_ids": {
                "autopet_main": "wrong_autopet",
                "brain_mri_backup": brain_id,
            }
        },
    )

    mapping_path = tmp_path / "mapping.json"
    _write_json(
        mapping_path,
        {
            "schema_version": "1.0.0",
            "rubrics": [
                {
                    "rubric_id": "demo",
                    "rubric_title": "Demo",
                    "source": "new/Materials/demo.xlsx",
                    "criteria": [
                        {
                            "criterion_id": "C1",
                            "label": "readme present",
                            "required_evidence": ["README.md"],
                        }
                    ],
                }
            ],
        },
    )

    script_path = repo_root / "scripts" / "run_readiness_gate.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--results-root",
            str(results_root),
            "--autopet-run-id",
            autopet_id,
            "--brain-mri-run-id",
            brain_id,
            "--evidence-pack-run-id",
            evidence_id,
            "--mapping-path",
            str(mapping_path),
        ],
        check=False,
    )
    assert proc.returncode != 0

    report = json.loads((evidence_dir / "READINESS_GATE.json").read_text(encoding="utf-8"))
    assert report["overall_status"] == "fail"
    failed_ids = {row["id"] for row in report["checks"] if row["status"] == "fail"}
    assert "evidence_manifest_consistency" in failed_ids


def test_run_readiness_gate_protocol_benchmark_on_dedicated_run(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results_root = tmp_path / "results"

    autopet_id = "autopet_main"
    autopet_xai_id = "autopet_xai"
    brain_id = "brain_ok"
    evidence_id = "evidence_ok"

    autopet_dir = results_root / autopet_id
    autopet_xai_dir = results_root / autopet_xai_id
    brain_dir = results_root / brain_id
    evidence_dir = results_root / evidence_id

    for name in ["segmentation_metrics.json", "run_config.json", "review_cases.json", "README.md"]:
        if name.endswith(".json"):
            _write_json(autopet_dir / name, {"ok": True})
            _write_json(autopet_xai_dir / name, {"ok": True})
        else:
            (autopet_dir / name).parent.mkdir(parents=True, exist_ok=True)
            (autopet_dir / name).write_text("ok", encoding="utf-8")
            (autopet_xai_dir / name).parent.mkdir(parents=True, exist_ok=True)
            (autopet_xai_dir / name).write_text("ok", encoding="utf-8")

    _write_json(autopet_xai_dir / "xai_analysis_summary.json", {"ok": True})
    _write_json(autopet_xai_dir / "method_benchmark.json", {"ranking": [{"method": "m", "rank": 1}]})

    for name in ["metrics.json", "run_config.json", "README.md"]:
        if name.endswith(".json"):
            _write_json(brain_dir / name, {"ok": True})
        else:
            (brain_dir / name).parent.mkdir(parents=True, exist_ok=True)
            (brain_dir / name).write_text("ok", encoding="utf-8")

    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "README.md").write_text("ok", encoding="utf-8")
    _write_json(
        evidence_dir / "evidence_manifest.json",
        {
            "run_ids": {
                "autopet_main": autopet_id,
                "autopet_xai_analysis": autopet_xai_id,
                "brain_mri_backup": brain_id,
            }
        },
    )

    mapping_path = tmp_path / "mapping.json"
    _write_json(
        mapping_path,
        {
            "schema_version": "1.0.0",
            "rubrics": [
                {
                    "rubric_id": "demo",
                    "rubric_title": "Demo",
                    "source": "new/Materials/demo.xlsx",
                    "criteria": [
                        {
                            "criterion_id": "C1",
                            "label": "readme present",
                            "required_evidence": ["README.md"],
                        }
                    ],
                }
            ],
        },
    )

    script_path = repo_root / "scripts" / "run_readiness_gate.py"
    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--results-root",
            str(results_root),
            "--autopet-run-id",
            autopet_id,
            "--autopet-xai-run-id",
            autopet_xai_id,
            "--brain-mri-run-id",
            brain_id,
            "--evidence-pack-run-id",
            evidence_id,
            "--mapping-path",
            str(mapping_path),
            "--require-autopet-protocol-benchmark",
        ],
        check=True,
    )

    report = json.loads((evidence_dir / "READINESS_GATE.json").read_text(encoding="utf-8"))
    assert report["overall_status"] == "pass"
