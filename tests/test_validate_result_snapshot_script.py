from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_validate_snapshot_autopet_ok(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_autopet"
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ["segmentation_metrics.json", "run_config.json", "review_cases.json", "README.md"]:
        (run_dir / name).write_text("{}", encoding="utf-8")
    (run_dir / "xai_analysis_summary.json").write_text("{}", encoding="utf-8")
    (run_dir / "method_benchmark.json").write_text("{}", encoding="utf-8")

    script = Path(__file__).resolve().parents[1] / "scripts" / "validate_result_snapshot.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--run-dir",
            str(run_dir),
            "--track",
            "autopet",
            "--require-protocol-benchmark",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "[ok] Snapshot valid" in result.stdout


def test_validate_snapshot_detects_missing_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_missing"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "segmentation_metrics.json").write_text("{}", encoding="utf-8")
    (run_dir / "README.md").write_text("{}", encoding="utf-8")

    script = Path(__file__).resolve().parents[1] / "scripts" / "validate_result_snapshot.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--run-dir",
            str(run_dir),
            "--track",
            "autopet",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "missing" in result.stderr.lower()
