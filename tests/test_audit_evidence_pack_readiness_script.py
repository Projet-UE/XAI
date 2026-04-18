from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_audit_evidence_pack_readiness_scoring(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pack_dir = tmp_path / "results" / "evidence_pack_test"
    pack_dir.mkdir(parents=True, exist_ok=True)

    (pack_dir / "a.txt").write_text("ok", encoding="utf-8")
    (pack_dir / "c.txt").write_text("ok", encoding="utf-8")

    mapping = {
        "schema_version": "1.0.0",
        "rubrics": [
            {
                "rubric_id": "rubric-demo",
                "rubric_title": "Rubric demo",
                "source": "new/Materials/does_not_need_to_exist.xlsx",
                "criteria": [
                    {
                        "criterion_id": "CRIT-1",
                        "label": "First criterion",
                        "weight": 1.0,
                        "required_evidence": ["a.txt", "b.txt"],
                        "optional_evidence": [],
                    },
                    {
                        "criterion_id": "CRIT-2",
                        "label": "Second criterion",
                        "weight": 1.0,
                        "required_evidence": ["c.txt"],
                        "optional_evidence": [],
                    },
                ],
            }
        ],
    }
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")

    script_path = repo_root / "scripts" / "audit_evidence_pack_readiness.py"
    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--pack-dir",
            str(pack_dir),
            "--mapping-path",
            str(mapping_path),
        ],
        check=True,
    )

    report_path = pack_dir / "EVALUATION_READINESS.json"
    markdown_path = pack_dir / "EVALUATION_READINESS.md"
    assert report_path.exists()
    assert markdown_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["overall"]["status"] == "partial"
    assert abs(report["overall"]["coverage_score"] - 0.75) < 1e-6
    assert report["overall"]["criteria_total"] == 2
    assert report["overall"]["criteria_fully_covered"] == 1

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "CRIT-1" in markdown
    assert "CRIT-2" in markdown
