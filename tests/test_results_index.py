from __future__ import annotations

import json
from pathlib import Path


def test_results_index_has_required_fields() -> None:
    index_path = Path(__file__).resolve().parents[1] / "results" / "index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "1.0.0"
    assert isinstance(payload.get("run_states"), list)
    assert len(payload["run_states"]) >= 4

    required_keys = {
        "state_name",
        "run_id",
        "track",
        "split_name",
        "dataset_split_hash_sha256",
        "checkpoint_reference",
        "metrics_file",
        "run_config_file",
    }
    for state in payload["run_states"]:
        assert required_keys.issubset(state.keys())

    checksums = payload.get("script_checksums_sha256", {})
    assert "scripts/autopet_analyze_xai.py" in checksums
    assert "scripts/autopet_export_results.py" in checksums
