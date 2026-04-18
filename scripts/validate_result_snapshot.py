#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List


REQUIRED_FILES: Dict[str, List[str]] = {
    "autopet": [
        "segmentation_metrics.json",
        "run_config.json",
        "review_cases.json",
        "README.md",
    ],
    "brain_mri": [
        "metrics.json",
        "run_config.json",
        "README.md",
    ],
}


def collect_missing_snapshot_files(
    run_dir: Path,
    track: str,
    *,
    require_protocol_benchmark: bool = False,
) -> List[str]:
    if track not in REQUIRED_FILES:
        raise ValueError(f"Unknown track: {track}")
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    missing: List[str] = []
    for name in REQUIRED_FILES[track]:
        if not (run_dir / name).exists():
            missing.append(name)

    if track == "autopet" and require_protocol_benchmark:
        if not (run_dir / "xai_analysis_summary.json").exists():
            missing.append("xai_analysis_summary.json")
        if not (run_dir / "method_benchmark.json").exists():
            missing.append("method_benchmark.json")

    return sorted(set(missing))


def assert_snapshot_valid(
    run_dir: Path,
    track: str,
    *,
    require_protocol_benchmark: bool = False,
) -> None:
    missing = collect_missing_snapshot_files(
        run_dir,
        track,
        require_protocol_benchmark=require_protocol_benchmark,
    )
    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(f"Snapshot validation failed for {run_dir}: missing {missing_text}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate that a tracked results snapshot is self-contained.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--track", type=str, required=True, choices=sorted(REQUIRED_FILES.keys()))
    parser.add_argument(
        "--require-protocol-benchmark",
        action="store_true",
        help="Require method benchmark files for protocol-grade XAI reporting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert_snapshot_valid(
        args.run_dir,
        args.track,
        require_protocol_benchmark=args.require_protocol_benchmark,
    )
    print(f"[ok] Snapshot valid: {args.run_dir}")


if __name__ == "__main__":
    main()
