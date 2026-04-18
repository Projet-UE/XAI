#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List

from audit_evidence_pack_readiness import audit_evidence_pack, load_mapping, resolve_mapping_path
from validate_result_snapshot import collect_missing_snapshot_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a final readiness gate across tracked autoPET and Brain MRI snapshots "
            "plus evidence-pack rubric readiness."
        )
    )
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--autopet-run-id", type=str, required=True)
    parser.add_argument(
        "--autopet-xai-run-id",
        type=str,
        default=None,
        help=(
            "Optional run id containing protocol XAI benchmark files for autoPET "
            "(xai_analysis_summary.json + method_benchmark.json)."
        ),
    )
    parser.add_argument("--brain-mri-run-id", type=str, required=True)
    parser.add_argument("--evidence-pack-run-id", type=str, required=True)
    parser.add_argument(
        "--mapping-path",
        type=Path,
        default=Path("configs/evaluation_readiness_mapping.json"),
        help="Rubric mapping used by the evidence readiness audit.",
    )
    parser.add_argument(
        "--require-autopet-protocol-benchmark",
        action="store_true",
        help="Require xai_analysis_summary.json and method_benchmark.json in the autoPET snapshot.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Where to write the gate report JSON. Default: <evidence-pack>/READINESS_GATE.json",
    )
    parser.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        default=True,
        help="Exit non-zero when one or more checks fail (default: enabled).",
    )
    parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Always return exit code 0, even when checks fail.",
    )
    return parser.parse_args()


def _check_snapshot(
    *,
    run_dir: Path,
    check_id: str,
    track: str,
    require_protocol_benchmark: bool = False,
) -> Dict[str, Any]:
    try:
        missing = collect_missing_snapshot_files(
            run_dir,
            track,
            require_protocol_benchmark=require_protocol_benchmark,
        )
    except FileNotFoundError as exc:
        return {
            "id": check_id,
            "status": "fail",
            "details": {"error": str(exc), "missing_files": []},
        }

    status = "pass" if not missing else "fail"
    return {
        "id": check_id,
        "status": status,
        "details": {"run_dir": str(run_dir), "missing_files": missing},
    }


def _check_manifest_consistency(
    *,
    evidence_pack_dir: Path,
    autopet_run_id: str,
    autopet_xai_run_id: str | None,
    brain_mri_run_id: str,
    evidence_pack_run_id: str,
) -> Dict[str, Any]:
    manifest_path = evidence_pack_dir / "evidence_manifest.json"
    if not manifest_path.exists():
        return {
            "id": "evidence_manifest_consistency",
            "status": "fail",
            "details": {"error": f"Missing manifest: {manifest_path}"},
        }

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "id": "evidence_manifest_consistency",
            "status": "fail",
            "details": {"error": f"Invalid JSON manifest: {exc}"},
        }

    run_ids = manifest.get("run_ids", {})
    mismatches: List[str] = []
    if run_ids.get("autopet_main") != autopet_run_id:
        mismatches.append("run_ids.autopet_main")
    if autopet_xai_run_id is not None and run_ids.get("autopet_xai_analysis") != autopet_xai_run_id:
        mismatches.append("run_ids.autopet_xai_analysis")
    if run_ids.get("brain_mri_backup") != brain_mri_run_id:
        mismatches.append("run_ids.brain_mri_backup")

    details = {
        "manifest_path": str(manifest_path),
        "expected": {
            "autopet_main": autopet_run_id,
            "autopet_xai_analysis": autopet_xai_run_id,
            "brain_mri_backup": brain_mri_run_id,
            "evidence_pack": evidence_pack_run_id,
        },
        "actual": {
            "autopet_main": run_ids.get("autopet_main"),
            "autopet_xai_analysis": run_ids.get("autopet_xai_analysis"),
            "brain_mri_backup": run_ids.get("brain_mri_backup"),
        },
        "mismatches": mismatches,
    }
    return {
        "id": "evidence_manifest_consistency",
        "status": "pass" if not mismatches else "fail",
        "details": details,
    }


def _check_evidence_readiness(
    *,
    evidence_pack_dir: Path,
    mapping_path: Path,
) -> Dict[str, Any]:
    try:
        resolved = resolve_mapping_path(mapping_path)
        mapping = load_mapping(resolved)
        report = audit_evidence_pack(evidence_pack_dir, mapping)
    except Exception as exc:
        return {
            "id": "evidence_readiness",
            "status": "fail",
            "details": {"error": str(exc)},
        }

    overall = report.get("overall", {})
    status = "pass" if overall.get("status") == "covered" else "fail"
    return {
        "id": "evidence_readiness",
        "status": status,
        "details": {
            "mapping_path": str(resolved),
            "overall_status": overall.get("status"),
            "coverage_score": overall.get("coverage_score"),
            "criteria_fully_covered": overall.get("criteria_fully_covered"),
            "criteria_total": overall.get("criteria_total"),
            "quality_checks_passed": overall.get("quality_checks_passed"),
            "quality_checks_total": overall.get("quality_checks_total"),
        },
    }


def main() -> None:
    args = parse_args()
    results_root = args.results_root

    autopet_dir = results_root / args.autopet_run_id
    autopet_xai_dir = results_root / args.autopet_xai_run_id if args.autopet_xai_run_id else autopet_dir
    brain_dir = results_root / args.brain_mri_run_id
    evidence_pack_dir = results_root / args.evidence_pack_run_id

    checks = [
        _check_snapshot(
            run_dir=autopet_dir,
            check_id="autopet_snapshot",
            track="autopet",
            require_protocol_benchmark=False,
        ),
        _check_snapshot(
            run_dir=autopet_xai_dir,
            check_id="autopet_xai_snapshot",
            track="autopet",
            require_protocol_benchmark=args.require_autopet_protocol_benchmark,
        ),
        _check_snapshot(
            run_dir=brain_dir,
            check_id="brain_mri_snapshot",
            track="brain_mri",
            require_protocol_benchmark=False,
        ),
        _check_manifest_consistency(
            evidence_pack_dir=evidence_pack_dir,
            autopet_run_id=args.autopet_run_id,
            autopet_xai_run_id=args.autopet_xai_run_id,
            brain_mri_run_id=args.brain_mri_run_id,
            evidence_pack_run_id=args.evidence_pack_run_id,
        ),
        _check_evidence_readiness(
            evidence_pack_dir=evidence_pack_dir,
            mapping_path=args.mapping_path,
        ),
    ]

    failed = [check for check in checks if check.get("status") != "pass"]
    overall_status = "pass" if not failed else "fail"

    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "inputs": {
            "results_root": str(results_root),
            "autopet_run_id": args.autopet_run_id,
            "autopet_xai_run_id": args.autopet_xai_run_id,
            "brain_mri_run_id": args.brain_mri_run_id,
            "evidence_pack_run_id": args.evidence_pack_run_id,
            "mapping_path": str(args.mapping_path),
            "strict": args.strict,
        },
        "overall_status": overall_status,
        "checks_total": len(checks),
        "checks_failed": len(failed),
        "checks": checks,
    }

    output_json = args.output_json or (evidence_pack_dir / "READINESS_GATE.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[{overall_status}] readiness gate: {output_json}")
    if failed:
        for check in failed:
            print(f"- failed: {check.get('id')}")

    if args.strict and failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
