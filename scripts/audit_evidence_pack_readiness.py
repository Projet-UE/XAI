#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit an evidence-pack folder against evaluation rubrics "
            "(client, soutenance, plan projet) and export a scored readiness report."
        )
    )
    parser.add_argument("--pack-dir", type=Path, required=True, help="Path to the generated evidence pack directory.")
    parser.add_argument(
        "--mapping-path",
        type=Path,
        default=Path("configs/evaluation_readiness_mapping.json"),
        help="JSON mapping file that links rubric criteria to expected evidence paths.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output JSON path. Default: <pack-dir>/EVALUATION_READINESS.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Output Markdown path. Default: <pack-dir>/EVALUATION_READINESS.md",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit code when any criterion is not fully covered.",
    )
    return parser.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_mapping_path(path: Path) -> Path:
    if path.exists():
        return path
    candidate = _repo_root() / path
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Mapping file not found: {path}")


def load_mapping(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _criterion_status(required_total: int, required_present: int) -> str:
    if required_total == 0:
        return "covered"
    if required_present == required_total:
        return "covered"
    if required_present > 0:
        return "partial"
    return "missing"


def _exists(pack_dir: Path, relative_path: str) -> bool:
    return (pack_dir / relative_path).exists()


def _score_status(score: float) -> str:
    if score >= 0.999:
        return "covered"
    if score > 0.0:
        return "partial"
    return "missing"


def audit_evidence_pack(pack_dir: Path, mapping: Dict[str, Any]) -> Dict[str, Any]:
    repo_root = _repo_root()
    rubric_rows: List[Dict[str, Any]] = []
    total_weight = 0.0
    total_weighted_score = 0.0
    criteria_total = 0
    criteria_fully_covered = 0

    for rubric in mapping.get("rubrics", []):
        criteria_rows: List[Dict[str, Any]] = []
        rubric_weight_sum = 0.0
        rubric_weighted_score = 0.0
        source_rel = str(rubric.get("source", ""))
        source_path = repo_root / source_rel if source_rel else None

        for criterion in rubric.get("criteria", []):
            required = [str(item) for item in criterion.get("required_evidence", []) if str(item).strip()]
            optional = [str(item) for item in criterion.get("optional_evidence", []) if str(item).strip()]

            required_present = [item for item in required if _exists(pack_dir, item)]
            optional_present = [item for item in optional if _exists(pack_dir, item)]

            required_total = len(required)
            required_present_count = len(required_present)
            optional_total = len(optional)
            optional_present_count = len(optional_present)

            coverage_ratio = 1.0 if required_total == 0 else required_present_count / required_total
            weight = _safe_float(criterion.get("weight", 1.0), 1.0)
            weighted_score = coverage_ratio * weight
            status = _criterion_status(required_total, required_present_count)

            criteria_rows.append(
                {
                    "criterion_id": criterion.get("criterion_id", ""),
                    "label": criterion.get("label", ""),
                    "weight": weight,
                    "status": status,
                    "coverage_ratio": round(coverage_ratio, 4),
                    "required_total": required_total,
                    "required_present_count": required_present_count,
                    "required_missing": [item for item in required if item not in required_present],
                    "required_evidence_present": required_present,
                    "optional_total": optional_total,
                    "optional_present_count": optional_present_count,
                    "optional_evidence_present": optional_present,
                }
            )

            criteria_total += 1
            if status == "covered":
                criteria_fully_covered += 1
            rubric_weight_sum += weight
            rubric_weighted_score += weighted_score

        rubric_score = (rubric_weighted_score / rubric_weight_sum) if rubric_weight_sum > 0 else 0.0
        rubric_status = _score_status(rubric_score)

        rubric_rows.append(
            {
                "rubric_id": rubric.get("rubric_id", ""),
                "rubric_title": rubric.get("rubric_title", ""),
                "source": source_rel,
                "source_exists": bool(source_path and source_path.exists()),
                "coverage_score": round(rubric_score, 4),
                "status": rubric_status,
                "criteria": criteria_rows,
            }
        )

        total_weight += rubric_weight_sum
        total_weighted_score += rubric_weighted_score

    overall_score = (total_weighted_score / total_weight) if total_weight > 0 else 0.0
    overall_status = _score_status(overall_score)

    return {
        "schema_version": mapping.get("schema_version", "1.0.0"),
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pack_dir": str(pack_dir.resolve()),
        "overall": {
            "coverage_score": round(overall_score, 4),
            "status": overall_status,
            "criteria_total": criteria_total,
            "criteria_fully_covered": criteria_fully_covered,
        },
        "rubrics": rubric_rows,
    }


def render_markdown(report: Dict[str, Any]) -> str:
    overall = report.get("overall", {})
    lines = [
        "# Evaluation Readiness Report",
        "",
        f"- Overall coverage score: `{overall.get('coverage_score', 0.0):.4f}`",
        f"- Overall status: `{overall.get('status', 'missing')}`",
        f"- Fully covered criteria: `{overall.get('criteria_fully_covered', 0)}/{overall.get('criteria_total', 0)}`",
        "",
    ]

    for rubric in report.get("rubrics", []):
        lines.append(f"## {rubric.get('rubric_title', rubric.get('rubric_id', 'rubric'))}")
        lines.append("")
        lines.append(f"- Rubric id: `{rubric.get('rubric_id', '')}`")
        lines.append(f"- Source: `{rubric.get('source', '')}`")
        lines.append(f"- Source available in repository: `{rubric.get('source_exists', False)}`")
        lines.append(f"- Coverage score: `{_safe_float(rubric.get('coverage_score', 0.0)):.4f}`")
        lines.append(f"- Status: `{rubric.get('status', 'missing')}`")
        lines.append("")
        lines.append("| Criterion | Status | Coverage | Missing required evidence |")
        lines.append("|---|---|---:|---|")
        for criterion in rubric.get("criteria", []):
            missing = criterion.get("required_missing", [])
            missing_text = ", ".join(f"`{item}`" for item in missing) if missing else "none"
            lines.append(
                "| "
                f"{criterion.get('criterion_id', '')} — {criterion.get('label', '')} | "
                f"{criterion.get('status', 'missing')} | "
                f"{_safe_float(criterion.get('coverage_ratio', 0.0)):.2f} | "
                f"{missing_text} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "- `covered`: all required evidence files for the criterion are present.",
            "- `partial`: only part of the required evidence is present.",
            "- `missing`: no required evidence found for the criterion.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    pack_dir = args.pack_dir
    mapping_path = resolve_mapping_path(args.mapping_path)
    mapping = load_mapping(mapping_path)

    report = audit_evidence_pack(pack_dir, mapping)

    output_json = args.output_json or (pack_dir / "EVALUATION_READINESS.json")
    output_md = args.output_md or (pack_dir / "EVALUATION_READINESS.md")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(render_markdown(report) + "\n", encoding="utf-8")

    if args.strict and report.get("overall", {}).get("status") != "covered":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
