#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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


def _exists(pack_dir: Path, relative_path: str) -> bool:
    return (pack_dir / relative_path).exists()


def _score_status(score: float) -> str:
    if score >= 0.999:
        return "covered"
    if score > 0.0:
        return "partial"
    return "missing"


def _aggregate_status(statuses: List[str], fallback_score: float) -> str:
    if statuses and all(status == "covered" for status in statuses):
        return "covered"
    if any(status in {"covered", "partial"} for status in statuses):
        return "partial"
    return _score_status(fallback_score)


def _resolve_key_path(payload: Any, key_path: str) -> Tuple[bool, Any]:
    if key_path == "":
        return True, payload

    current: Any = payload
    for token in key_path.split("."):
        if isinstance(current, dict):
            if token not in current:
                return False, None
            current = current[token]
            continue
        if isinstance(current, list):
            if not token.isdigit():
                return False, None
            index = int(token)
            if index < 0 or index >= len(current):
                return False, None
            current = current[index]
            continue
        return False, None
    return True, current


def _load_json_cached(
    *,
    pack_dir: Path,
    relative_path: str,
    cache: Dict[str, Tuple[Optional[Dict[str, Any]], Optional[str]]],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if relative_path in cache:
        return cache[relative_path]

    full_path = pack_dir / relative_path
    if not full_path.exists():
        result = (None, f"missing file: {relative_path}")
        cache[relative_path] = result
        return result

    try:
        payload = json.loads(full_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive path
        result = (None, f"invalid json at {relative_path}: {exc}")
        cache[relative_path] = result
        return result

    if not isinstance(payload, dict):
        result = (None, f"json root is not an object in {relative_path}")
        cache[relative_path] = result
        return result

    result = (payload, None)
    cache[relative_path] = result
    return result


def _evaluate_quality_checks(
    *,
    pack_dir: Path,
    checks: List[Dict[str, Any]],
    json_cache: Dict[str, Tuple[Optional[Dict[str, Any]], Optional[str]]],
) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    passed = 0

    for index, raw_check in enumerate(checks):
        check = dict(raw_check)
        check_id = str(check.get("id", f"check_{index + 1}"))
        check_type = str(check.get("type", "")).strip()
        target_path = str(check.get("path", "")).strip()
        status = "fail"
        message = "unsupported or invalid quality check"

        if check_type == "json_has_keys":
            payload, error = _load_json_cached(pack_dir=pack_dir, relative_path=target_path, cache=json_cache)
            key_paths = [str(item) for item in check.get("keys", []) if str(item).strip()]
            if error is not None:
                message = error
            elif payload is None:
                message = f"unreadable json: {target_path}"
            elif not key_paths:
                status = "pass"
                message = "no keys requested"
            else:
                missing = [path for path in key_paths if not _resolve_key_path(payload, path)[0]]
                if missing:
                    message = "missing keys: " + ", ".join(missing)
                else:
                    status = "pass"
                    message = "all keys present"

        elif check_type == "json_value_range":
            payload, error = _load_json_cached(pack_dir=pack_dir, relative_path=target_path, cache=json_cache)
            key_path = str(check.get("key", "")).strip()
            minimum = check.get("min")
            maximum = check.get("max")
            if error is not None:
                message = error
            elif payload is None:
                message = f"unreadable json: {target_path}"
            elif key_path == "":
                message = "missing check.key"
            else:
                found, value = _resolve_key_path(payload, key_path)
                if not found:
                    message = f"missing key: {key_path}"
                else:
                    try:
                        numeric_value = float(value)
                    except (TypeError, ValueError):
                        message = f"non-numeric value at {key_path}: {value}"
                    else:
                        if minimum is not None and numeric_value < float(minimum):
                            message = f"value {numeric_value} < min {minimum}"
                        elif maximum is not None and numeric_value > float(maximum):
                            message = f"value {numeric_value} > max {maximum}"
                        else:
                            status = "pass"
                            message = f"value {numeric_value} in range"

        elif check_type == "json_list_min_length":
            payload, error = _load_json_cached(pack_dir=pack_dir, relative_path=target_path, cache=json_cache)
            key_path = str(check.get("key", "")).strip()
            min_length = int(check.get("min_length", 0))
            if error is not None:
                message = error
            elif payload is None:
                message = f"unreadable json: {target_path}"
            elif key_path == "":
                message = "missing check.key"
            else:
                found, value = _resolve_key_path(payload, key_path)
                if not found:
                    message = f"missing key: {key_path}"
                elif not isinstance(value, list):
                    message = f"value at {key_path} is not a list"
                elif len(value) < min_length:
                    message = f"list length {len(value)} < min_length {min_length}"
                else:
                    status = "pass"
                    message = f"list length {len(value)} >= {min_length}"

        elif check_type == "json_count_by_key_value":
            payload, error = _load_json_cached(pack_dir=pack_dir, relative_path=target_path, cache=json_cache)
            list_key = str(check.get("list_key", "")).strip()
            field = str(check.get("field", "")).strip()
            expected_value = check.get("value")
            min_count = int(check.get("min_count", 0))
            if error is not None:
                message = error
            elif payload is None:
                message = f"unreadable json: {target_path}"
            elif list_key == "" or field == "":
                message = "missing list_key or field"
            else:
                found, value = _resolve_key_path(payload, list_key)
                if not found:
                    message = f"missing key: {list_key}"
                elif not isinstance(value, list):
                    message = f"value at {list_key} is not a list"
                else:
                    count = 0
                    for item in value:
                        if isinstance(item, dict) and item.get(field) == expected_value:
                            count += 1
                    if count < min_count:
                        message = f"count {count} < min_count {min_count}"
                    else:
                        status = "pass"
                        message = f"count {count} >= {min_count}"

        result_row = {
            "id": check_id,
            "type": check_type,
            "path": target_path,
            "status": status,
            "message": message,
        }
        results.append(result_row)
        if status == "pass":
            passed += 1

    total = len(results)
    ratio = 1.0 if total == 0 else (passed / total)
    return {
        "checks_total": total,
        "checks_passed": passed,
        "checks_failed": total - passed,
        "quality_ratio": round(ratio, 4),
        "check_results": results,
    }


def _criterion_status(*, required_ratio: float, quality_ratio: float) -> str:
    if required_ratio >= 0.999 and quality_ratio >= 0.999:
        return "covered"
    if required_ratio > 0.0 or quality_ratio > 0.0:
        return "partial"
    return "missing"


def audit_evidence_pack(pack_dir: Path, mapping: Dict[str, Any]) -> Dict[str, Any]:
    repo_root = _repo_root()
    json_cache: Dict[str, Tuple[Optional[Dict[str, Any]], Optional[str]]] = {}
    rubric_rows: List[Dict[str, Any]] = []
    total_weight = 0.0
    total_weighted_score = 0.0
    criteria_total = 0
    criteria_fully_covered = 0
    criteria_partially_covered = 0
    quality_checks_total = 0
    quality_checks_passed = 0

    for rubric in mapping.get("rubrics", []):
        criteria_rows: List[Dict[str, Any]] = []
        rubric_weight_sum = 0.0
        rubric_weighted_score = 0.0
        source_rel = str(rubric.get("source", ""))
        source_path = repo_root / source_rel if source_rel else None

        for criterion in rubric.get("criteria", []):
            required = [str(item) for item in criterion.get("required_evidence", []) if str(item).strip()]
            optional = [str(item) for item in criterion.get("optional_evidence", []) if str(item).strip()]
            quality_checks = [
                dict(item)
                for item in criterion.get("quality_checks", [])
                if isinstance(item, dict)
            ]

            required_present = [item for item in required if _exists(pack_dir, item)]
            optional_present = [item for item in optional if _exists(pack_dir, item)]

            required_total = len(required)
            required_present_count = len(required_present)
            optional_total = len(optional)
            optional_present_count = len(optional_present)

            required_ratio = 1.0 if required_total == 0 else required_present_count / required_total
            quality_summary = _evaluate_quality_checks(
                pack_dir=pack_dir,
                checks=quality_checks,
                json_cache=json_cache,
            )
            quality_ratio = _safe_float(quality_summary.get("quality_ratio", 1.0), 1.0)
            if quality_summary.get("checks_total", 0) == 0:
                coverage_ratio = required_ratio
            else:
                coverage_ratio = required_ratio * quality_ratio

            weight = _safe_float(criterion.get("weight", 1.0), 1.0)
            weighted_score = coverage_ratio * weight
            status = _criterion_status(required_ratio=required_ratio, quality_ratio=quality_ratio)

            criteria_rows.append(
                {
                    "criterion_id": criterion.get("criterion_id", ""),
                    "label": criterion.get("label", ""),
                    "weight": weight,
                    "status": status,
                    "coverage_ratio": round(coverage_ratio, 4),
                    "required_ratio": round(required_ratio, 4),
                    "required_total": required_total,
                    "required_present_count": required_present_count,
                    "required_missing": [item for item in required if item not in required_present],
                    "required_evidence_present": required_present,
                    "optional_total": optional_total,
                    "optional_present_count": optional_present_count,
                    "optional_evidence_present": optional_present,
                    "quality_checks_total": quality_summary["checks_total"],
                    "quality_checks_passed": quality_summary["checks_passed"],
                    "quality_checks_failed": quality_summary["checks_failed"],
                    "quality_ratio": quality_summary["quality_ratio"],
                    "quality_check_results": quality_summary["check_results"],
                }
            )

            criteria_total += 1
            if status == "covered":
                criteria_fully_covered += 1
            elif status == "partial":
                criteria_partially_covered += 1
            quality_checks_total += int(quality_summary["checks_total"])
            quality_checks_passed += int(quality_summary["checks_passed"])
            rubric_weight_sum += weight
            rubric_weighted_score += weighted_score

        rubric_score = (rubric_weighted_score / rubric_weight_sum) if rubric_weight_sum > 0 else 0.0
        rubric_status = _aggregate_status(
            [str(row.get("status", "missing")) for row in criteria_rows],
            rubric_score,
        )

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
    overall_status = _aggregate_status(
        [str(row.get("status", "missing")) for rubric in rubric_rows for row in rubric.get("criteria", [])],
        overall_score,
    )

    return {
        "schema_version": mapping.get("schema_version", "1.0.0"),
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pack_dir": str(pack_dir.resolve()),
        "overall": {
            "coverage_score": round(overall_score, 4),
            "status": overall_status,
            "criteria_total": criteria_total,
            "criteria_fully_covered": criteria_fully_covered,
            "criteria_partially_covered": criteria_partially_covered,
            "quality_checks_total": quality_checks_total,
            "quality_checks_passed": quality_checks_passed,
            "quality_checks_failed": quality_checks_total - quality_checks_passed,
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
        f"- Partially covered criteria: `{overall.get('criteria_partially_covered', 0)}`",
        f"- Quality checks passed: `{overall.get('quality_checks_passed', 0)}/{overall.get('quality_checks_total', 0)}`",
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
        lines.append("| Criterion | Status | Coverage | Quality checks | Missing required evidence |")
        lines.append("|---|---|---:|---:|---|")
        for criterion in rubric.get("criteria", []):
            missing = criterion.get("required_missing", [])
            missing_text = ", ".join(f"`{item}`" for item in missing) if missing else "none"
            lines.append(
                "| "
                f"{criterion.get('criterion_id', '')} — {criterion.get('label', '')} | "
                f"{criterion.get('status', 'missing')} | "
                f"{_safe_float(criterion.get('coverage_ratio', 0.0)):.2f} | "
                f"{criterion.get('quality_checks_passed', 0)}/{criterion.get('quality_checks_total', 0)} | "
                f"{missing_text} |"
            )
            failed_checks = [
                check
                for check in criterion.get("quality_check_results", [])
                if check.get("status") != "pass"
            ]
            if failed_checks:
                lines.append("|  |  |  |  |  |")
                for check in failed_checks:
                    lines.append(
                        f"|  |  |  |  | failed check `{check.get('id', '')}` "
                        f"({check.get('type', '')}): {check.get('message', '')} |"
                    )
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "- `covered`: required evidence is complete and all quality checks pass.",
            "- `partial`: required evidence and/or quality checks are incomplete.",
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
