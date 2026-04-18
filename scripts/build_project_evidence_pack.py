#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from audit_evidence_pack_readiness import (
    audit_evidence_pack,
    load_mapping as load_readiness_mapping,
    render_markdown as render_readiness_markdown,
    resolve_mapping_path,
)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(payload: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a single reviewable evidence pack from tracked autoPET and Brain MRI runs, "
            "including metrics, benchmark summaries, selected figures, and requirement traceability."
        )
    )
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--run-index-path", type=Path, default=Path("results/index.json"))
    parser.add_argument("--autopet-main-run-id", type=str, required=True)
    parser.add_argument("--autopet-comparison-run-id", type=str, required=True)
    parser.add_argument("--brain-mri-run-id", type=str, required=True)
    parser.add_argument("--autopet-xai-analysis-run-id", type=str, default=None)
    parser.add_argument("--brain-mri-xai-benchmark-run-id", type=str, default=None)
    parser.add_argument("--output-run-id", type=str, default=None)
    parser.add_argument("--max-figures-per-track", type=int, default=6)
    parser.add_argument(
        "--evaluation-mapping-path",
        type=Path,
        default=Path("configs/evaluation_readiness_mapping.json"),
        help="Rubric-to-evidence mapping used to generate EVALUATION_READINESS.{json,md}.",
    )
    return parser.parse_args()


def _copy_files(paths: Iterable[Path], destination_root: Path) -> List[str]:
    copied: List[str] = []
    used_names: Dict[str, int] = {}
    for source in paths:
        if not source.exists() or not source.is_file():
            continue
        candidate = source.name
        if candidate in used_names or (destination_root / candidate).exists():
            candidate = f"{source.parent.name}__{source.name}"
        if candidate in used_names or (destination_root / candidate).exists():
            base = Path(candidate).stem
            suffix = Path(candidate).suffix
            index = 2
            while True:
                alt = f"{base}_{index}{suffix}"
                if alt not in used_names and not (destination_root / alt).exists():
                    candidate = alt
                    break
                index += 1
        used_names[candidate] = used_names.get(candidate, 0) + 1
        target = destination_root / candidate
        ensure_dir(target.parent)
        shutil.copy2(source, target)
        copied.append(candidate)
    return copied


def _collect_pngs(folder: Path, limit: int) -> List[Path]:
    if not folder.exists():
        return []
    images = sorted(path for path in folder.rglob("*.png") if path.is_file())
    return images[:limit]


def _require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def _traceability(requirement_id: str, description: str, evidence_files: List[str]) -> Dict[str, Any]:
    return {
        "requirement_id": requirement_id,
        "description": description,
        "status": "covered" if evidence_files else "missing_evidence",
        "evidence_files": evidence_files,
    }


def _format_delta(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    sign = "+" if value >= 0 else "-"
    return f"{sign}{abs(float(value)):.4f}"


def _autopet_tradeoff_lines(comparison: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    tradeoffs = comparison.get("tradeoffs", {})
    best = tradeoffs.get("best_dice_vs_raw", {})
    low_fp = tradeoffs.get("low_fp_vs_raw", {})
    if best:
        lines.append(
            "- Variante best-Dice vs raw: "
            f"Dice {_format_delta(best.get('mean_dice_delta'))}, "
            f"FN {_format_delta(best.get('mean_false_negative_volume_ml_delta'))} mL, "
            f"FP {_format_delta(best.get('mean_false_positive_volume_ml_delta'))} mL."
        )
    if low_fp:
        lines.append(
            "- Variante low-FP vs raw: "
            f"Dice {_format_delta(low_fp.get('mean_dice_delta'))}, "
            f"FN {_format_delta(low_fp.get('mean_false_negative_volume_ml_delta'))} mL, "
            f"FP {_format_delta(low_fp.get('mean_false_positive_volume_ml_delta'))} mL."
        )
    return lines


def _extract_top_method(payload: Dict[str, Any]) -> Optional[str]:
    ranking = payload.get("ranking", [])
    if ranking:
        return ranking[0].get("method")
    ranking = payload.get("method_benchmark", {}).get("ranking", [])
    if ranking:
        return ranking[0].get("method")
    return payload.get("preferred_method")


def _status_from_expected(output_dir: Path, expected_paths: List[str]) -> str:
    if not expected_paths:
        return "missing"
    present = [item for item in expected_paths if (output_dir / item).exists()]
    if len(present) == len(expected_paths):
        return "covered"
    if present:
        return "partial"
    return "missing"


def _build_evaluation_alignment(output_dir: Path) -> str:
    sections = [
        {
            "criterion": "Client clôture — état fonctionnel des livrables",
            "source": "EVALUATION_Client_2025.xlsx (phase 3: clôture, fonctionnel)",
            "expected": [
                "autopet/segmentation_metrics.json",
                "autopet/comparison.json",
                "brain_mri/metrics.json",
            ],
            "note": "Montre que les pipelines produisent des sorties exploitables et comparables.",
        },
        {
            "criterion": "Client clôture — état qualitatif des livrables",
            "source": "EVALUATION_Client_2025.xlsx (phase 3: clôture, qualitatif)",
            "expected": [
                "autopet/method_benchmark.json",
                "brain_mri/xai_method_benchmark.json",
                "INTERPRETATION.md",
            ],
            "note": "Supporte l'analyse des méthodes XAI et l'interprétation attendue en revue cliente.",
        },
        {
            "criterion": "Client clôture — maintenance / évolutivité",
            "source": "EVALUATION_Client_2025.xlsx (phase 3: maintenance et évolution)",
            "expected": [
                "traceability/run_index.json",
                "evidence_manifest.json",
                "traceability/requirement_traceability.json",
            ],
            "note": "Assure la traçabilité des runs, des versions et des preuves pour la reprise ultérieure.",
        },
        {
            "criterion": "Soutenance — contexte / livrables / preuves",
            "source": "EVALUATION_Soutenance_projet 2026 .xlsx",
            "expected": [
                "README.md",
                "autopet/comparison.json",
                "autopet/segmentation_metrics.json",
                "brain_mri/metrics.json",
                "traceability/requirement_traceability.json",
            ],
            "note": "Fournit des artefacts directs pour justifier le contexte, les livrables et les résultats.",
        },
        {
            "criterion": "Plan projet V3 — accès, couverture, traçabilité",
            "source": "EVALUATION_Plan_Projet_et_UE_Projet.xlsx",
            "expected": [
                "README.md",
                "INTERPRETATION.md",
                "traceability/requirement_traceability.json",
                "evidence_manifest.json",
            ],
            "note": "Rend vérifiable l'accès aux preuves citées dans le plan projet.",
        },
    ]

    lines = [
        "# Evaluation Alignment",
        "",
        "This file maps the evidence pack to key criteria extracted from the project evaluation rubrics",
        "available in `new/Materials` (client, soutenance, and plan-projet grids).",
        "",
    ]
    for row in sections:
        status = _status_from_expected(output_dir, row["expected"])
        lines.append(f"## {row['criterion']}")
        lines.append("")
        lines.append(f"- Source rubric: `{row['source']}`")
        lines.append(f"- Status: `{status}`")
        lines.append("- Expected evidence:")
        for expected_path in row["expected"]:
            exists = (output_dir / expected_path).exists()
            marker = "x" if exists else " "
            lines.append(f"  - [{marker}] `{expected_path}`")
        lines.append(f"- Why it matters: {row['note']}")
        lines.append("")

    lines.extend(
        [
            "## Notes",
            "",
            "- This alignment file is an operational checklist for review sessions.",
            "- It does not replace model metrics; it links rubric questions to concrete evidence paths.",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_demo_runbook(
    *,
    autopet_main_run_id: str,
    autopet_comparison_run_id: str,
    brain_mri_run_id: str,
    autopet_top_method: Optional[str],
    brain_top_method: Optional[str],
) -> str:
    lines = [
        "# Demo Runbook (2-3 minutes)",
        "",
        "Use this script during soutenance/client review to cover the mandatory acceptance points",
        "without running heavy training jobs live.",
        "",
        "## Objective coverage",
        "",
        "- `REQ-C2` (pipeline exécutable): show script interfaces and validated snapshots.",
        "- `REQ-C4` (explications XAI): show attribution outputs and method benchmark files.",
        "- `REQ-C5` (analyse critique): show tradeoff and interpretation files.",
        "",
        "## Step-by-step sequence",
        "",
        "### 1. Context and tracked runs (20-30s)",
        f"- Open `README.md` and state the three tracked runs: `{autopet_main_run_id}`, `{autopet_comparison_run_id}`, `{brain_mri_run_id}`.",
        "- Point to `traceability/requirement_traceability.json` for requirement coverage.",
        "",
        "### 2. Core segmentation result and tradeoff (35-45s)",
        "- Open `autopet/segmentation_metrics.json` (main result).",
        "- Open `autopet/comparison.json` and explain best-Dice vs low-FP tradeoff.",
        "- This covers the analysis expected by `REQ-C5`.",
        "",
        "### 3. autoPET XAI evidence (30-40s)",
        "- Open one or two files in `autopet/figures/`.",
        "- Open `autopet/method_benchmark.json` and cite top method.",
        f"- Current top method: `{autopet_top_method if autopet_top_method else 'n/a'}`.",
        "- This covers explainability demonstration for `REQ-C4`.",
        "",
        "### 4. Brain MRI backup evidence (25-35s)",
        "- Open `brain_mri/metrics.json`.",
        "- Open `brain_mri/xai_method_benchmark.json` when present.",
        f"- Current top method: `{brain_top_method if brain_top_method else 'n/a'}`.",
        "",
        "### 5. Quick reproducibility proof (20-30s)",
        "- Show validation commands (do not launch heavy training):",
        "",
        "```bash",
        "python scripts/validate_result_snapshot.py \\",
        f"  --run-dir results/{autopet_main_run_id} \\",
        "  --track autopet",
        "",
        "python scripts/validate_result_snapshot.py \\",
        f"  --run-dir results/{brain_mri_run_id} \\",
        "  --track brain_mri",
        "```",
        "",
        "These checks support `REQ-C2` by proving runnable, self-contained tracked outputs.",
        "",
        "## Final one-line project message",
        "",
        "autoPET FDG is the primary scientific line (segmentation + XAI tradeoff analysis), and Brain MRI is a reproducible backup line that confirms the XAI workflow on a second medical setting.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    today = dt.date.today().strftime("%Y%m%d")
    output_run_id = args.output_run_id or f"evidence_pack_{today}"
    output_dir = args.results_root / output_run_id
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir = ensure_dir(output_dir)

    autopet_main_dir = args.results_root / args.autopet_main_run_id
    autopet_comparison_dir = args.results_root / args.autopet_comparison_run_id
    brain_mri_dir = args.results_root / args.brain_mri_run_id
    autopet_xai_dir = args.results_root / (
        args.autopet_xai_analysis_run_id or args.autopet_main_run_id
    )
    brain_mri_xai_dir = args.results_root / args.brain_mri_xai_benchmark_run_id if args.brain_mri_xai_benchmark_run_id else None

    autopet_metrics_path = _require_file(autopet_main_dir / "segmentation_metrics.json", "autoPET main metrics")
    autopet_comparison_path = _require_file(autopet_comparison_dir / "comparison.json", "autoPET comparison")
    brain_mri_metrics_path = _require_file(brain_mri_dir / "metrics.json", "Brain MRI metrics")

    autopet_xai_summary_path = autopet_xai_dir / "xai_analysis_summary.json"
    autopet_method_benchmark_path = autopet_xai_dir / "method_benchmark.json"
    if not autopet_method_benchmark_path.exists() and autopet_xai_summary_path.exists():
        # Fallback: analysis summary may already contain benchmark details.
        autopet_method_benchmark_path = autopet_xai_summary_path

    autopet_metrics = load_json(autopet_metrics_path)
    autopet_comparison = load_json(autopet_comparison_path)
    brain_mri_metrics = load_json(brain_mri_metrics_path)
    run_index = load_json(args.run_index_path) if args.run_index_path.exists() else {}
    autopet_xai_summary = load_json(autopet_xai_summary_path) if autopet_xai_summary_path.exists() else {}
    autopet_method_benchmark = (
        load_json(autopet_method_benchmark_path) if autopet_method_benchmark_path.exists() else {}
    )

    brain_mri_xai_benchmark = {}
    if brain_mri_xai_dir is not None:
        benchmark_path = brain_mri_xai_dir / "xai_method_benchmark.json"
        if benchmark_path.exists():
            brain_mri_xai_benchmark = load_json(benchmark_path)

    autopet_target = ensure_dir(output_dir / "autopet")
    brain_mri_target = ensure_dir(output_dir / "brain_mri")
    traceability_target = ensure_dir(output_dir / "traceability")

    save_json(autopet_metrics, autopet_target / "segmentation_metrics.json")
    save_json(autopet_comparison, autopet_target / "comparison.json")
    if autopet_xai_summary:
        save_json(autopet_xai_summary, autopet_target / "xai_analysis_summary.json")
    if autopet_method_benchmark:
        save_json(autopet_method_benchmark, autopet_target / "method_benchmark.json")

    save_json(brain_mri_metrics, brain_mri_target / "metrics.json")
    if brain_mri_xai_benchmark:
        save_json(brain_mri_xai_benchmark, brain_mri_target / "xai_method_benchmark.json")

    autopet_figures = _collect_pngs(autopet_main_dir / "figures", args.max_figures_per_track)
    if len(autopet_figures) < args.max_figures_per_track:
        autopet_figures.extend(
            _collect_pngs(autopet_xai_dir / "figures", args.max_figures_per_track - len(autopet_figures))
        )
    autopet_figures = autopet_figures[: args.max_figures_per_track]
    copied_autopet_figures = _copy_files(autopet_figures, ensure_dir(autopet_target / "figures"))

    brain_figures = _collect_pngs(brain_mri_dir / "figures", args.max_figures_per_track)
    if len(brain_figures) < args.max_figures_per_track:
        brain_figures.extend(
            _collect_pngs(brain_mri_dir / "xai", args.max_figures_per_track - len(brain_figures))
        )
    brain_figures = brain_figures[: args.max_figures_per_track]
    copied_brain_figures = _copy_files(brain_figures, ensure_dir(brain_mri_target / "figures"))

    traceability_map = {
        "requirements": [
            _traceability(
                "REQ-C2",
                "Pipeline expérimental XAI sur images médicales.",
                [
                    "autopet/segmentation_metrics.json",
                    "autopet/comparison.json",
                    "autopet/method_benchmark.json" if autopet_method_benchmark else "",
                    "brain_mri/metrics.json",
                ],
            ),
            _traceability(
                "REQ-C3",
                "Modèle appliqué aux données pour produire des prédictions.",
                [
                    "autopet/segmentation_metrics.json",
                    "brain_mri/metrics.json",
                ],
            ),
            _traceability(
                "REQ-C4",
                "Génération d'explications XAI.",
                [
                    "autopet/xai_analysis_summary.json" if autopet_xai_summary else "",
                    "autopet/method_benchmark.json" if autopet_method_benchmark else "",
                    "brain_mri/xai_method_benchmark.json" if brain_mri_xai_benchmark else "",
                ],
            ),
            _traceability(
                "REQ-C5",
                "Analyse des résultats XAI et limites.",
                [
                    "autopet/comparison.json",
                    "autopet/method_benchmark.json" if autopet_method_benchmark else "",
                    "brain_mri/xai_method_benchmark.json" if brain_mri_xai_benchmark else "",
                ],
            ),
            _traceability(
                "REQ-C6",
                "Comparaison de méthodes XAI selon un protocole commun.",
                [
                    "autopet/method_benchmark.json" if autopet_method_benchmark else "",
                    "brain_mri/xai_method_benchmark.json" if brain_mri_xai_benchmark else "",
                ],
            ),
            _traceability(
                "REQ-U1",
                "Traçabilité des versions et jalons.",
                [
                    "traceability/run_index.json" if run_index else "",
                ],
            ),
            _traceability(
                "REQ-U2",
                "Traçabilité des décisions et artefacts.",
                [
                    "traceability/requirement_traceability.json",
                    "autopet/comparison.json",
                    "autopet/segmentation_metrics.json",
                    "brain_mri/metrics.json",
                ],
            ),
            _traceability(
                "REQ-U3",
                "Support de restitution finale.",
                [
                    "README.md",
                    "autopet/figures/",
                    "brain_mri/figures/",
                ],
            ),
        ]
    }
    # Remove empty evidence strings introduced by optional files.
    for row in traceability_map["requirements"]:
        row["evidence_files"] = [item for item in row["evidence_files"] if item]
        row["status"] = "covered" if row["evidence_files"] else "missing_evidence"

    save_json(traceability_map, traceability_target / "requirement_traceability.json")
    if run_index:
        save_json(run_index, traceability_target / "run_index.json")

    top_autopet_method = None
    if autopet_method_benchmark:
        top_autopet_method = _extract_top_method(autopet_method_benchmark)
    elif autopet_xai_summary:
        top_autopet_method = _extract_top_method(autopet_xai_summary)

    top_brain_method = None
    if brain_mri_xai_benchmark:
        top_brain_method = _extract_top_method(brain_mri_xai_benchmark)
    brain_refresh_gallery_dir = args.results_root / "brain_mri_refresh_xai_20260418"
    selected_brain_benchmark_run_id = args.brain_mri_xai_benchmark_run_id
    selected_brain_benchmark_dir = (
        args.results_root / selected_brain_benchmark_run_id if selected_brain_benchmark_run_id else None
    )

    evidence_manifest = {
        "run_ids": {
            "autopet_main": args.autopet_main_run_id,
            "autopet_comparison": args.autopet_comparison_run_id,
            "brain_mri_backup": args.brain_mri_run_id,
            "autopet_xai_analysis": args.autopet_xai_analysis_run_id or args.autopet_main_run_id,
            "brain_mri_xai_benchmark": args.brain_mri_xai_benchmark_run_id,
        },
        "copied_files": {
            "pack_root": [
                "README.md",
                "INTERPRETATION.md",
                "EVALUATION_ALIGNMENT.md",
                "EVALUATION_READINESS.md",
                "EVALUATION_READINESS.json",
                "DEMO_RUNBOOK.md",
                "evidence_manifest.json",
            ],
            "autopet": [
                "autopet/segmentation_metrics.json",
                "autopet/comparison.json",
                "autopet/xai_analysis_summary.json" if autopet_xai_summary else "",
                "autopet/method_benchmark.json" if autopet_method_benchmark else "",
                *[f"autopet/figures/{name}" for name in copied_autopet_figures],
            ],
            "brain_mri": [
                "brain_mri/metrics.json",
                "brain_mri/xai_method_benchmark.json" if brain_mri_xai_benchmark else "",
                *[f"brain_mri/figures/{name}" for name in copied_brain_figures],
            ],
            "traceability": [
                "traceability/requirement_traceability.json",
                "traceability/run_index.json" if run_index else "",
            ],
        },
    }
    for key, values in evidence_manifest["copied_files"].items():
        evidence_manifest["copied_files"][key] = [value for value in values if value]
    save_json(evidence_manifest, output_dir / "evidence_manifest.json")

    interpretation_lines = [
        "# Interpretation synthétique",
        "",
        "## autoPET FDG (ligne principale)",
        "",
        (
            "- Le snapshot principal confirme une segmentation mesurée par Dice "
            f"`{autopet_metrics.get('mean_dice', 0.0):.4f}` avec FP moyen "
            f"`{autopet_metrics.get('mean_false_positive_volume_ml', 0.0):.4f}` mL et FN moyen "
            f"`{autopet_metrics.get('mean_false_negative_volume_ml', 0.0):.4f}` mL."
        ),
        (
            f"- Méthode XAI prioritaire sur autoPET: `{top_autopet_method}`."
            if top_autopet_method
            else "- Méthode XAI prioritaire sur autoPET: n/a."
        ),
    ]
    interpretation_lines.extend(_autopet_tradeoff_lines(autopet_comparison))
    interpretation_lines.extend(
        [
            "",
            "## Brain MRI (ligne backup)",
            "",
            (
                "- Le backup Brain MRI reste cohérent pour la soutenance: "
                f"accuracy `{brain_mri_metrics.get('accuracy', 0.0):.4f}`, "
                f"F1 `{brain_mri_metrics.get('f1', 0.0):.4f}`, "
                f"ROC-AUC `{brain_mri_metrics.get('roc_auc', 0.0):.4f}`."
            ),
            (
                f"- Méthode XAI prioritaire sur Brain MRI: `{top_brain_method}`."
                if top_brain_method
                else "- Méthode XAI prioritaire sur Brain MRI: n/a."
            ),
            "",
            "## Message projet recommandé",
            "",
            "- Storyline: autoPET FDG est la contribution scientifique principale; Brain MRI confirme la robustesse de la démarche XAI sur un second cadre.",
            "- Le dossier met en évidence le compromis Dice/FN/FP des variantes post-traitées plutôt qu'un unique score isolé.",
            "- Les méthodes XAI sont interprétées comme explications de décision du modèle, pas comme preuve clinique directe.",
        ]
    )
    if brain_refresh_gallery_dir.exists():
        interpretation_lines.append(
            "- Une galerie XAI élargie (16 cas équilibrés) est disponible dans `results/brain_mri_refresh_xai_20260418/`."
        )
    if selected_brain_benchmark_dir is not None and selected_brain_benchmark_dir.exists():
        interpretation_lines.append(
            f"- Le benchmark Brain MRI utilisé dans ce pack est `results/{selected_brain_benchmark_run_id}/`."
        )
    (output_dir / "INTERPRETATION.md").write_text("\n".join(interpretation_lines) + "\n", encoding="utf-8")

    optional_content_lines: List[str] = []
    if brain_refresh_gallery_dir.exists():
        optional_content_lines.append(
            "- `../brain_mri_refresh_xai_20260418/`: expanded qualitative Brain MRI XAI gallery (`16` balanced cases)"
        )
    if selected_brain_benchmark_dir is not None and selected_brain_benchmark_dir.exists():
        optional_content_lines.append(
            f"- `../{selected_brain_benchmark_run_id}/`: selected Brain MRI benchmark used for this evidence pack"
        )
    optional_contents = "\n".join(optional_content_lines)

    readme = f"""# Project evidence pack

This folder consolidates the most important, review-ready artifacts for project evaluation.

- autoPET main run: `{args.autopet_main_run_id}`
- autoPET comparison run: `{args.autopet_comparison_run_id}`
- Brain MRI backup run: `{args.brain_mri_run_id}`
- includes run index: `{"yes" if bool(run_index) else "no"}`

## Key metrics

- autoPET mean Dice: `{autopet_metrics.get('mean_dice', 0.0):.4f}`
- autoPET mean FN volume (mL): `{autopet_metrics.get('mean_false_negative_volume_ml', 0.0):.4f}`
- autoPET mean FP volume (mL): `{autopet_metrics.get('mean_false_positive_volume_ml', 0.0):.4f}`
- Brain MRI accuracy: `{brain_mri_metrics.get('accuracy', 0.0):.4f}`
- Brain MRI F1: `{brain_mri_metrics.get('f1', 0.0):.4f}`
- Brain MRI ROC-AUC: `{brain_mri_metrics.get('roc_auc', 0.0):.4f}`

## XAI benchmark highlights

- autoPET top method: `{top_autopet_method if top_autopet_method else 'n/a'}`
- Brain MRI top method: `{top_brain_method if top_brain_method else 'n/a'}`

## Contents

- `autopet/`: metrics, comparison, optional method benchmark, selected figures
- `brain_mri/`: metrics, optional method benchmark, selected figures
- `traceability/`: requirement traceability map and optional run index snapshot
- `evidence_manifest.json`: explicit inventory of copied evidence files
- `INTERPRETATION.md`: concise interpretation blocks ready for report/slides
- `EVALUATION_ALIGNMENT.md`: rubric-oriented checklist for client/soutenance/plan-projet review
- `EVALUATION_READINESS.md`: scored readiness report aligned with official evaluation grids
- `DEMO_RUNBOOK.md`: deterministic 2-3 minute demo flow aligned with `REQ-C2/C4/C5`
{optional_contents}

## Figure counts

- autoPET copied figures: `{len(copied_autopet_figures)}`
- Brain MRI copied figures: `{len(copied_brain_figures)}`
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    (output_dir / "EVALUATION_ALIGNMENT.md").write_text(
        _build_evaluation_alignment(output_dir),
        encoding="utf-8",
    )
    (output_dir / "DEMO_RUNBOOK.md").write_text(
        _build_demo_runbook(
            autopet_main_run_id=args.autopet_main_run_id,
            autopet_comparison_run_id=args.autopet_comparison_run_id,
            brain_mri_run_id=args.brain_mri_run_id,
            autopet_top_method=top_autopet_method,
            brain_top_method=top_brain_method,
        ),
        encoding="utf-8",
    )

    readiness_mapping_path: Optional[Path] = None
    try:
        readiness_mapping_path = resolve_mapping_path(args.evaluation_mapping_path)
        readiness_mapping = load_readiness_mapping(readiness_mapping_path)
        readiness_report = audit_evidence_pack(output_dir, readiness_mapping)
        save_json(readiness_report, output_dir / "EVALUATION_READINESS.json")
        (output_dir / "EVALUATION_READINESS.md").write_text(
            render_readiness_markdown(readiness_report) + "\n",
            encoding="utf-8",
        )
        evidence_manifest["evaluation_readiness"] = {
            "status": readiness_report.get("overall", {}).get("status", "missing"),
            "coverage_score": readiness_report.get("overall", {}).get("coverage_score", 0.0),
            "mapping_path": str(readiness_mapping_path),
        }
    except FileNotFoundError:
        evidence_manifest["evaluation_readiness"] = {
            "status": "mapping_not_found",
            "coverage_score": 0.0,
            "mapping_path": str(args.evaluation_mapping_path),
        }

    save_json(evidence_manifest, output_dir / "evidence_manifest.json")


if __name__ == "__main__":
    main()
