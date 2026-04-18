#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


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
    return parser.parse_args()


def _copy_files(paths: Iterable[Path], destination_root: Path) -> List[str]:
    copied: List[str] = []
    for source in paths:
        if not source.exists() or not source.is_file():
            continue
        target = destination_root / source.name
        ensure_dir(target.parent)
        shutil.copy2(source, target)
        copied.append(target.name)
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

    evidence_manifest = {
        "run_ids": {
            "autopet_main": args.autopet_main_run_id,
            "autopet_comparison": args.autopet_comparison_run_id,
            "brain_mri_backup": args.brain_mri_run_id,
            "autopet_xai_analysis": args.autopet_xai_analysis_run_id or args.autopet_main_run_id,
            "brain_mri_xai_benchmark": args.brain_mri_xai_benchmark_run_id,
        },
        "copied_files": {
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
    (output_dir / "INTERPRETATION.md").write_text("\n".join(interpretation_lines) + "\n", encoding="utf-8")

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

## Figure counts

- autoPET copied figures: `{len(copied_autopet_figures)}`
- Brain MRI copied figures: `{len(copied_brain_figures)}`
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
