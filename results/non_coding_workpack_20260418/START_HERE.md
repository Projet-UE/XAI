# Non-Coding Workpack (Project Handoff)

This folder contains the project outputs that are most useful for report writing, slides, and oral defense.
It excludes training code and heavy artifacts.

## 1) If you only have 10 minutes

Open these files first:

1. `KEY_NUMBERS.md`
2. `autopet/figures/`
3. `brain_mri/figures/`
4. `source_references/INTERPRETATION.md`

## 2) What each subfolder is for

- `autopet/`
  - Segmentation metrics, post-processing comparison, single-method tracked XAI analysis, autoPET figures.
- `brain_mri/`
  - Classification metrics, XAI benchmark, Brain MRI figures, extra qualitative XAI examples.
- `tables/`
  - Ready-to-paste CSV tables for report/slides.
- `source_references/`
  - Traceability/readiness material copied from the evidence pack.

## 3) Minimal narrative to reuse in report

- Primary line: **autoPET FDG segmentation + XAI**.
- Backup line: **Brain MRI classification + XAI**.
- autoPET result was consolidated around the 50-epoch checkpoint and post-processed variants.
- autoPET XAI is report-ready but currently consolidated on `integrated_gradients` only.
- Brain MRI provides a stable classification baseline with interpretable XAI panels.
- Brain MRI carries the full multi-method XAI benchmark used for method comparison.

## 4) Suggested figure usage

- autoPET:
  - Use `autopet/figures/integrated_gradients.png` plus 2-3 case-specific panels.
- Brain MRI:
  - Use `brain_mri/figures/confusion_matrix.png`
  - Use `brain_mri/figures/metrics_overview.png`
  - Use `brain_mri/figures/gradcam.png`, `integrated_gradients.png`, `occlusion.png`
  - Optionally add examples from `brain_mri/extra_xai_examples/` for qualitative discussion.

## 5) Scope note

This workpack is a curated snapshot for non-coding use.  
Source of truth for full traceability remains:

- `results/evidence_pack_20260418_grid/`
