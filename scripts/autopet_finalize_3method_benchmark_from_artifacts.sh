#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
if [[ ! -d "${VENV_DIR}" && -d "${HOME}/XAI/.venv" ]]; then
  VENV_DIR="${HOME}/XAI/.venv"
fi

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  echo "Virtual environment not found: ${VENV_DIR}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
source "${VENV_DIR}/bin/activate"

ARTIFACTS_DIR="${ARTIFACTS_DIR:-${REPO_ROOT}/artifacts/autopet_fdg_poc}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results}"
SPLIT_NAME="${SPLIT_NAME:-fdg_full}"
DATASET_ID="${DATASET_ID:-501}"
TRAINER="${TRAINER:-nnUNetTrainer_50epochs}"
CONFIGURATION="${CONFIGURATION:-3d_fullres}"
FOLD="${FOLD:-0}"
DEVICE="${DEVICE:-cuda}"
SWEEP_ID="${SWEEP_ID:-fdg_full_50epochs_postprocess_mean_pet_20260419_rebuild}"
RUN_ID="${RUN_ID:-autopet_fdg_full_rebuild_best_label_50epochs_xai_3methods_20260419}"
STATE_NAME="${STATE_NAME:-rebuild_best_label_50epochs_xai_3methods_20260419}"
SNAPSHOT_TITLE="${SNAPSHOT_TITLE:-autoPET FDG rebuilt 50-epoch XAI snapshot (3 methods)}"
ANALYSIS_TITLE="${ANALYSIS_TITLE:-autoPET FDG rebuilt 50-epoch XAI analysis (3 methods)}"
XAI_DIR="${XAI_DIR:-${ARTIFACTS_DIR}/${SPLIT_NAME}/xai_rebuild_best_label_50epochs_3methods}"
EXPORT_CONFIG_PATH="${EXPORT_CONFIG_PATH:-${ARTIFACTS_DIR}/${SPLIT_NAME}/export_configs/rebuild_best_label_50epochs_xai_3methods_run_config.json}"

SWEEP_ROOT="${ARTIFACTS_DIR}/${SPLIT_NAME}/postprocess_sweeps/${SWEEP_ID}"
SWEEP_SUMMARY="${SWEEP_ROOT}/summary.json"
TRAINING_RUN_CONFIG="${ARTIFACTS_DIR}/${SPLIT_NAME}/training_run_config.json"
export nnUNet_raw="${ARTIFACTS_DIR}/${SPLIT_NAME}/nnunet_raw"
export nnUNet_preprocessed="${ARTIFACTS_DIR}/${SPLIT_NAME}/nnunet_preprocessed"
export nnUNet_results="${ARTIFACTS_DIR}/${SPLIT_NAME}/nnunet_results"

if [[ ! -f "${SWEEP_SUMMARY}" ]]; then
  echo "Missing sweep summary: ${SWEEP_SUMMARY}" >&2
  exit 1
fi

if [[ ! -f "${TRAINING_RUN_CONFIG}" ]]; then
  echo "Missing training run config: ${TRAINING_RUN_CONFIG}" >&2
  exit 1
fi

BEST_LABEL="$(
  SWEEP_SUMMARY="${SWEEP_SUMMARY}" python - <<'PY'
import json
import os
from pathlib import Path

summary = json.loads(Path(os.environ["SWEEP_SUMMARY"]).read_text(encoding="utf-8"))
print(summary["ranking"][0]["label"])
PY
)"

BEST_PRED="${SWEEP_ROOT}/predictions/${BEST_LABEL}"
BEST_METRICS="${SWEEP_ROOT}/metrics/${BEST_LABEL}.json"

if [[ ! -d "${BEST_PRED}" ]]; then
  echo "Missing best prediction directory: ${BEST_PRED}" >&2
  exit 1
fi
if [[ ! -f "${BEST_METRICS}" ]]; then
  echo "Missing best metrics file: ${BEST_METRICS}" >&2
  exit 1
fi

echo "[finalize] using rebuilt best label ${BEST_LABEL}"

echo "[xai] generating 3-method qualitative exports"
python scripts/autopet_generate_xai.py \
  --artifacts-dir "${ARTIFACTS_DIR}" \
  --split-name "${SPLIT_NAME}" \
  --dataset-id "${DATASET_ID}" \
  --configuration "${CONFIGURATION}" \
  --fold "${FOLD}" \
  --trainer "${TRAINER}" \
  --device "${DEVICE}" \
  --prediction-dir "${BEST_PRED}" \
  --output-dir "${XAI_DIR}" \
  --methods saliency integrated_gradients occlusion \
  --max-cases 8

echo "[xai] validating generated review cases and run config"
EXPECTED_METHODS="integrated_gradients occlusion saliency" \
REVIEW_CASES_PATH="${XAI_DIR}/review_cases.json" \
XAI_RUN_CONFIG_PATH="${XAI_DIR}/xai_run_config.json" \
python - <<'PY'
import json
import os
from pathlib import Path

expected_methods = sorted(os.environ["EXPECTED_METHODS"].split())
review_cases_path = Path(os.environ["REVIEW_CASES_PATH"])
run_config_path = Path(os.environ["XAI_RUN_CONFIG_PATH"])

review_cases = json.loads(review_cases_path.read_text(encoding="utf-8"))
run_config = json.loads(run_config_path.read_text(encoding="utf-8"))

selection = review_cases.get("selection", {})
selected_case_ids = (
    review_cases.get("selected_case_ids")
    or selection.get("selected_case_ids")
    or run_config.get("selected_case_ids")
    or [case.get("case_id") for case in review_cases.get("cases", []) if case.get("case_id")]
)
if len(selected_case_ids) != 7:
    raise SystemExit(f"Expected 7 frozen review cases, got {len(selected_case_ids)}")

config_methods = sorted(run_config.get("methods", []))
if config_methods != expected_methods:
    raise SystemExit(f"Unexpected XAI methods in xai_run_config.json: {config_methods}")

for case in review_cases.get("cases", []):
    case_methods = sorted(
        method_entry.get("method")
        for method_entry in case.get("methods", [])
        if method_entry.get("method")
    )
    if case_methods != expected_methods:
        raise SystemExit(
            f"Case {case.get('case_id')} has unexpected methods: {case_methods}"
        )

print("xai_methods_ok")
PY

echo "[analyze] generating narrative XAI summary"
python scripts/autopet_analyze_xai.py \
  --review-cases-path "${XAI_DIR}/review_cases.json" \
  --metrics-path "${BEST_METRICS}" \
  --output-dir "${XAI_DIR}" \
  --state-name "${STATE_NAME}" \
  --title "${ANALYSIS_TITLE}"

echo "[compare] building protocol benchmark"
python scripts/autopet_compare_xai_methods.py \
  --review-cases-path "${XAI_DIR}/review_cases.json" \
  --metrics-path "${BEST_METRICS}" \
  --output-dir "${XAI_DIR}" \
  --state-name "${STATE_NAME}"

echo "[compare] validating 3-method benchmark output"
EXPECTED_METHODS="integrated_gradients occlusion saliency" \
BENCHMARK_PATH="${XAI_DIR}/method_benchmark.json" \
python - <<'PY'
import json
import os
from pathlib import Path

expected_methods = sorted(os.environ["EXPECTED_METHODS"].split())
benchmark_path = Path(os.environ["BENCHMARK_PATH"])
benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))

ranking = benchmark.get("ranking", [])
ranking_methods = sorted(item.get("method") for item in ranking if item.get("method"))
if len(ranking) != 3 or ranking_methods != expected_methods:
    raise SystemExit(
        f"Benchmark ranking is not the expected 3-method set: {ranking_methods}"
    )

method_summaries = benchmark.get("method_summaries", {})
summary_methods = sorted(method_summaries.keys())
if summary_methods != expected_methods:
    raise SystemExit(
        f"Benchmark method summaries do not match expected methods: {summary_methods}"
    )

pairwise = benchmark.get("paired_delta_ci", {})
required_pair_count = 3
for key, rows in pairwise.items():
    if rows and len(rows) != required_pair_count:
        raise SystemExit(
            f"Benchmark pairwise section {key} has {len(rows)} rows; expected {required_pair_count}"
        )

print("benchmark_methods_ok")
PY

echo "[export] freezing tracked autoPET rebuilt 3-method benchmark snapshot"
mkdir -p "$(dirname "${EXPORT_CONFIG_PATH}")"
TRAINING_RUN_CONFIG="${TRAINING_RUN_CONFIG}" \
EXPORT_CONFIG_PATH="${EXPORT_CONFIG_PATH}" \
SWEEP_SUMMARY="${SWEEP_SUMMARY}" \
BEST_PRED="${BEST_PRED}" \
BEST_METRICS="${BEST_METRICS}" \
BEST_LABEL="${BEST_LABEL}" \
SNAPSHOT_TITLE="${SNAPSHOT_TITLE}" \
STATE_NAME="${STATE_NAME}" \
python - <<'PY'
import json
import os
from pathlib import Path

training = json.loads(Path(os.environ["TRAINING_RUN_CONFIG"]).read_text(encoding="utf-8"))
summary = json.loads(Path(os.environ["SWEEP_SUMMARY"]).read_text(encoding="utf-8"))
best_label = os.environ["BEST_LABEL"]
best_entry = next(item for item in summary["ranking"] if item["label"] == best_label)

training.update(
    {
        "state_name": os.environ["STATE_NAME"],
        "snapshot_title": os.environ["SNAPSHOT_TITLE"],
        "prediction_dir": os.environ["BEST_PRED"],
        "metrics_path": os.environ["BEST_METRICS"],
        "xai_methods": ["saliency", "integrated_gradients", "occlusion"],
        "postprocess_label": best_label,
        "postprocess": {
            "rank_by": best_entry.get("rank_by"),
            "min_component_volume_ml": best_entry.get("min_component_volume_ml"),
            "max_components": best_entry.get("max_components"),
        },
        "rebuild_note": (
            "This XAI benchmark was produced from the reproducible rebuilt 50-epoch FDG state "
            "because the original heavy Grenoble artifacts were no longer available."
        ),
    }
)
export_path = Path(os.environ["EXPORT_CONFIG_PATH"])
export_path.write_text(json.dumps(training, indent=2) + "\n", encoding="utf-8")
PY

python scripts/autopet_export_results.py \
  --artifacts-dir "${ARTIFACTS_DIR}" \
  --split-name "${SPLIT_NAME}" \
  --results-root "${RESULTS_ROOT}" \
  --run-id "${RUN_ID}" \
  --snapshot-title "${SNAPSHOT_TITLE}" \
  --metrics-path "${BEST_METRICS}" \
  --review-cases-path "${XAI_DIR}/review_cases.json" \
  --xai-dir "${XAI_DIR}" \
  --run-config-path "${EXPORT_CONFIG_PATH}" \
  --analysis-summary-path "${XAI_DIR}/xai_analysis_summary.json" \
  --method-benchmark-path "${XAI_DIR}/method_benchmark.json" \
  --method-benchmark-md-path "${XAI_DIR}/method_benchmark.md" \
  --require-review-cases \
  --require-xai-dir \
  --require-analysis-summary \
  --require-method-benchmark \
  --require-protocol-benchmark \
  --max-figures 7

echo "[done] ${RUN_ID}"
