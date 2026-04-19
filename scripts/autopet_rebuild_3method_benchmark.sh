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
SOURCE_ROOT="${SOURCE_ROOT:-${HOME}/data/autopet-fdg/prepared}"
PREPARED_ROOT="${PREPARED_ROOT:-${HOME}/data/autopet-fdg/prepared_rebuild_20260419}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results}"
DATASET_ID="${DATASET_ID:-501}"
SPLIT_NAME="${SPLIT_NAME:-fdg_full}"
TRAINER="${TRAINER:-nnUNetTrainer_50epochs}"
CONFIGURATION="${CONFIGURATION:-3d_fullres}"
FOLD="${FOLD:-0}"
DEVICE="${DEVICE:-cuda}"
SWEEP_ID="${SWEEP_ID:-fdg_full_50epochs_postprocess_mean_pet_20260419_rebuild}"
EXPECTED_BEST_LABEL="${EXPECTED_BEST_LABEL:-rank-mean_pet__minml-5p0__max-1}"
REQUIRE_EXPECTED_BEST_LABEL="${REQUIRE_EXPECTED_BEST_LABEL:-1}"
RUN_ID="${RUN_ID:-autopet_fdg_full_post_best_dice_50epochs_xai_3methods_20260419}"
STATE_NAME="${STATE_NAME:-post_best_dice_50epochs_xai_3methods_20260419}"
SNAPSHOT_TITLE="${SNAPSHOT_TITLE:-autoPET FDG XAI snapshot (best Dice, 3 methods)}"
XAI_DIR="${XAI_DIR:-${ARTIFACTS_DIR}/${SPLIT_NAME}/xai_post_best_dice_50epochs_3methods}"
EXPORT_CONFIG_PATH="${EXPORT_CONFIG_PATH:-${ARTIFACTS_DIR}/${SPLIT_NAME}/export_configs/post_best_dice_50epochs_xai_3methods_run_config.json}"
ANALYSIS_TITLE="${ANALYSIS_TITLE:-autoPET FDG XAI analysis (best Dice, 3 methods)}"

echo "[prepare] rebuilding manifests and nnUNet raw dataset"
python scripts/autopet_prepare_fdg.py \
  --source-root "${SOURCE_ROOT}" \
  --prepared-root "${PREPARED_ROOT}" \
  --artifacts-dir "${ARTIFACTS_DIR}" \
  --dataset-id "${DATASET_ID}" \
  --seed 42 \
  --train-count 48 \
  --val-count 8 \
  --review-count 8 \
  --link-mode symlink

echo "[prepare] verifying frozen fdg_full review IDs"
EXPECTED_REVIEW_IDS="PETCT_05bed31780 PETCT_3bce0eb7aa PETCT_402c061122 PETCT_4848bebb10 PETCT_a1db71e797 PETCT_be3e55a32f PETCT_e2309b8f92" \
SPLIT_PATH="${ARTIFACTS_DIR}/${SPLIT_NAME}/splits/${SPLIT_NAME}.json" \
python - <<'PY'
import json
import os
from pathlib import Path

expected = os.environ["EXPECTED_REVIEW_IDS"].split()
split_path = Path(os.environ["SPLIT_PATH"])
actual = json.loads(split_path.read_text(encoding="utf-8"))["review"]["case_ids"]
if actual != expected:
    raise SystemExit(f"Unexpected review IDs in {split_path}: {actual}")
print("review_ids_ok")
PY

echo "[train] planning, preprocessing, and training ${TRAINER}"
python scripts/autopet_train_nnunet.py \
  --artifacts-dir "${ARTIFACTS_DIR}" \
  --split-name "${SPLIT_NAME}" \
  --dataset-id "${DATASET_ID}" \
  --configuration "${CONFIGURATION}" \
  --fold "${FOLD}" \
  --trainer "${TRAINER}" \
  --device "${DEVICE}"

echo "[predict] generating raw review predictions"
python scripts/autopet_predict_nnunet.py \
  --artifacts-dir "${ARTIFACTS_DIR}" \
  --split-name "${SPLIT_NAME}" \
  --dataset-id "${DATASET_ID}" \
  --configuration "${CONFIGURATION}" \
  --fold "${FOLD}" \
  --trainer "${TRAINER}" \
  --device "${DEVICE}"

echo "[sweep] evaluating lightweight post-processing variants"
python scripts/autopet_sweep_postprocess.py \
  --artifacts-dir "${ARTIFACTS_DIR}" \
  --split-name "${SPLIT_NAME}" \
  --sweep-id "${SWEEP_ID}" \
  --rank-by mean_pet \
  --min-component-ml 0 5 10 20 30 50 \
  --max-components 0 1 2 3

BEST_LABEL="$(
  SWEEP_SUMMARY="${ARTIFACTS_DIR}/${SPLIT_NAME}/postprocess_sweeps/${SWEEP_ID}/summary.json" \
  EXPECTED_BEST_LABEL="${EXPECTED_BEST_LABEL}" \
  REQUIRE_EXPECTED_BEST_LABEL="${REQUIRE_EXPECTED_BEST_LABEL}" \
  python - <<'PY'
import json
import os
from pathlib import Path

summary = json.loads(Path(os.environ["SWEEP_SUMMARY"]).read_text(encoding="utf-8"))
best_label = summary["ranking"][0]["label"]
expected = os.environ["EXPECTED_BEST_LABEL"]
require_expected = os.environ["REQUIRE_EXPECTED_BEST_LABEL"] == "1"
if require_expected and best_label != expected:
    raise SystemExit(f"Expected top label {expected}, got {best_label}")
print(best_label)
PY
)"

SWEEP_ROOT="${ARTIFACTS_DIR}/${SPLIT_NAME}/postprocess_sweeps/${SWEEP_ID}"
BEST_PRED="${SWEEP_ROOT}/predictions/${BEST_LABEL}"
BEST_METRICS="${SWEEP_ROOT}/metrics/${BEST_LABEL}.json"

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

selected_case_ids = review_cases.get("selected_case_ids", [])
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

echo "[export] freezing tracked autoPET 3-method benchmark snapshot"
mkdir -p "$(dirname "${EXPORT_CONFIG_PATH}")"
TRAINING_RUN_CONFIG="${ARTIFACTS_DIR}/${SPLIT_NAME}/training_run_config.json" \
EXPORT_CONFIG_PATH="${EXPORT_CONFIG_PATH}" \
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
training.update(
    {
        "state_name": os.environ["STATE_NAME"],
        "snapshot_title": os.environ["SNAPSHOT_TITLE"],
        "prediction_dir": os.environ["BEST_PRED"],
        "metrics_path": os.environ["BEST_METRICS"],
        "xai_methods": ["saliency", "integrated_gradients", "occlusion"],
        "postprocess_label": os.environ["BEST_LABEL"],
        "postprocess": {
            "rank_by": "mean_pet",
            "min_component_volume_ml": 5.0,
            "max_components": 1,
        },
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
