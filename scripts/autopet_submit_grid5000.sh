#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "" ]]; then
  cat <<'EOF'
Usage:
  scripts/autopet_submit_grid5000.sh <mode> [split] [trainer] [walltime] [cluster]

Examples:
  scripts/autopet_submit_grid5000.sh train fdg_full nnUNetTrainer_10epochs 04:00:00 kinovis
  scripts/autopet_submit_grid5000.sh predict fdg_full nnUNetTrainer_10epochs 02:00:00 kinovis
  scripts/autopet_submit_grid5000.sh xai fdg_full nnUNetTrainer_10epochs 02:00:00 kinovis

Notes:
  - Run this from a Grenoble frontend session.
  - The script submits a single-GPU besteffort OAR job.
  - It assumes the repo lives in $HOME/XAI and the FDG artifacts in $HOME/XAI/artifacts/autopet_fdg_poc.
EOF
  exit 1
fi

MODE="${1}"
SPLIT="${2:-fdg_full}"
TRAINER="${3:-nnUNetTrainer_10epochs}"
WALLTIME="${4:-04:00:00}"
CLUSTER="${5:-kinovis}"

REPO_ROOT="${HOME}/XAI"
ARTIFACTS_DIR="${REPO_ROOT}/artifacts/autopet_fdg_poc"
DATASET_ID="${DATASET_ID:-501}"
CONFIGURATION="${CONFIGURATION:-3d_fullres}"
FOLD="${FOLD:-0}"
DEVICE="${DEVICE:-cuda}"

COMMON_ENV="cd ${REPO_ROOT} && source .venv/bin/activate && export nnUNet_raw=${ARTIFACTS_DIR}/${SPLIT}/nnunet_raw && export nnUNet_preprocessed=${ARTIFACTS_DIR}/${SPLIT}/nnunet_preprocessed && export nnUNet_results=${ARTIFACTS_DIR}/${SPLIT}/nnunet_results"

case "${MODE}" in
  train)
    PAYLOAD="${COMMON_ENV} && python scripts/autopet_train_nnunet.py --artifacts-dir ${ARTIFACTS_DIR} --split-name ${SPLIT} --dataset-id ${DATASET_ID} --configuration ${CONFIGURATION} --fold ${FOLD} --trainer ${TRAINER} --device ${DEVICE}"
    ;;
  predict)
    PAYLOAD="${COMMON_ENV} && python scripts/autopet_predict_nnunet.py --artifacts-dir ${ARTIFACTS_DIR} --split-name ${SPLIT} --dataset-id ${DATASET_ID} --configuration ${CONFIGURATION} --fold ${FOLD} --trainer ${TRAINER} --device ${DEVICE}"
    ;;
  xai)
    PAYLOAD="${COMMON_ENV} && python scripts/autopet_generate_xai.py --artifacts-dir ${ARTIFACTS_DIR} --split-name ${SPLIT} --dataset-id ${DATASET_ID} --configuration ${CONFIGURATION} --fold ${FOLD} --trainer ${TRAINER} --device ${DEVICE} --max-cases 4"
    ;;
  *)
    echo "Unsupported mode: ${MODE}" >&2
    exit 2
    ;;
esac

echo "Submitting ${MODE} job on Grenoble:"
echo "  split      = ${SPLIT}"
echo "  trainer    = ${TRAINER}"
echo "  walltime   = ${WALLTIME}"
echo "  cluster    = ${CLUSTER}"
echo "  device     = ${DEVICE}"

oarsub \
  --project irit \
  -q besteffort \
  -d "${HOME}" \
  -l "gpu=1,walltime=${WALLTIME}" \
  -p "cluster='${CLUSTER}'" \
  "${PAYLOAD}"
