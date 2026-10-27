#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
VOC_DATA_ROOT="${VOC_DATA_ROOT:-${PROJECT_ROOT}/data}"
SMOKE_SAMPLES="${SMOKE_SAMPLES:-16}"
SMOKE_EPOCHS="${SMOKE_EPOCHS:-1}"
WORK_DIR="${WORK_DIR:-${SCRIPT_DIR}/runs/smoke}"

export VOC_DATA_ROOT

"${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/check_environment.py" --require-cuda
"${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/check_baseline.py" \
  --data-root "${VOC_DATA_ROOT}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/train_baseline.py" \
  --data-root "${VOC_DATA_ROOT}" \
  --num-samples "${SMOKE_SAMPLES}" \
  --epochs "${SMOKE_EPOCHS}" \
  --skip-val \
  --work-dir "${WORK_DIR}"

echo "Smoke test completed. Checkpoint directory: ${WORK_DIR}"
