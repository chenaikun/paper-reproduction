#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
VOC_DATA_ROOT="${VOC_DATA_ROOT:-${PROJECT_ROOT}/data/VOCdevkit}"

export VOC_DATA_ROOT
export LD_PROFILE="${LD_PROFILE:-auto}"
export WORKERS_PER_GPU="${WORKERS_PER_GPU:-0}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/check_environment.py" --require-cuda
"${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/check_dataset.py" \
  --data-root "${VOC_DATA_ROOT}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/check_baseline.py" \
  --data-root "${VOC_DATA_ROOT}"

if [[ -n "${TEACHER_CHECKPOINT:-}" ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/check_baseline.py" \
    --config "${SCRIPT_DIR}/configs/ld_r18_voc_full.py" \
    --data-root "${VOC_DATA_ROOT}" \
    --teacher-checkpoint "${TEACHER_CHECKPOINT}"
else
  echo "LD step skipped: set TEACHER_CHECKPOINT after training the R101 teacher."
fi

echo "Smoke test completed."
