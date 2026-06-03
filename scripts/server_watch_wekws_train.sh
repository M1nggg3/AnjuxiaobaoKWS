#!/usr/bin/env bash
set -euo pipefail

if [[ -f "${HOME}/.anju_xiaobao_kws_env" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/.anju_xiaobao_kws_env"
fi

: "${ANJU_RUN_ROOT:=${HOME}/runs/AnJuXiaoBaoKWS}"
: "${PYTHON_BIN:=${HOME}/miniforge3/envs/cosyvoice310/bin/python}"

RUN_DIR="${1:-}"
if [[ -z "${RUN_DIR}" ]]; then
  RUN_DIR="$(ls -td "${ANJU_RUN_ROOT}"/wekws_train_farfield_main_* 2>/dev/null | head -1 || true)"
fi

if [[ -z "${RUN_DIR}" ]] || [[ ! -d "${RUN_DIR}" ]]; then
  echo "No wekws_train_farfield_main run directory found."
  exit 1
fi

META_FILE="${RUN_DIR}/run_meta.json"
STDOUT_LOG="${RUN_DIR}/train.out.log"
STDERR_LOG="${RUN_DIR}/train.err.log"
EXP_DIR=""
PREPARED_DIR=""
if [[ -f "${META_FILE}" ]]; then
  read -r EXP_DIR PREPARED_DIR < <("${PYTHON_BIN}" - "${META_FILE}" <<'PY'
import json
import sys
from pathlib import Path
meta = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(meta.get("experiment_dir", ""), meta.get("prepared_dir", ""))
PY
)
fi

echo "Run directory: ${RUN_DIR}"
echo "Experiment: ${EXP_DIR:-unknown}"
echo "Prepared data: ${PREPARED_DIR:-unknown}"
echo

if [[ -n "${EXP_DIR}" ]] && [[ -d "${EXP_DIR}" ]]; then
  echo "Checkpoints:"
  find "${EXP_DIR}" -maxdepth 1 -type f \( -name '*.pt' -o -name 'final.pt' \) -printf '%f %s bytes %TY-%Tm-%Td %TH:%TM:%TS\n' | sort -V | tail -10 || true
fi

if [[ -n "${PREPARED_DIR}" ]] && [[ -f "${PREPARED_DIR}/summary.json" ]]; then
  echo
  echo "Prepared summary:"
  "${PYTHON_BIN}" - "${PREPARED_DIR}/summary.json" <<'PY'
import json
import sys
from pathlib import Path
s = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(json.dumps({
    "balanced": s.get("balanced"),
    "splits": s.get("splits"),
}, ensure_ascii=False, indent=2))
PY
fi

echo
echo "stdout tail:"
tail -n 30 "${STDOUT_LOG}" 2>/dev/null || true

echo
echo "stderr tail:"
tail -n 30 "${STDERR_LOG}" 2>/dev/null || true
