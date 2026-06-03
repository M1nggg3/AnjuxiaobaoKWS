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
  RUN_DIR="$(ls -td "${ANJU_RUN_ROOT}"/mix_cosyvoice3_farfield_* 2>/dev/null | head -1 || true)"
fi

if [[ -z "${RUN_DIR}" ]] || [[ ! -d "${RUN_DIR}" ]]; then
  echo "No mix_cosyvoice3_farfield run directory found."
  exit 1
fi

META_FILE="${RUN_DIR}/run_meta.json"
STDOUT_LOG="${RUN_DIR}/stdout.log"
STDERR_LOG="${RUN_DIR}/stderr.log"
OUTPUT_DATASET=""
if [[ -f "${META_FILE}" ]]; then
  OUTPUT_DATASET="$("${PYTHON_BIN}" - "${META_FILE}" <<'PY'
import json
import sys
from pathlib import Path
meta = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(meta.get("output_dataset", ""))
PY
)"
fi

echo "Run directory: ${RUN_DIR}"
echo "Output dataset: ${OUTPUT_DATASET:-unknown}"
echo

if [[ -n "${OUTPUT_DATASET}" ]] && [[ -d "${OUTPUT_DATASET}" ]]; then
  for dir_name in clean_positive clean_negative farfield_positive farfield_negative; do
    count=0
    if [[ -d "${OUTPUT_DATASET}/${dir_name}" ]]; then
      count="$(find "${OUTPUT_DATASET}/${dir_name}" -type f -name '*.wav' | wc -l)"
    fi
    printf "%-20s %s\n" "${dir_name}:" "${count}"
  done

  if [[ -f "${OUTPUT_DATASET}/summary.json" ]]; then
    echo
    echo "summary.json exists: task finished."
    "${PYTHON_BIN}" - "${OUTPUT_DATASET}/summary.json" <<'PY'
import json
import sys
from pathlib import Path
s = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for key in [
    "input_clean_count",
    "kept_clean_count",
    "rejected_clean_count",
    "clean_positive_count",
    "clean_negative_count",
    "farfield_positive_count",
    "farfield_negative_count",
    "farfield_total_count",
]:
    print(f"{key}: {s.get(key)}")
PY
  else
    echo
    echo "summary.json missing: task still running or interrupted before finalization."
  fi
fi

echo
echo "stdout tail:"
tail -n 20 "${STDOUT_LOG}" 2>/dev/null || true

echo
echo "stderr tail:"
tail -n 20 "${STDERR_LOG}" 2>/dev/null || true
