#!/usr/bin/env bash
set -euo pipefail

# Clean/filter CosyVoice3 pure wake-word TTS and mix it with RK3566 office noise.

if [[ -f "${HOME}/.anju_xiaobao_kws_env" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/.anju_xiaobao_kws_env"
fi

: "${ANJU_PROJECT:=${HOME}/projects/AnJuXiaoBaoKWS}"
: "${ANJU_DATA_ROOT:=${HOME}/datasets/AnJuXiaoBaoKWS/data}"
: "${ANJU_BASE_DATASET:=${ANJU_DATA_ROOT}/anju_xiaobao_kws_dataset_20260508}"
: "${ANJU_RUN_ROOT:=${HOME}/runs/AnJuXiaoBaoKWS}"
: "${PYTHON_BIN:=${HOME}/miniforge3/envs/cosyvoice310/bin/python}"

INPUT_CLEAN_DIR="${1:-${ANJU_DATA_ROOT}/anju_xiaobao_cosyvoice3_clean_purewake_20260520_155613}"
OUTPUT_DATASET="${2:-${ANJU_DATA_ROOT}/anju_xiaobao_cosyvoice3_cleaned_farfield_20260521}"
NOISE_MANIFEST="${ANJU_BASE_DATASET}/manifests/real_office_noise_segments.jsonl"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="mix_cosyvoice3_farfield_${RUN_STAMP}"
RUN_DIR="${ANJU_RUN_ROOT}/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

COMMAND_FILE="${RUN_DIR}/command.sh"
STDOUT_LOG="${RUN_DIR}/stdout.log"
STDERR_LOG="${RUN_DIR}/stderr.log"
META_FILE="${RUN_DIR}/run_meta.json"

if [[ -e "${OUTPUT_DATASET}" ]] && [[ -n "$(find "${OUTPUT_DATASET}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refuse to overwrite non-empty output dataset: ${OUTPUT_DATASET}" >&2
  exit 1
fi

cat > "${COMMAND_FILE}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source "${HOME}/.anju_xiaobao_kws_env"
cd "${ANJU_PROJECT}"
"${PYTHON_BIN}" -m anju_kws.data.clean_and_mix_cosyvoice3_clean \\
  --input_clean_dir "${INPUT_CLEAN_DIR}" \\
  --noise_dataset_root "${ANJU_BASE_DATASET}" \\
  --noise_manifest "${NOISE_MANIFEST}" \\
  --output_dir "${OUTPUT_DATASET}" \\
  --min_positive_sec 0.8 \\
  --min_negative_sec 0.8 \\
  --max_positive_sec 4.5 \\
  --max_negative_sec 8.0 \\
  --profiles near_0p5m,mid_1m,far_2m \\
  --seed 20260521 \\
  --copy_clean
EOF
chmod +x "${COMMAND_FILE}"

cat > "${META_FILE}" <<EOF
{
  "run_name": "${RUN_NAME}",
  "run_dir": "${RUN_DIR}",
  "project": "${ANJU_PROJECT}",
  "input_clean_dir": "${INPUT_CLEAN_DIR}",
  "base_dataset": "${ANJU_BASE_DATASET}",
  "noise_manifest": "${NOISE_MANIFEST}",
  "output_dataset": "${OUTPUT_DATASET}",
  "stdout_log": "${STDOUT_LOG}",
  "stderr_log": "${STDERR_LOG}",
  "python_bin": "${PYTHON_BIN}",
  "min_positive_sec": 0.8,
  "min_negative_sec": 0.8,
  "max_positive_sec": 4.5,
  "max_negative_sec": 8.0,
  "profiles": ["near_0p5m", "mid_1m", "far_2m"],
  "snr_per_profile": 3,
  "expected_farfield_multiplier": 9,
  "created_at": "$(date --iso-8601=seconds)"
}
EOF

echo "Run directory: ${RUN_DIR}"
echo "Input clean dataset: ${INPUT_CLEAN_DIR}"
echo "Output farfield dataset: ${OUTPUT_DATASET}"
echo "Command file: ${COMMAND_FILE}"
echo "Logs: ${STDOUT_LOG}, ${STDERR_LOG}"
echo
echo "Starting CosyVoice3 clean filtering and RK3566 farfield noise mixing..."

bash "${COMMAND_FILE}" > "${STDOUT_LOG}" 2> "${STDERR_LOG}"

echo "CosyVoice3 farfield mixing finished."
echo "Inspect logs with:"
echo "  tail -n 50 ${STDOUT_LOG}"
echo "  tail -n 50 ${STDERR_LOG}"
