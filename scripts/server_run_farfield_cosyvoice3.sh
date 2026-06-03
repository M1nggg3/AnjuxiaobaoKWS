#!/usr/bin/env bash
set -euo pipefail

# Run far-field CosyVoice3 dataset generation on the shared server and keep
# command/log/meta files under ~/runs/AnJuXiaoBaoKWS.

if [[ -f "${HOME}/.anju_xiaobao_kws_env" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/.anju_xiaobao_kws_env"
fi

: "${ANJU_PROJECT:=${HOME}/projects/AnJuXiaoBaoKWS}"
: "${COSYVOICE_REPO:=${HOME}/projects/CosyVoice}"
: "${COSYVOICE3_MODEL:=${HOME}/models/Fun-CosyVoice3-0.5B}"
: "${ANJU_DATA_ROOT:=${HOME}/datasets/AnJuXiaoBaoKWS/data}"
: "${ANJU_BASE_DATASET:=${ANJU_DATA_ROOT}/anju_xiaobao_kws_dataset_20260508}"
: "${ANJU_RUN_ROOT:=${HOME}/runs/AnJuXiaoBaoKWS}"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${1:-farfield_cosyvoice3_${RUN_STAMP}}"
OUTPUT_DATASET="${2:-${ANJU_DATA_ROOT}/anju_xiaobao_farfield_cosyvoice3_${RUN_STAMP}}"

RUN_DIR="${ANJU_RUN_ROOT}/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

COMMAND_FILE="${RUN_DIR}/command.sh"
STDOUT_LOG="${RUN_DIR}/stdout.log"
STDERR_LOG="${RUN_DIR}/stderr.log"
META_FILE="${RUN_DIR}/run_meta.json"

cat > "${COMMAND_FILE}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source "${HOME}/.anju_xiaobao_kws_env"
python -m anju_kws.tts.generate_farfield_cosyvoice3_dataset \\
  --dataset_root "${ANJU_BASE_DATASET}" \\
  --cosyvoice_repo "${COSYVOICE_REPO}" \\
  --model_dir "${COSYVOICE3_MODEL}" \\
  --output_dir "${OUTPUT_DATASET}" \\
  --max_aishell3_speakers 218 \\
  --max_aishell_speakers 200 \\
  --samples_per_speaker 2 \\
  --profiles near_0p5m,mid_1m,far_2m
EOF
chmod +x "${COMMAND_FILE}"

cat > "${META_FILE}" <<EOF
{
  "run_name": "${RUN_NAME}",
  "run_dir": "${RUN_DIR}",
  "project": "${ANJU_PROJECT}",
  "cosyvoice_repo": "${COSYVOICE_REPO}",
  "cosyvoice3_model": "${COSYVOICE3_MODEL}",
  "base_dataset": "${ANJU_BASE_DATASET}",
  "output_dataset": "${OUTPUT_DATASET}",
  "stdout_log": "${STDOUT_LOG}",
  "stderr_log": "${STDERR_LOG}",
  "created_at": "$(date --iso-8601=seconds)"
}
EOF

echo "Run directory: ${RUN_DIR}"
echo "Output dataset: ${OUTPUT_DATASET}"
echo "Command file: ${COMMAND_FILE}"
echo "Logs: ${STDOUT_LOG}, ${STDERR_LOG}"
echo
echo "Starting generation..."

bash "${COMMAND_FILE}" > "${STDOUT_LOG}" 2> "${STDERR_LOG}"

echo "Generation finished."
echo "Inspect logs with:"
echo "  tail -n 50 ${STDOUT_LOG}"
echo "  tail -n 50 ${STDERR_LOG}"
