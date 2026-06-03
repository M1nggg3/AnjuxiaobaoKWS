#!/usr/bin/env bash
set -euo pipefail

# Generate formal gender-balanced pure wake-word clean CosyVoice3 TTS on the shared server.
# Far-field/noise mixing is intentionally left for the local workstation.

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
RUN_NAME="${1:-clean_cosyvoice3_purewake_gender_balanced_${RUN_STAMP}}"
OUTPUT_DATASET="${2:-${ANJU_DATA_ROOT}/anju_xiaobao_cosyvoice3_clean_purewake_${RUN_STAMP}}"

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
  --stage clean \\
  --dataset_root "${ANJU_BASE_DATASET}" \\
  --cosyvoice_repo "${COSYVOICE_REPO}" \\
  --model_dir "${COSYVOICE3_MODEL}" \\
  --output_dir "${OUTPUT_DATASET}" \\
  --max_aishell3_speakers 0 \\
  --max_aishell_speakers 0 \\
  --max_prompt_candidates_per_speaker 12 \\
  --max_speakers_per_gender 0 \\
  --samples_per_speaker 6 \\
  --negatives_per_speaker 2
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
  "stage": "clean",
  "positive_text_mode": "pure_wake_word",
  "gender_balance": true,
  "max_aishell3_speakers": 0,
  "max_aishell_speakers": 0,
  "max_prompt_candidates_per_speaker": 12,
  "max_speakers_per_gender": 0,
  "samples_per_speaker": 6,
  "negatives_per_speaker": 2,
  "include_homophone_negatives": false,
  "created_at": "$(date --iso-8601=seconds)"
}
EOF

echo "Run directory: ${RUN_DIR}"
echo "Output clean dataset: ${OUTPUT_DATASET}"
echo "Command file: ${COMMAND_FILE}"
echo "Logs: ${STDOUT_LOG}, ${STDERR_LOG}"
echo
echo "Starting formal pure wake-word clean TTS generation..."

bash "${COMMAND_FILE}" > "${STDOUT_LOG}" 2> "${STDERR_LOG}"

echo "Formal pure wake-word clean TTS generation finished."
echo "Inspect logs with:"
echo "  tail -n 50 ${STDOUT_LOG}"
echo "  tail -n 50 ${STDERR_LOG}"
