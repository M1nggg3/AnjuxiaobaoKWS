#!/usr/bin/env bash
set -euo pipefail

if [[ -f "${HOME}/.anju_xiaobao_kws_env" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/.anju_xiaobao_kws_env"
fi

: "${ANJU_PROJECT:=${HOME}/projects/AnJuXiaoBaoKWS}"
: "${ANJU_DATA_ROOT:=${HOME}/datasets/AnJuXiaoBaoKWS/data}"
: "${ANJU_RUN_ROOT:=${HOME}/runs/AnJuXiaoBaoKWS}"
: "${WEKWS_MODEL_ROOT:=${HOME}/models/AnJuXiaoBaoKWS/wekws}"
: "${PYTHON_BIN:=${HOME}/miniforge3/envs/cosyvoice310/bin/python}"

DATASET_ROOT="${ANJU_DATA_ROOT}/anju_xiaobao_cosyvoice3_cleaned_farfield_20260521"
PREPARED_DIR="${ANJU_DATA_ROOT}/prepared_wekws_farfield_main_20260521"
DICT_DIR="${WEKWS_MODEL_ROOT}/dict/farfield_main_20260521"
PRETRAIN_DIR="${WEKWS_MODEL_ROOT}/pretrained/fsmn_ctc_wenwen"
CONFIG="${ANJU_PROJECT}/configs/train/fsmn_ctc_farfield_main_20260521.yaml"
EXP_DIR="${WEKWS_MODEL_ROOT}/experiments/fsmn_ctc_farfield_main_20260521_001"
TENSORBOARD_DIR="${ANJU_RUN_ROOT}/tensorboard"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="wekws_train_farfield_main_${RUN_STAMP}"
RUN_DIR="${ANJU_RUN_ROOT}/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

COMMAND_FILE="${RUN_DIR}/command.sh"
STDOUT_LOG="${RUN_DIR}/train.out.log"
STDERR_LOG="${RUN_DIR}/train.err.log"
META_FILE="${RUN_DIR}/run_meta.json"
INIT_CKPT="${EXP_DIR}/partial_pretrain_init.pt"

if [[ -e "${EXP_DIR}" ]] && [[ -n "$(find "${EXP_DIR}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refuse to overwrite non-empty experiment directory: ${EXP_DIR}" >&2
  exit 1
fi

cat > "${COMMAND_FILE}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${ANJU_PROJECT}/src:${ANJU_PROJECT}/third_party/wekws:\${PYTHONPATH:-}"
export LOCAL_RANK=0
export WORLD_SIZE=1
cd "${ANJU_PROJECT}"

"${PYTHON_BIN}" -m anju_kws.data.build_farfield_wekws_dataset \\
  --dataset_root "${DATASET_ROOT}" \\
  --output_dir "${PREPARED_DIR}" \\
  --dict_dir "${DICT_DIR}" \\
  --include_clean \\
  --balance_negatives

mkdir -p "${EXP_DIR}" "${TENSORBOARD_DIR}"

"${PYTHON_BIN}" -m anju_kws.train.make_partial_pretrain_checkpoint \\
  --config "${CONFIG}" \\
  --pretrained "${PRETRAIN_DIR}/avg_30.pt" \\
  --output "${INIT_CKPT}"

"${PYTHON_BIN}" "${ANJU_PROJECT}/third_party/wekws/wekws/bin/train.py" \\
  --config "${CONFIG}" \\
  --train_data "${PREPARED_DIR}/train/data.list" \\
  --cv_data "${PREPARED_DIR}/dev/data.list" \\
  --model_dir "${EXP_DIR}" \\
  --tensorboard_dir "${TENSORBOARD_DIR}" \\
  --dict "${DICT_DIR}" \\
  --num_keywords 6 \\
  --min_duration 5 \\
  --num_workers 4 \\
  --prefetch 8 \\
  --gpus 0 \\
  --checkpoint "${INIT_CKPT}"
EOF
chmod +x "${COMMAND_FILE}"

cat > "${META_FILE}" <<EOF
{
  "run_name": "${RUN_NAME}",
  "run_dir": "${RUN_DIR}",
  "project": "${ANJU_PROJECT}",
  "python_bin": "${PYTHON_BIN}",
  "dataset_root": "${DATASET_ROOT}",
  "prepared_dir": "${PREPARED_DIR}",
  "dict_dir": "${DICT_DIR}",
  "pretrain_dir": "${PRETRAIN_DIR}",
  "config": "${CONFIG}",
  "experiment_dir": "${EXP_DIR}",
  "init_checkpoint": "${INIT_CKPT}",
  "stdout_log": "${STDOUT_LOG}",
  "stderr_log": "${STDERR_LOG}",
  "tensorboard_dir": "${TENSORBOARD_DIR}",
  "gpus": "0",
  "max_epoch": 20,
  "created_at": "$(date --iso-8601=seconds)"
}
EOF

echo "Run directory: ${RUN_DIR}"
echo "Experiment directory: ${EXP_DIR}"
echo "Prepared data: ${PREPARED_DIR}"
echo "Command file: ${COMMAND_FILE}"
echo "Logs: ${STDOUT_LOG}, ${STDERR_LOG}"
echo
echo "Starting WeKWS farfield training..."

bash "${COMMAND_FILE}" > "${STDOUT_LOG}" 2> "${STDERR_LOG}"

echo "WeKWS farfield training finished."
echo "Inspect logs with:"
echo "  tail -n 50 ${STDOUT_LOG}"
echo "  tail -n 50 ${STDERR_LOG}"
