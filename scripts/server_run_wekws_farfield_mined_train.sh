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
PREPARED_DIR="${ANJU_DATA_ROOT}/prepared_wekws_farfield_mined_20260521"
DICT_DIR="${WEKWS_MODEL_ROOT}/dict/farfield_mined_20260521"
BASE_CONFIG="${ANJU_PROJECT}/configs/train/fsmn_ctc_farfield_main_20260521.yaml"
BASE_EXP_DIR="${WEKWS_MODEL_ROOT}/experiments/fsmn_ctc_farfield_main_20260521_001"
BASE_CHECKPOINT="${BASE_EXP_DIR}/final.pt"
EXP_DIR="${WEKWS_MODEL_ROOT}/experiments/fsmn_ctc_farfield_mined_20260521_002"
TENSORBOARD_DIR="${ANJU_RUN_ROOT}/tensorboard"
HARD_NEG_REPEAT="${HARD_NEG_REPEAT:-100}"
HARD_NEG_MIN_SCORE="${HARD_NEG_MIN_SCORE:-0.0}"
FINETUNE_LR="${FINETUNE_LR:-0.0003}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-10}"

if [[ -z "${EVAL_RUN_DIR:-}" ]]; then
  EVAL_RUN_DIR="$(find "${ANJU_RUN_ROOT}" -maxdepth 1 -type d -name 'wekws_score_ctc_eval_*' | sort | tail -n 1)"
fi
if [[ -z "${EVAL_RUN_DIR}" || ! -d "${EVAL_RUN_DIR}" ]]; then
  echo "Cannot find score_ctc eval run dir. Set EVAL_RUN_DIR explicitly." >&2
  exit 1
fi

HARD_NEG_DATA_LIST="${EVAL_RUN_DIR}/data/continuous_5s/data.list"
HARD_NEG_SCORE_FILE="${EVAL_RUN_DIR}/scores/continuous_5s.score"
if [[ ! -f "${HARD_NEG_DATA_LIST}" || ! -f "${HARD_NEG_SCORE_FILE}" ]]; then
  echo "Missing hard negative inputs under ${EVAL_RUN_DIR}" >&2
  exit 1
fi

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="wekws_train_farfield_mined_${RUN_STAMP}"
RUN_DIR="${ANJU_RUN_ROOT}/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

CONFIG="${RUN_DIR}/fsmn_ctc_farfield_mined_20260521.yaml"
COMMAND_FILE="${RUN_DIR}/command.sh"
STDOUT_LOG="${RUN_DIR}/train.out.log"
STDERR_LOG="${RUN_DIR}/train.err.log"
META_FILE="${RUN_DIR}/run_meta.json"

if [[ -e "${EXP_DIR}" ]] && [[ -n "$(find "${EXP_DIR}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refuse to overwrite non-empty experiment directory: ${EXP_DIR}" >&2
  exit 1
fi

"${PYTHON_BIN}" - "${BASE_CONFIG}" "${CONFIG}" "${FINETUNE_LR}" "${FINETUNE_EPOCHS}" <<'PY'
import sys
from pathlib import Path
import yaml

src, dst, lr, epochs = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
conf = yaml.safe_load(Path(src).read_text(encoding="utf-8"))
conf["optim_conf"]["lr"] = lr
conf["training_config"]["max_epoch"] = epochs
Path(dst).write_text(yaml.safe_dump(conf, allow_unicode=True, sort_keys=False), encoding="utf-8")
PY

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
  --balance_negatives \\
  --hard_negative_data_list "${HARD_NEG_DATA_LIST}" \\
  --hard_negative_score_file "${HARD_NEG_SCORE_FILE}" \\
  --hard_negative_min_score "${HARD_NEG_MIN_SCORE}" \\
  --hard_negative_repeat "${HARD_NEG_REPEAT}" \\
  --hard_negative_train_only

mkdir -p "${EXP_DIR}" "${TENSORBOARD_DIR}"

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
  --checkpoint "${BASE_CHECKPOINT}"
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
  "base_config": "${BASE_CONFIG}",
  "config": "${CONFIG}",
  "base_experiment_dir": "${BASE_EXP_DIR}",
  "base_checkpoint": "${BASE_CHECKPOINT}",
  "experiment_dir": "${EXP_DIR}",
  "eval_run_dir": "${EVAL_RUN_DIR}",
  "hard_negative_data_list": "${HARD_NEG_DATA_LIST}",
  "hard_negative_score_file": "${HARD_NEG_SCORE_FILE}",
  "hard_negative_repeat": ${HARD_NEG_REPEAT},
  "hard_negative_min_score": ${HARD_NEG_MIN_SCORE},
  "finetune_lr": ${FINETUNE_LR},
  "max_epoch": ${FINETUNE_EPOCHS},
  "stdout_log": "${STDOUT_LOG}",
  "stderr_log": "${STDERR_LOG}",
  "tensorboard_dir": "${TENSORBOARD_DIR}",
  "gpus": "0",
  "created_at": "$(date --iso-8601=seconds)"
}
EOF

echo "Run directory: ${RUN_DIR}"
echo "Experiment directory: ${EXP_DIR}"
echo "Prepared data: ${PREPARED_DIR}"
echo "Eval run: ${EVAL_RUN_DIR}"
echo "Hard negatives: ${HARD_NEG_SCORE_FILE}"
echo "Hard negative repeat: ${HARD_NEG_REPEAT}"
echo "Finetune lr: ${FINETUNE_LR}"
echo "Epochs: ${FINETUNE_EPOCHS}"
echo "Command file: ${COMMAND_FILE}"
echo "Logs: ${STDOUT_LOG}, ${STDERR_LOG}"
echo
echo "Starting mined hard-negative WeKWS training..."

bash "${COMMAND_FILE}" > "${STDOUT_LOG}" 2> "${STDERR_LOG}"

echo "Mined hard-negative WeKWS training finished."
echo "Inspect logs with:"
echo "  tail -n 50 ${STDOUT_LOG}"
echo "  tail -n 50 ${STDERR_LOG}"
