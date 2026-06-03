#!/usr/bin/env bash
set -euo pipefail

# Initialize the personal workspace layout for AnJuXiaoBaoKWS on the shared server.
# This script is intentionally non-destructive: it creates missing directories
# and does not move, delete, or overwrite existing project data.

USER_HOME="${HOME}"
PROJECT_NAME="AnJuXiaoBaoKWS"

mkdir -p "${USER_HOME}/projects"
mkdir -p "${USER_HOME}/models/Fun-CosyVoice3-0.5B"
mkdir -p "${USER_HOME}/models/${PROJECT_NAME}/wekws"
mkdir -p "${USER_HOME}/models/${PROJECT_NAME}/prototype_dscnn"
mkdir -p "${USER_HOME}/datasets/${PROJECT_NAME}/data"
mkdir -p "${USER_HOME}/outputs/${PROJECT_NAME}/smoke"
mkdir -p "${USER_HOME}/outputs/${PROJECT_NAME}/inference_debug"
mkdir -p "${USER_HOME}/runs/${PROJECT_NAME}"
mkdir -p "${USER_HOME}/logs/${PROJECT_NAME}"
mkdir -p "${USER_HOME}/downloads"
mkdir -p "${USER_HOME}/tmp"
mkdir -p "${USER_HOME}/archive/${PROJECT_NAME}"

ENV_FILE="${USER_HOME}/.anju_xiaobao_kws_env"
if [[ ! -f "${ENV_FILE}" ]]; then
  cat > "${ENV_FILE}" <<'EOF'
# AnJuXiaoBaoKWS personal server paths.
export ANJU_PROJECT="$HOME/projects/AnJuXiaoBaoKWS"
export COSYVOICE_REPO="$HOME/projects/CosyVoice"
export COSYVOICE3_MODEL="$HOME/models/Fun-CosyVoice3-0.5B"
export ANJU_DATA_ROOT="$HOME/datasets/AnJuXiaoBaoKWS/data"
export ANJU_BASE_DATASET="$ANJU_DATA_ROOT/anju_xiaobao_kws_dataset_20260508"
export ANJU_OUTPUT_ROOT="$HOME/outputs/AnJuXiaoBaoKWS"
export ANJU_RUN_ROOT="$HOME/runs/AnJuXiaoBaoKWS"
export PYTHONPATH="$ANJU_PROJECT/src:$ANJU_PROJECT:$COSYVOICE_REPO:$COSYVOICE_REPO/third_party/Matcha-TTS:${PYTHONPATH:-}"
EOF
else
  echo "Environment file already exists, not overwritten: ${ENV_FILE}"
fi

DATA_README="${USER_HOME}/datasets/${PROJECT_NAME}/README.md"
if [[ ! -f "${DATA_README}" ]]; then
  cat > "${DATA_README}" <<'EOF'
# AnJuXiaoBaoKWS datasets

This directory stores formal datasets for the AnJuXiaoBaoKWS project.

Recommended layout:

- `data/anju_xiaobao_kws_dataset_20260508/`: curated base dataset.
- `data/anju_xiaobao_farfield_cosyvoice3_YYYYMMDD/`: generated far-field simulation datasets.

Rules:

- Do not store source code here.
- Do not store model checkpoints here.
- Each formal dataset should include `README.md` or `summary.json`.
- Temporary listening tests and smoke outputs should go to `~/outputs/AnJuXiaoBaoKWS`.
EOF
fi

echo "Workspace initialized."
echo
echo "Load environment variables with:"
echo "  source ${ENV_FILE}"
echo
echo "Directory check:"
ls -ld \
  "${USER_HOME}/projects" \
  "${USER_HOME}/models" \
  "${USER_HOME}/datasets" \
  "${USER_HOME}/outputs" \
  "${USER_HOME}/runs" \
  "${USER_HOME}/logs" \
  "${USER_HOME}/downloads" \
  "${USER_HOME}/tmp" \
  "${USER_HOME}/archive"
