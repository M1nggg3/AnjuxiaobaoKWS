#!/usr/bin/env bash
set -euo pipefail

BASE_ENV="${HOME}/miniforge3/envs/cosyvoice310"
WEKWS_ENV="${HOME}/miniforge3/envs/wekws310"
CONDA_BIN="${HOME}/miniforge3/bin/conda"

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "conda not found: ${CONDA_BIN}" >&2
  exit 1
fi

if [[ ! -d "${WEKWS_ENV}" ]]; then
  "${CONDA_BIN}" create -y -p "${WEKWS_ENV}" --clone "${BASE_ENV}"
fi

PYTHON_BIN="${WEKWS_ENV}/bin/python"
"${PYTHON_BIN}" -m pip install \
  tensorboardX lmdb langid pypinyin onnx onnxruntime tensorboard \
  "git+https://github.com/wenet-e2e/wenet.git@1a94771df418325fd335b0b53776f83d74a8111d"

"${PYTHON_BIN}" - <<'PY'
import importlib
import torch
mods = ["torch", "torchaudio", "yaml", "tensorboardX", "lmdb", "langid", "pypinyin", "wenet"]
for name in mods:
    mod = importlib.import_module(name)
    print("OK", name, getattr(mod, "__version__", ""))
print("cuda", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
