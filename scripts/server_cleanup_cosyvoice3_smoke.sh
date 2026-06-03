#!/usr/bin/env bash
set -euo pipefail

# Preview or remove temporary CosyVoice3 smoke/dryrun datasets and run records.
# This script is conservative by design: it never removes anything unless
# --apply is passed, and every deletion candidate must live under ANJU_DATA_ROOT
# or ANJU_RUN_ROOT.

APPLY=0
YES=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/server_cleanup_cosyvoice3_smoke.sh [--apply] [--yes]

Options:
  --apply   Actually remove matched smoke/dryrun directories.
  --yes     Skip the interactive confirmation when used with --apply.
  -h,--help Show this help.

Default behavior is preview-only.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply)
      APPLY=1
      ;;
    --yes)
      YES=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if [[ -f "${HOME}/.anju_xiaobao_kws_env" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/.anju_xiaobao_kws_env"
fi

: "${ANJU_DATA_ROOT:=${HOME}/datasets/AnJuXiaoBaoKWS/data}"
: "${ANJU_RUN_ROOT:=${HOME}/runs/AnJuXiaoBaoKWS}"

DATA_ROOT="$(realpath -m "${ANJU_DATA_ROOT}")"
RUN_ROOT="$(realpath -m "${ANJU_RUN_ROOT}")"

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "Data root does not exist: ${DATA_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${RUN_ROOT}" ]]; then
  echo "Run root does not exist: ${RUN_ROOT}" >&2
  exit 1
fi

shopt -s nullglob

declare -a CANDIDATES=()
declare -A SEEN=()

add_candidate() {
  local root="$1"
  local path="$2"
  local resolved
  resolved="$(realpath -m "${path}")"

  case "${resolved}" in
    "${root}"/*)
      ;;
    *)
      echo "Refusing candidate outside root: ${resolved}" >&2
      exit 1
      ;;
  esac

  if [[ -d "${resolved}" && -z "${SEEN[${resolved}]:-}" ]]; then
    CANDIDATES+=("${resolved}")
    SEEN["${resolved}"]=1
  fi
}

for path in "${DATA_ROOT}"/anju_xiaobao_cosyvoice3_clean_smoke_*; do
  add_candidate "${DATA_ROOT}" "${path}"
done

for path in "${DATA_ROOT}"/anju_xiaobao_cosyvoice3_clean_dryrun_*; do
  add_candidate "${DATA_ROOT}" "${path}"
done

for path in "${RUN_ROOT}"/clean_cosyvoice3_smoke_*; do
  add_candidate "${RUN_ROOT}" "${path}"
done

for path in "${RUN_ROOT}"/clean_cosyvoice3_dryrun_*; do
  add_candidate "${RUN_ROOT}" "${path}"
done

for path in "${RUN_ROOT}"/farfield_cosyvoice3_smoke_*; do
  add_candidate "${RUN_ROOT}" "${path}"
done

for path in "${RUN_ROOT}"/farfield_cosyvoice3_dryrun_*; do
  add_candidate "${RUN_ROOT}" "${path}"
done

if [[ ${#CANDIDATES[@]} -eq 0 ]]; then
  echo "No CosyVoice3 smoke/dryrun directories found."
  exit 0
fi

echo "Matched CosyVoice3 smoke/dryrun directories:"
for path in "${CANDIDATES[@]}"; do
  size="$(du -sh -- "${path}" 2>/dev/null | awk '{print $1}')"
  printf '  %8s  %s\n' "${size:-unknown}" "${path}"
done

if [[ "${APPLY}" -ne 1 ]]; then
  echo
  echo "Preview only. Re-run with --apply to delete these directories."
  exit 0
fi

if [[ "${YES}" -ne 1 ]]; then
  echo
  read -r -p "Type DELETE to permanently remove these directories: " confirmation
  if [[ "${confirmation}" != "DELETE" ]]; then
    echo "Cleanup cancelled."
    exit 1
  fi
fi

for path in "${CANDIDATES[@]}"; do
  echo "Removing ${path}"
  rm -rf -- "${path}"
done

echo "Cleanup complete."
