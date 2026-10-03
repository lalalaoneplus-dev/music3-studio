#!/usr/bin/env bash
# One-line macOS Apple Silicon installer. Safe when piped and when run from a checkout.
set -euo pipefail

DEFAULT_INSTALL_DIR="${HOME}/music3-studio"
DEFAULT_REPO_URL="https://github.com/lalalaoneplus-dev/music3-studio.git"
DEFAULT_TARBALL_URL="https://github.com/lalalaoneplus-dev/music3-studio/archive/refs/heads/main.tar.gz"

die() {
  echo "$*" >&2
  exit 1
}

require_macos_apple_silicon() {
  local sys mach
  sys="$(uname -s)"
  mach="$(uname -m)"
  if [[ "$sys" != "Darwin" ]]; then
    die "This installer runs on macOS. Detected ${sys}."
  fi
  if [[ "$mach" != "arm64" ]]; then
    die "Music3 Studio runs on Apple Silicon. Detected ${mach}."
  fi
}

looks_like_repo() {
  local d=$1
  [[ -f "${d}/app_qt.py" && -f "${d}/engine.py" && -f "${d}/config.py" ]]
}

script_checkout() {
  local src="${BASH_SOURCE[0]:-}"
  case "$src" in
    ""|"-"|"bash"|/dev/fd/*|/proc/self/fd/*) return 1 ;;
  esac
  [[ -f "$src" ]] || return 1
  local dir
  dir="$(cd "$(dirname "$src")" && pwd)"
  looks_like_repo "$dir" || return 1
  printf '%s\n' "$dir"
}

ensure_uv() {
  export PATH="${HOME}/.local/bin:${PATH}"
  if command -v uv >/dev/null 2>&1; then
    return 0
  fi
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
  command -v uv >/dev/null 2>&1 || die "uv is required. It installs into ${HOME}/.local/bin."
}

sync_source() {
  local dest=$1
  local url="${REPO_URL:-$DEFAULT_REPO_URL}"
  local tarball="${TARBALL_URL:-$DEFAULT_TARBALL_URL}"

  mkdir -p "$dest"

  if [[ -d "$url" ]]; then
    url="$(cd "$url" && pwd)"
    dest="$(cd "$dest" && pwd)"
    if [[ "$url" == "$dest" ]]; then
      return 0
    fi
    /usr/bin/rsync -a \
      --exclude '.git/' \
      --exclude '.venv/' \
      --exclude '__pycache__/' \
      --exclude '*.pyc' \
      --exclude '*.log' \
      --exclude 'config.json' \
      "${url}/" "${dest}/"
    return 0
  fi

  if command -v git >/dev/null 2>&1; then
    if [[ -d "${dest}/.git" ]]; then
      git -C "$dest" pull --ff-only
      return 0
    fi
    if [[ -z "$(ls -A "$dest" 2>/dev/null || true)" ]]; then
      git clone "$url" "$dest"
      return 0
    fi
    return 0
  fi

  local tmp inner
  tmp="$(mktemp -d)"
  curl -fsSL "$tarball" -o "${tmp}/src.tar.gz"
  tar -xzf "${tmp}/src.tar.gz" -C "$tmp"
  for inner in "$tmp"/*; do
    if [[ -d "$inner" ]]; then
      /usr/bin/rsync -a --exclude 'config.json' "${inner}/" "${dest}/"
      break
    fi
  done
  rm -rf "$tmp"
}

weights_complete() {
  local py=$1
  local engine=$2
  local model_dir=$3
  [[ -x "$py" && -f "$engine" && -f "${model_dir}/model_manifest.json" ]] || return 1
  "$py" "$engine" --model-dir "$model_dir" --check | python3 -c '
import json, sys
raw = sys.stdin.read().strip().splitlines()
if not raw:
    sys.exit(1)
data = json.loads(raw[-1])
sys.exit(0 if not data.get("missing") else 1)
'
}

require_macos_apple_silicon

REPO_DIR=""
if checkout="$(script_checkout 2>/dev/null || true)" && [[ -n "${checkout}" && -z "${INSTALL_DIR:-}" ]]; then
  REPO_DIR="$checkout"
elif [[ -z "${INSTALL_DIR:-}" && -z "${REPO_URL:-}" ]] && looks_like_repo "$PWD"; then
  REPO_DIR="$PWD"
else
  REPO_DIR="${INSTALL_DIR:-$DEFAULT_INSTALL_DIR}"
  sync_source "$REPO_DIR"
fi

looks_like_repo "$REPO_DIR" || die "music3-studio files were not found in ${REPO_DIR}"
[[ -f "${REPO_DIR}/requirements-engine.txt" ]] || die "Missing ${REPO_DIR}/requirements-engine.txt"
[[ -f "${REPO_DIR}/requirements.txt" ]] || die "Missing ${REPO_DIR}/requirements.txt"

ensure_uv

ENGINE_VENV="${HOME}/models/music3-venv"
MODEL_DIR="${HOME}/models/MiniMax-Music3-MLX"
mkdir -p "${HOME}/models"

if [[ ! -x "${ENGINE_VENV}/bin/python" ]] || ! "${ENGINE_VENV}/bin/python" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))'; then
  uv venv --python 3.12 --clear "$ENGINE_VENV"
fi
uv pip install -r "${REPO_DIR}/requirements-engine.txt" --python "${ENGINE_VENV}/bin/python"

if [[ ! -x "${REPO_DIR}/.venv/bin/python" ]]; then
  uv venv --python 3.12 "${REPO_DIR}/.venv"
fi
uv pip install -r "${REPO_DIR}/requirements.txt" --python "${REPO_DIR}/.venv/bin/python"

if [[ "${SKIP_WEIGHTS:-}" == "1" ]]; then
  echo "Skipping weight download (SKIP_WEIGHTS=1)."
elif weights_complete "${ENGINE_VENV}/bin/python" "${REPO_DIR}/engine.py" "$MODEL_DIR"; then
  echo "Model folder already complete: ${MODEL_DIR}"
else
  bash "${REPO_DIR}/packaging/fetch-weights.sh" "$MODEL_DIR"
fi

if [[ -f "${MODEL_DIR}/model_manifest.json" ]]; then
  "${ENGINE_VENV}/bin/python" "${REPO_DIR}/engine.py" --model-dir "$MODEL_DIR" --check
fi

if [[ "${NO_START:-}" != "1" ]]; then
  nohup "${REPO_DIR}/.venv/bin/python" "${REPO_DIR}/app_qt.py" >"${REPO_DIR}/music3-studio.log" 2>&1 &
fi

echo "Music3 Studio is in ${REPO_DIR}"
echo "Launch: ${REPO_DIR}/.venv/bin/python ${REPO_DIR}/app_qt.py"
echo "Engine: ${ENGINE_VENV}  Weights: ${MODEL_DIR}"
