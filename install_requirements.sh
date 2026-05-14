#!/usr/bin/env bash
# Full setup for Server2: system packages (apt) + venv + Python requirements.
#
# Usage:
#   cd ~/Gomicro_Common/Server2
#   bash install_requirements.sh              # apt deps + venv + pip (aarch64 → jetson requirements)
#   bash install_requirements.sh --jetson     # force requirements-jetson-jp61.txt
#   bash install_requirements.sh --cpu        # force requirements.txt
#   bash install_requirements.sh --no-onnxruntime   # skip onnxruntime+onnxslim if ORT crashes on import
#   bash install_requirements.sh --no-apt     # skip sudo apt (CI or no sudo)
#
# Apt step installs: ffmpeg (infer/video recording), python3-venv, PyGObject + GStreamer introspection, pip.
# You will be prompted for sudo unless passwordless sudo is configured.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

MODE="auto"
SKIP_ONNXRUNTIME=0
NO_APT=0
for arg in "$@"; do
  case "${arg}" in
    --jetson) MODE="jetson" ;;
    --cpu)    MODE="cpu" ;;
    --no-onnxruntime) SKIP_ONNXRUNTIME=1 ;;
    --no-apt) NO_APT=1 ;;
    -h|--help)
      sed -n '1,30p' "$0"
      exit 0
      ;;
  esac
done

install_system_deps() {
  if [[ "${NO_APT}" -eq 1 ]]; then
    echo "=== skipping apt (--no-apt) ==="
    return 0
  fi
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "=== no apt-get; skipping system packages (install ffmpeg + GStreamer manually) ==="
    return 0
  fi
  echo "=== system packages (sudo apt-get) — ffmpeg, venv, GStreamer, PyGObject ==="
  export DEBIAN_FRONTEND=noninteractive
  sudo apt-get update -qq
  sudo apt-get install -y --no-install-recommends \
    ffmpeg \
    python3-pip \
    python3-venv \
    python3-gi \
    python3-gi-cairo \
    gir1.2-gstreamer-1.0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good
}

install_system_deps

if [[ "${MODE}" == "auto" ]]; then
  if [[ "$(uname -m)" == "aarch64" ]]; then
    MODE="jetson"
  else
    MODE="cpu"
  fi
fi

if [[ "${MODE}" == "jetson" ]]; then
  REQ="requirements-jetson-jp61.txt"
else
  REQ="requirements.txt"
fi

REQ_PATH="${ROOT}/${REQ}"
if [[ ! -f "${REQ_PATH}" ]]; then
  echo "ERROR: missing ${REQ_PATH}" >&2
  exit 1
fi

if ! python3 -m venv -h >/dev/null 2>&1; then
  echo "ERROR: python3-venv still missing after apt. Install: sudo apt install -y python3-venv" >&2
  exit 1
fi

echo "=== venv: ${ROOT}/.venv (system-site-packages for PyGObject/gi) ==="
if [[ ! -x "${ROOT}/.venv/bin/python" ]]; then
  python3 -m venv "${ROOT}/.venv" --system-site-packages
fi

PY="${ROOT}/.venv/bin/python"
PIP="${ROOT}/.venv/bin/pip"

"${PY}" -m pip install -U pip setuptools wheel

if [[ "${SKIP_ONNXRUNTIME}" -eq 1 ]]; then
  echo "=== installing from ${REQ} (mode=${MODE}), skipping onnxruntime/onnxslim ==="
  TMP="$(mktemp)"
  grep -v '^[[:space:]]*onnxslim' "${REQ_PATH}" | grep -v '^[[:space:]]*onnxruntime' > "${TMP}"
  "${PIP}" install -r "${TMP}"
  rm -f "${TMP}"
else
  echo "=== full install from ${REQ} (mode=${MODE}) ==="
  "${PIP}" install -r "${REQ_PATH}"
fi

echo ""
echo "Done. Activate with:"
echo "  source ${ROOT}/.venv/bin/activate"
echo "Run server:"
echo "  uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1"
