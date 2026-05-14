#!/usr/bin/env bash
# Interactive helper: convert a weights file to a TensorRT .engine
#
# Prompts for:
#   - Model path (relative to repo root, or absolute)
#   - Export image size (e.g. 960, 640); used as square side (HxW) for both backends
#   - Type: 1 = YOLO (Ultralytics, .pt), 2 = RF-DETR (.pth)
#
# Run from anywhere:
#   bash scripts/convert_model_to_engine.sh
#
# Optional non-interactive mode:
#   MODEL=models/best.pt IMGSZ=960 TYPE=1 bash scripts/convert_model_to_engine.sh
#   MODEL=models/foo.pth IMGSZ=640 TYPE=2 NUM_CLASSES=4 bash scripts/convert_model_to_engine.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

TRTEXEC_BIN="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"

resolve_model_path() {
  local raw="$1"
  raw="${raw/#\~/${HOME}}"
  if [[ -z "${raw}" ]]; then
    echo "ERROR: empty model path" >&2
    exit 1
  fi
  if [[ "${raw}" = /* ]]; then
    echo "${raw}"
    return
  fi
  echo "${REPO_ROOT}/${raw}"
}

# --- inputs: env overrides or prompts ---
if [[ -z "${MODEL:-}" ]]; then
  read -r -p "Model path (relative to ${REPO_ROOT}, or absolute): " MODEL
fi
MODEL="$(resolve_model_path "${MODEL}")"

if [[ -z "${IMGSZ:-}" ]]; then
  read -r -p "Export image size (square H=W, e.g. 960 or 640) [640]: " IMGSZ
  IMGSZ="${IMGSZ:-640}"
fi
if ! [[ "${IMGSZ}" =~ ^[0-9]+$ ]] || [[ "${IMGSZ}" -lt 32 ]]; then
  echo "ERROR: IMGSZ must be an integer >= 32, got '${IMGSZ}'" >&2
  exit 1
fi

if [[ -z "${TYPE:-}" ]]; then
  echo ""
  echo "Conversion type:"
  echo "  1) YOLO — Ultralytics export (use .pt weights)"
  echo "  2) RF-DETR — ONNX export + trtexec (use .pth checkpoint)"
  read -r -p "Choose 1 or 2 [1]: " TYPE
  TYPE="${TYPE:-1}"
fi

if [[ ! -f "${MODEL}" ]]; then
  echo "ERROR: file not found: ${MODEL}" >&2
  exit 1
fi

ext="$(printf '%s' "${MODEL}" | tr '[:upper:]' '[:lower:]')"
ext="${ext##*.}"

# --- YOLO ---
convert_yolo() {
  if [[ "${ext}" != "pt" ]]; then
    echo "WARN: YOLO path usually ends in .pt; you gave .${ext}" >&2
  fi
  if ! command -v yolo >/dev/null 2>&1; then
    echo "ERROR: 'yolo' CLI not found. Install ultralytics in your active env, e.g. pip install ultralytics" >&2
    exit 2
  fi
  echo ""
  echo "==> YOLO export: model=${MODEL} format=engine imgsz=${IMGSZ}"
  # device=0 half=True are typical on Jetson; override with env if needed
  local dev="${YOLO_DEVICE:-0}"
  yolo export model="${MODEL}" format=engine "imgsz=${IMGSZ}" "device=${dev}" half=True
  local stem
  stem="$(basename "${MODEL%.*}")"
  local dir
  dir="$(dirname "${MODEL}")"
  echo ""
  echo "DONE (Ultralytics writes engine next to weights, often ${dir}/${stem}.engine)."
}

# --- RF-DETR ---
convert_rfdetr() {
  if [[ $((IMGSZ % 32)) -ne 0 ]]; then
    echo "ERROR: RF-DETR export requires image size divisible by 32; got ${IMGSZ}" >&2
    exit 1
  fi
  if [[ "${ext}" != "pth" ]]; then
    echo "WARN: RF-DETR checkpoints are usually .pth; you gave .${ext}" >&2
  fi
  if ! command -v "${TRTEXEC_BIN}" >/dev/null 2>&1; then
    echo "ERROR: trtexec not found at ${TRTEXEC_BIN}. Set TRTEXEC=/path/to/trtexec" >&2
    exit 2
  fi

  if [[ -z "${NUM_CLASSES:-}" ]]; then
    read -r -p "RF-DETR num_classes (must match checkpoint) [4]: " NUM_CLASSES
    NUM_CLASSES="${NUM_CLASSES:-4}"
  fi
  if ! [[ "${NUM_CLASSES}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: NUM_CLASSES must be an integer" >&2
    exit 1
  fi

  local stem out_dir onnx_path engine_path
  stem="$(basename "${MODEL%.*}")"
  out_dir="${REPO_ROOT}/models/rfdetr_export_${stem}_${IMGSZ}"
  mkdir -p "${out_dir}"
  onnx_path="${out_dir}/inference_model.onnx"
  engine_path="${out_dir}/rfdetr_fp16.engine"

  echo ""
  echo "==> RF-DETR ONNX export: weights=${MODEL} shape=${IMGSZ}x${IMGSZ} num_classes=${NUM_CLASSES}"
  python3 <<PY
import os
import torch.nn.functional as F
from rfdetr import RFDETRSmall

weights = r"""${MODEL}"""
out_dir = r"""${out_dir}"""
side = int(${IMGSZ})
num_classes = int(${NUM_CLASSES})

_real = F.interpolate

def _patched(input, size=None, scale_factor=None, mode="nearest", align_corners=None, recompute_scale_factor=None, antialias=False):
    if antialias:
        antialias = False
        if mode == "bicubic":
            mode = "bilinear"
    return _real(
        input,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
        recompute_scale_factor=recompute_scale_factor,
        antialias=antialias,
    )

F.interpolate = _patched
try:
    m = RFDETRSmall(pretrain_weights=weights, num_classes=num_classes)
    m.export(output_dir=out_dir, simplify=False, shape=(side, side), batch_size=1, opset_version=17, verbose=True)
finally:
    F.interpolate = _real

print("WROTE:", os.path.join(out_dir, "inference_model.onnx"))
PY

  echo ""
  echo "==> TensorRT build: ${TRTEXEC_BIN}"
  "${TRTEXEC_BIN}" \
    --onnx="${onnx_path}" \
    --saveEngine="${engine_path}" \
    --fp16 \
    --memPoolSize=workspace:4096 \
    --verbose=0

  echo ""
  echo "DONE"
  echo "  ONNX:   ${onnx_path}"
  echo "  ENGINE: ${engine_path}"
}

case "${TYPE}" in
  1|yolo|YOLO|y|Y)
    convert_yolo
    ;;
  2|rfdetr|RF-DETR|RFDETR|r|R)
    convert_rfdetr
    ;;
  *)
    echo "ERROR: invalid type '${TYPE}' (use 1 or 2)" >&2
    exit 1
    ;;
esac
