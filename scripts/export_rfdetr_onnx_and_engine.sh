#!/usr/bin/env bash
set -euo pipefail

# Export RF-DETR weights -> ONNX -> TensorRT engine on Jetson.
#
# Why the torch.nn.functional.interpolate monkeypatch:
# RF-DETR can use antialiased upsampling ops that the legacy torch.onnx exporter cannot emit.
# Disabling antialias during export is a practical workaround; keep the same patch for any
# "compare PT vs ONNX" checks.
#
# Usage:
#   bash scripts/export_rfdetr_onnx_and_engine.sh models/checkpoint_best_regular.pth 640 4
#
# Outputs:
#   models/rfdetr_export_<stem>/inference_model.onnx
#   models/rfdetr_export_<stem>/rfdetr_fp16.engine

WEIGHTS="${1:?path to .pth}"
SIDE="${2:-640}"
NUM_CLASSES="${3:-4}"

STEM="$(basename "${WEIGHTS%.*}")"
OUT_DIR="models/rfdetr_export_${STEM}"
mkdir -p "${OUT_DIR}"

TRTEXEC_BIN="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
if ! command -v "${TRTEXEC_BIN}" >/dev/null 2>&1; then
  echo "ERROR: trtexec not found at ${TRTEXEC_BIN}. Install JetPack TensorRT tools or set TRTEXEC=/path/to/trtexec" >&2
  exit 2
fi

python3 <<PY
import os
import torch.nn.functional as F
from rfdetr import RFDETRSmall

weights = "${WEIGHTS}"
out_dir = "${OUT_DIR}"
side = int(${SIDE})
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

onnx_path = os.path.join(out_dir, "inference_model.onnx")
print("\nWROTE:", onnx_path)
PY

ONNX_PATH="${OUT_DIR}/inference_model.onnx"
ENGINE_PATH="${OUT_DIR}/rfdetr_fp16.engine"

"${TRTEXEC_BIN}" \
  --onnx="${ONNX_PATH}" \
  --saveEngine="${ENGINE_PATH}" \
  --fp16 \
  --memPoolSize=workspace:4096 \
  --verbose=0

echo ""
echo "DONE"
echo "ONNX:   ${ONNX_PATH}"
echo "ENGINE: ${ENGINE_PATH}"
