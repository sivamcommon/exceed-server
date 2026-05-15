"""
Re-export RF-DETR to ONNX with dynamic batch axis, then build a TRT fp16 engine.

Usage:
    .venv/bin/python models/export_rfdetr_dynamic.py
"""
import os
import sys
import subprocess

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import torch
import torch.nn.functional as F

# Patch antialias interpolate so tracing doesn't fail on Jetson
_real_interp = F.interpolate
def _patched(input, size=None, scale_factor=None, mode="nearest",
             align_corners=None, recompute_scale_factor=None, antialias=False):
    if antialias:
        antialias = False
        if mode == "bicubic":
            mode = "bilinear"
    return _real_interp(input, size=size, scale_factor=scale_factor, mode=mode,
                        align_corners=align_corners,
                        recompute_scale_factor=recompute_scale_factor,
                        antialias=antialias)
F.interpolate = _patched

from rfdetr import RFDETRSmall

WEIGHTS   = "models/rf_detr_640_29-5.pth"
OUT_DIR   = "models"
IMGSZ     = 512
N_CLASSES = 6
OPSET     = 17

print(f"Loading RF-DETR weights: {WEIGHTS}")
model = RFDETRSmall(pretrain_weights=WEIGHTS, num_classes=N_CLASSES)
model.model.model.eval()

print(f"Exporting ONNX (dynamic_batch=True, shape={IMGSZ}x{IMGSZ}) …")
model.export(
    output_dir=OUT_DIR,
    shape=(IMGSZ, IMGSZ),
    batch_size=1,
    dynamic_batch=True,
    opset_version=OPSET,
    verbose=True,
)

# Find the exported file
import glob
candidates = sorted(glob.glob(f"{OUT_DIR}/*.onnx"), key=os.path.getmtime, reverse=True)
if not candidates:
    print("ERROR: no .onnx found in models/")
    sys.exit(1)
onnx_path = candidates[0]
print(f"ONNX exported: {onnx_path}")

# Verify dynamic axes
import onnx
m = onnx.load(onnx_path)
print("Input shapes:")
for inp in m.graph.input:
    s = [d.dim_param if d.dim_param else d.dim_value for d in inp.type.tensor_type.shape.dim]
    print(f"  {inp.name}: {s}")
print("Output shapes:")
for out in m.graph.output:
    s = [d.dim_param if d.dim_param else d.dim_value for d in out.type.tensor_type.shape.dim]
    print(f"  {out.name}: {s}")

engine_out = "models/rfdetr_dynamic_fp16.engine"
trtexec = "/usr/src/tensorrt/bin/trtexec"
cmd = [
    trtexec,
    f"--onnx={onnx_path}",
    "--fp16",
    "--minShapes=input:1x3x512x512",
    "--optShapes=input:6x3x512x512",
    "--maxShapes=input:16x3x512x512",
    f"--saveEngine={engine_out}",
]
print("\nBuilding TRT engine (this takes ~5 min) …")
print(" ".join(cmd))
ret = subprocess.run(cmd, capture_output=False)
if ret.returncode == 0:
    print(f"\nEngine saved: {engine_out}")
else:
    print(f"\ntrtexec failed (code {ret.returncode}) — check output above")
    sys.exit(ret.returncode)
