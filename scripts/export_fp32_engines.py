#!/usr/bin/env python3
"""
Export two models to TensorRT FP32 engines:
1) YOLO segmentation .pt  -> .engine
2) RF-DETR .pth           -> ONNX -> .engine (FP32)

Example:
  python3 scripts/export_fp32_engines.py \
    --yolo-pt "models/yolov11m seg.pt" \
    --rfdetr-pth "models/checkpoint_best_regular.pth" \
    --out-dir "models/fp32_exports" \
    --imgsz 960 \
    --rfdetr-side 640 \
    --rfdetr-num-classes 4
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _check_file(path: str, label: str) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"{label} not found: {p}")
    return p


def _ensure_dir(path: str) -> Path:
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def export_yolo_fp32_engine(yolo_pt: Path, out_dir: Path, imgsz: int, device: int) -> Path:
    from ultralytics import YOLO

    model = YOLO(str(yolo_pt), task="segment")
    # FP32 export: half=False, int8=False
    engine_path = model.export(
        format="engine",
        imgsz=int(imgsz),
        half=False,
        int8=False,
        device=int(device),
        batch=1,
        workspace=4.0,
        verbose=False,
    )
    engine_path = Path(str(engine_path)).resolve()
    target = out_dir / f"{yolo_pt.stem}_fp32.engine"
    if engine_path != target:
        target.write_bytes(engine_path.read_bytes())
    return target


def export_rfdetr_onnx(rfdetr_pth: Path, out_dir: Path, side: int, num_classes: int) -> Path:
    import torch.nn.functional as F
    from rfdetr import RFDETRSmall

    _real = F.interpolate

    # Keep same export workaround used in repo shell script.
    def _patched(
        input,
        size=None,
        scale_factor=None,
        mode="nearest",
        align_corners=None,
        recompute_scale_factor=None,
        antialias=False,
    ):
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
        model = RFDETRSmall(pretrain_weights=str(rfdetr_pth), num_classes=int(num_classes))
        model.export(
            output_dir=str(out_dir),
            simplify=False,
            shape=(int(side), int(side)),
            batch_size=1,
            opset_version=17,
            verbose=False,
        )
    finally:
        F.interpolate = _real

    onnx_path = out_dir / "inference_model.onnx"
    if not onnx_path.is_file():
        raise RuntimeError(f"RF-DETR ONNX export failed: {onnx_path}")
    return onnx_path


def run_trtexec_fp32(onnx_path: Path, engine_path: Path, trtexec: str, workspace_mb: int) -> None:
    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--memPoolSize=workspace:{int(workspace_mb)}",
    ]
    # NOTE: No --fp16 / --int8 => FP32 engine build.
    proc = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"trtexec failed with exit code {proc.returncode}")
    if not engine_path.is_file():
        raise RuntimeError(f"Engine not generated: {engine_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Export YOLO + RF-DETR FP32 TensorRT engines")
    ap.add_argument("--yolo-pt", required=True, help="Path to YOLO seg .pt file")
    ap.add_argument("--rfdetr-pth", required=True, help="Path to RF-DETR .pth file")
    ap.add_argument("--out-dir", default="models/fp32_exports", help="Output directory")
    ap.add_argument("--imgsz", type=int, default=960, help="YOLO export image size")
    ap.add_argument("--device", type=int, default=0, help="CUDA device index for YOLO export")
    ap.add_argument("--rfdetr-side", type=int, default=640, help="RF-DETR ONNX export side")
    ap.add_argument("--rfdetr-num-classes", type=int, default=4, help="RF-DETR class count")
    ap.add_argument("--trtexec", default="/usr/src/tensorrt/bin/trtexec", help="Path to trtexec binary")
    ap.add_argument("--workspace-mb", type=int, default=4096, help="TensorRT workspace MB")
    args = ap.parse_args()

    yolo_pt = _check_file(args.yolo_pt, "YOLO .pt")
    rfdetr_pth = _check_file(args.rfdetr_pth, "RF-DETR .pth")
    out_dir = _ensure_dir(args.out_dir)

    if not Path(args.trtexec).is_file():
        raise FileNotFoundError(f"trtexec not found: {args.trtexec}")

    print(f"[1/3] Export YOLO FP32 engine from {yolo_pt}")
    yolo_engine = export_yolo_fp32_engine(yolo_pt, out_dir, args.imgsz, args.device)
    print(f"      -> {yolo_engine}")

    rfdetr_dir = _ensure_dir(str(out_dir / f"rfdetr_export_{rfdetr_pth.stem}"))
    print(f"[2/3] Export RF-DETR ONNX from {rfdetr_pth}")
    onnx_path = export_rfdetr_onnx(
        rfdetr_pth=rfdetr_pth,
        out_dir=rfdetr_dir,
        side=args.rfdetr_side,
        num_classes=args.rfdetr_num_classes,
    )
    print(f"      -> {onnx_path}")

    rfdetr_engine = out_dir / f"{rfdetr_pth.stem}_fp32.engine"
    print(f"[3/3] Build RF-DETR FP32 engine via trtexec")
    run_trtexec_fp32(
        onnx_path=onnx_path,
        engine_path=rfdetr_engine,
        trtexec=args.trtexec,
        workspace_mb=args.workspace_mb,
    )
    print(f"      -> {rfdetr_engine}")

    print("\nDONE")
    print(f"YOLO FP32 engine:   {yolo_engine}")
    print(f"RF-DETR FP32 engine:{rfdetr_engine}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

