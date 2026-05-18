"""
RF-DETR helpers for this repo.

Why this file exists:
- Stage-1/stage-2 inference in `infer_main.py` / `infer_detect.py` was built around Ultralytics YOLO.
- RF-DETR uses a different runtime (PyTorch `.pth/.pt`, ONNX, or TensorRT engine via ONNXRuntime).

This module provides:
- A small ONNXRuntime wrapper for `.engine` / `.onnx` RF-DETR exports
- A ByteTrack-wrapped "track()" compatible shim for stage-1 when `main.backend=rfdetr`
- Postprocessing that matches RF-DETR's `PostProcess` top-k selection logic (see upstream `rfdetr.models.postprocess`)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Keep transformers from importing TensorFlow/JAX in RF-DETR codepaths.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

import supervision as sv

try:
    import tensorrt as trt
except Exception:  # pragma: no cover
    trt = None  # type: ignore


def _read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _byte_track_from_ultralytics_yaml(tracker_yaml_path: str, frame_rate: int = 30) -> sv.ByteTrack:
    """
    Map Ultralytics' `bytetrack.yaml` knobs to supervision's ByteTrack ctor.

    This is best-effort: supervision exposes fewer knobs than Ultralytics, but the important
    thresholds come across closely enough for stable IDs in practice.
    """
    cfg = _read_yaml(tracker_yaml_path)
    th = float(cfg.get("track_high_thresh", 0.25))
    low = float(cfg.get("track_low_thresh", 0.1))
    new_t = float(cfg.get("new_track_thresh", th))
    buf = int(cfg.get("track_buffer", 30))
    match = float(cfg.get("match_thresh", 0.8))

    # supervision: track_activation_threshold ~= high thresh; minimum_matching_threshold ~= match_thresh
    # Use the min of high/new thresholds as activation (conservative).
    act = float(min(max(th, 1e-3), 0.999))
    min_match = float(min(max(match, 1e-3), 0.999))
    lost_buf = int(max(1, buf))

    # `low` isn't directly represented; keep defaults unless extreme.
    _ = low
    return sv.ByteTrack(
        track_activation_threshold=act,
        lost_track_buffer=lost_buf,
        minimum_matching_threshold=min_match,
        frame_rate=int(max(1, frame_rate)),
        minimum_consecutive_frames=1,
    )


class _SimpleBoxes:
    """Minimal YOLO-like `boxes` container for `infer_detect.extract_main_detections`."""

    def __init__(self, xyxy, conf, cls, id_=None):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls
        self.id = id_

    def __len__(self) -> int:
        return int(self.xyxy.shape[0])


class _SimpleResult:
    """Minimal YOLO-like `result` container for `infer_detect.extract_main_detections`."""

    def __init__(self, names: dict, boxes: _SimpleBoxes):
        self.names = names
        self.boxes = boxes


@dataclass
class RFDETRSessionConfig:
    """Runtime config for ORT+TRT RF-DETR sessions."""

    engine_path: str
    num_classes: int
    num_select: int = 300


class RFDETRTensorrtPredictor:
    """
    RF-DETR inference via native TensorRT runtime (preferred for `.engine`).

    Notes:
    - `trtexec` produces a TensorRT plan/engine file; ONNXRuntime cannot load that file directly.
    - Input is expected NCHW float32 in [0,1] normalized like RF-DETR training (ImageNet mean/std).
    """

    def __init__(self, cfg: RFDETRSessionConfig, means: Tuple[float, float, float], stds: Tuple[float, float, float]):
        if trt is None:
            raise RuntimeError("tensorrt python package is required for RF-DETR `.engine` inference")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for RF-DETR `.engine` inference on this integration path")

        self.cfg = cfg
        self.means = np.array(means, dtype=np.float32).reshape(1, 3, 1, 1)
        self.stds = np.array(stds, dtype=np.float32).reshape(1, 3, 1, 1)

        logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(logger)
        with open(cfg.engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {cfg.engine_path}")
        self.context = self.engine.create_execution_context()
        logging.info("RF-DETR TensorRT engine deserialized: %s", cfg.engine_path)

        self.input_name = None
        self.output_names: List[str] = []
        self.name_to_torch_dtype: Dict[str, torch.dtype] = {}
        self.name_to_shape: Dict[str, Tuple[int, ...]] = {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = tuple(self.engine.get_tensor_shape(name))
            dt = self.engine.get_tensor_dtype(name)
            torch_dt = {
                trt.float32: torch.float32,
                trt.float16: torch.float16,
                trt.int32: torch.int32,
                trt.int64: torch.int64,
            }.get(dt, torch.float32)
            self.name_to_torch_dtype[name] = torch_dt
            self.name_to_shape[name] = shape
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_names.append(name)

        if not self.input_name:
            raise RuntimeError("TensorRT engine has no input tensor")
        if len(self.output_names) < 2:
            raise RuntimeError(f"TensorRT engine missing expected outputs: {self.output_names}")

        logging.info(
            "RF-DETR TensorRT I/O: input=%s shape=%s outputs=%s",
            self.input_name,
            self.name_to_shape.get(self.input_name),
            [(n, self.name_to_shape.get(n)) for n in self.output_names],
        )

        # Detect dynamic-batch engine (batch dim == -1) and allocate buffers.
        input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        self._is_dynamic_batch = (len(input_shape) > 0 and input_shape[0] < 0)
        if self._is_dynamic_batch:
            try:
                profile_shapes = self.engine.get_tensor_profile_shape(self.input_name, 0)
                self._max_batch = int(profile_shapes[2][0])
            except Exception:
                self._max_batch = 16
            self._device_buffers = {}
            for name in self.name_to_shape:
                orig = self.name_to_shape[name]
                alloc = tuple(self._max_batch if d < 0 else d for d in orig)
                self._device_buffers[name] = torch.empty(alloc, dtype=self.name_to_torch_dtype[name], device="cuda")
            logging.info("RF-DETR TRT dynamic-batch engine, max_batch=%d", self._max_batch)
        else:
            self._max_batch = 1
            self._device_buffers = {}
            for name, shape in self.name_to_shape.items():
                self._device_buffers[name] = torch.empty(shape, dtype=self.name_to_torch_dtype[name], device="cuda")
        self._input_buffer = self._device_buffers[self.input_name]

    def _execute(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Set addresses, execute TRT, return (dets[n], labels[n]) as float32 numpy."""
        if self._is_dynamic_batch:
            h, w = self._input_buffer.shape[2], self._input_buffer.shape[3]
            self.context.set_input_shape(self.input_name, (n, 3, h, w))
        self.context.set_tensor_address(self.input_name, int(self._input_buffer.data_ptr()))
        for oname in self.output_names:
            self.context.set_tensor_address(oname, int(self._device_buffers[oname].data_ptr()))
        ok = self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed")
        torch.cuda.current_stream().synchronize()
        name_set = set(self.output_names)
        dets_name = "dets" if "dets" in name_set else self.output_names[0]
        labels_name = "labels" if "labels" in name_set else self.output_names[1]
        dets = self._device_buffers[dets_name][:n].detach().cpu().float().numpy()
        labels = self._device_buffers[labels_name][:n].detach().cpu().float().numpy()
        return dets, labels

    def predict_arrays(self, chw01: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """chw01: float32 [1,3,H,W] in [0,1]."""
        x = (chw01.astype(np.float32) - self.means) / self.stds
        x_cpu = torch.from_numpy(x)
        if x_cpu.dtype != self._input_buffer.dtype:
            x_cpu = x_cpu.to(dtype=self._input_buffer.dtype)
        self._input_buffer[:1].copy_(x_cpu, non_blocking=True)
        return self._execute(1)

    def predict_arrays_batch(self, batch_chw01: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run N images in one TRT call (dynamic-batch engine only).
        batch_chw01: float32 [N,3,H,W] in [0,1].
        Returns (dets [N,Q,4], labels [N,Q,C]).
        """
        n = int(batch_chw01.shape[0])
        if n < 1 or n > self._max_batch:
            raise ValueError(f"batch size {n} out of range [1, {self._max_batch}]")
        x = (batch_chw01.astype(np.float32) - self.means) / self.stds
        x_cpu = torch.from_numpy(x)
        if x_cpu.dtype != self._input_buffer.dtype:
            x_cpu = x_cpu.to(dtype=self._input_buffer.dtype)
        self._input_buffer[:n].copy_(x_cpu, non_blocking=True)
        return self._execute(n)


def rfdetr_postprocess_topk(
    pred_logits: np.ndarray,
    pred_boxes: np.ndarray,
    orig_hw: Tuple[int, int],
    num_select: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Numpy/torch hybrid postprocess equivalent to RF-DETR `PostProcess` for the non-mask head.

    pred_logits: [1,Q,C] raw logits
    pred_boxes:  [1,Q,4] cxcywh in [0,1] (RF-DETR)
    orig_hw: (H,W) of the image the boxes should be scaled to (typically model input HW)
    """
    logits_t = torch.from_numpy(pred_logits)
    boxes_t = torch.from_numpy(pred_boxes)

    prob = logits_t.sigmoid()  # [1,Q,C]
    q = prob.shape[1]
    c = prob.shape[2]
    topk_values, topk_indexes = torch.topk(prob.view(1, -1), min(int(num_select), q * c), dim=1)
    scores = topk_values[0].detach().cpu().numpy()
    topk_boxes = (topk_indexes[0] // c).detach().cpu().numpy().astype(np.int64)
    labels = (topk_indexes[0] % c).detach().cpu().numpy().astype(np.int64)

    # cxcywh -> xyxy in normalized coords, then scale to pixels
    from rfdetr.utilities.box_ops import box_cxcywh_to_xyxy

    xyxy_norm = box_cxcywh_to_xyxy(boxes_t[0]).detach().cpu().numpy()  # [Q,4]
    boxes = xyxy_norm[topk_boxes].copy()
    h, w = int(orig_hw[0]), int(orig_hw[1])
    boxes[:, [0, 2]] *= float(w)
    boxes[:, [1, 3]] *= float(h)
    return boxes.astype(np.float32), scores.astype(np.float32), labels


def preprocess_bgr_to_chw01_square(bgr: np.ndarray, side: int) -> np.ndarray:
    """Resize BGR uint8 -> RGB float CHW [0,1] with shape [1,3,S,S]."""
    if bgr is None or bgr.size == 0:
        raise ValueError("empty image")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (int(side), int(side)), interpolation=cv2.INTER_LINEAR)
    x = rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None, ...]
    return x


class RFDETRTrackedWrapper:
    """
    Stage-1 shim: provides `.track(**kwargs)` similar enough for `infer_frame_leaf_grouped_tracked`.

    Internally runs RF-DETR `predict()` then ByteTrack, then returns a list of `_SimpleResult`.
    """

    def __init__(
        self,
        *,
        weights_path: str,
        num_classes: int,
        shape_hw: Tuple[int, int],
        tracker_yaml: str,
        device: str = "cuda",
    ):
        from rfdetr import RFDETRSmall  # local import: keeps import cost down for YOLO-only users

        self._rfdetr = RFDETRSmall(pretrain_weights=weights_path, num_classes=int(num_classes))
        self.shape_hw = (int(shape_hw[0]), int(shape_hw[1]))
        self.device = device
        self._tracker = _byte_track_from_ultralytics_yaml(tracker_yaml, frame_rate=30)

        # Apply the same ONNX-export workaround globally for PyTorch RF-DETR inference:
        # antialiased upsampling ops are not supported by legacy ONNX exporter.
        self._patch_interpolate_for_export_compat()

    @staticmethod
    def _patch_interpolate_for_export_compat() -> None:
        import torch.nn.functional as F

        if getattr(F.interpolate, "_gomicro_rfdetr_patched", False):
            return
        _real = F.interpolate

        def _wrapped(input, size=None, scale_factor=None, mode="nearest", align_corners=None, recompute_scale_factor=None, antialias=False):
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

        _wrapped._gomicro_rfdetr_patched = True  # type: ignore[attr-defined]
        F.interpolate = _wrapped  # type: ignore[assignment]

    def track(self, **kwargs):
        import torch

        src = kwargs.get("source")
        if src is None:
            raise ValueError("RFDETRTrackedWrapper.track requires `source`")
        if not isinstance(src, np.ndarray):
            raise TypeError("RFDETRTrackedWrapper.track expects `source` as a numpy BGR image")

        thr = float(kwargs.get("conf", 0.25))
        shape = kwargs.get("imgsz", None)
        if shape is None:
            h, w = self.shape_hw
        else:
            # allow int or (h,w)
            if isinstance(shape, (list, tuple)) and len(shape) >= 2:
                h, w = int(shape[0]), int(shape[1])
            else:
                h = w = int(shape)

        dets = self._rfdetr.predict(src, threshold=thr, shape=(h, w))
        tracked = self._tracker.update_with_detections(dets)

        if tracked.xyxy is None or len(tracked.xyxy) == 0:
            res = _SimpleResult(names=self._class_names(), boxes=_SimpleBoxes(torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,)), None))
            return [res]

        xyxy = torch.from_numpy(tracked.xyxy).float()
        conf = torch.from_numpy(tracked.confidence).float()
        cls = torch.from_numpy(tracked.class_id).float()
        tid = torch.from_numpy(tracked.tracker_id).float() if tracked.tracker_id is not None else None
        boxes = _SimpleBoxes(xyxy, conf, cls, tid)
        res = _SimpleResult(names=self._class_names(), boxes=boxes)
        return [res]

    def _class_names(self) -> dict:
        # RF-DETR exposes class name list on the wrapper
        try:
            names = self._rfdetr.class_names
            if isinstance(names, dict):
                return {int(k): str(v) for k, v in names.items()}
            if isinstance(names, (list, tuple)):
                return {i: str(n) for i, n in enumerate(names)}
        except Exception:
            pass
        return {}


class RFDETRDefectWrapper:
    """
    Stage-2 shim: provides `.predict(**kwargs)` similar enough for `infer_detect.run_defect_on_leaf`.

    Supports:
    - `.engine` / `.onnx` via ONNXRuntime TensorRT EP (preferred)
    - `.pth` / `.pt` via RF-DETR PyTorch (`RFDETRSmall.predict`)
    """

    def __init__(
        self,
        *,
        weights_or_engine_path: str,
        num_classes: int,
        means: Tuple[float, float, float],
        stds: Tuple[float, float, float],
        device: int = 0,
        class_names: Optional[dict | list | tuple] = None,
    ):
        self.path = weights_or_engine_path
        self.num_classes = int(num_classes)
        self.device = int(device)
        self.means = means
        self.stds = stds
        self._torch_model = None
        self._ort: Optional[RFDETRTensorrtPredictor] = None
        self._class_names = self._normalize_class_names(class_names)

        ext = os.path.splitext(self.path)[1].lower()
        if ext in (".engine", ".onnx"):
            cfg = RFDETRSessionConfig(engine_path=self.path, num_classes=self.num_classes, num_select=300)
            self._ort = RFDETRTensorrtPredictor(cfg, means=means, stds=stds)
        elif ext in (".pth", ".pt"):
            from rfdetr import RFDETRSmall

            RFDETRTrackedWrapper._patch_interpolate_for_export_compat()
            self._torch_model = RFDETRSmall(pretrain_weights=self.path, num_classes=self.num_classes)
        else:
            raise ValueError(f"Unsupported RF-DETR artifact extension: {ext} ({self.path})")

    @property
    def supports_batch(self) -> bool:
        """True when the loaded TRT engine supports dynamic batch > 1."""
        return self._ort is not None and self._ort._is_dynamic_batch

    def runtime_summary(self) -> str:
        """Short string for startup logs (whether TRT or PyTorch path is active)."""
        if self._ort is not None:
            inm = self._ort.input_name
            shp = self._ort.name_to_shape.get(inm, ())
            batch_info = f" dynamic_batch=True max={self._ort._max_batch}" if self._ort._is_dynamic_batch else ""
            return f"TensorRT input={inm} shape={shp}{batch_info}"
        if self._torch_model is not None:
            return "PyTorch RF-DETR weights"
        return "no backend"

    @staticmethod
    def _normalize_class_names(class_names: Optional[dict | list | tuple]) -> dict:
        if isinstance(class_names, dict) and class_names:
            try:
                return {int(k): str(v) for k, v in class_names.items()}
            except Exception:
                return {}
        if isinstance(class_names, (list, tuple)) and class_names:
            return {i: str(v) for i, v in enumerate(class_names)}
        return {}

    def _resolve_names_map(self) -> dict:
        if self._class_names:
            return dict(self._class_names)
        if self._torch_model is not None:
            names = getattr(self._torch_model, "class_names", None)
            if isinstance(names, list):
                return {i: str(n) for i, n in enumerate(names)}
            if isinstance(names, dict):
                return {int(k): str(v) for k, v in names.items()}
        return {int(i): str(i) for i in range(max(1, self.num_classes))}

    def predict(self, **kwargs):
        import torch
        from ultralytics.engine.results import Results

        src = kwargs.get("source")
        if not isinstance(src, np.ndarray):
            raise TypeError("RFDETRDefectWrapper.predict expects `source` as numpy BGR image")
        conf = float(kwargs.get("conf", 0.25))
        imgsz = kwargs.get("imgsz", None)
        side = int(imgsz) if imgsz is not None else 640
        if side % 32 != 0:
            # RF-DETR validates divisibility; snap down to avoid hard failures.
            side = int(max(32, (side // 32) * 32))

        h0, w0 = src.shape[:2]

        if self._ort is not None:
            chw01 = preprocess_bgr_to_chw01_square(src, side)
            dets, labels = self._ort.predict_arrays(chw01)

            # RF-DETR TensorRT exports are commonly already postprocessed with outputs:
            #   dets   -> [1, N, 5] where columns are [x1, y1, x2, y2, score] on the resized input
            #   labels -> [1, N] or [1, N, 1] class indices
            # Some variants can still expose raw logits+boxes; keep a fallback for that path.
            if (
                isinstance(dets, np.ndarray)
                and dets.ndim == 3
                and dets.shape[-1] >= 5
                and isinstance(labels, np.ndarray)
                and labels.ndim in (2, 3)
            ):
                d0 = dets[0]
                xyxy = d0[:, :4].astype(np.float32, copy=False)
                scores = d0[:, 4].astype(np.float32, copy=False)
                if labels.ndim == 3:
                    cls_ids = labels[0, :, 0].astype(np.int64, copy=False)
                else:
                    cls_ids = labels[0].astype(np.int64, copy=False)
            else:
                # Fallback for raw RF-DETR outputs:
                # logits [1,Q,C] + boxes [1,Q,4] (cxcywh normalized).
                xyxy, scores, cls_ids = rfdetr_postprocess_topk(
                    labels, dets, orig_hw=(side, side), num_select=300
                )

            keep = scores >= conf
            xyxy = xyxy[keep]
            scores = scores[keep]
            cls_ids = cls_ids[keep]

            # scale boxes from square network coords -> original crop coords
            sx = float(w0) / float(side)
            sy = float(h0) / float(side)
            xyxy[:, [0, 2]] *= sx
            xyxy[:, [1, 3]] *= sy

            # Defensive cleanup: reject invalid/extreme boxes before passing to drawing/stats.
            xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0.0, max(0.0, float(w0 - 1)))
            xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0.0, max(0.0, float(h0 - 1)))
            w = xyxy[:, 2] - xyxy[:, 0]
            h = xyxy[:, 3] - xyxy[:, 1]
            valid = (w >= 1.0) & (h >= 1.0)
            xyxy = xyxy[valid]
            scores = scores[valid]
            cls_ids = cls_ids[valid]

            boxes_t = torch.from_numpy(np.concatenate([xyxy, scores[:, None], cls_ids[:, None].astype(np.float32)], axis=1))
            names = self._resolve_names_map()
            return [Results(src, path="crop", names=names, boxes=boxes_t)]

        # torch path
        assert self._torch_model is not None
        try:
            dets = self._torch_model.predict(src, threshold=conf, shape=(side, side))
        except Exception as exc:
            # RF-DETR can keep an optimization cache tied to a specific resolution
            # (e.g. optimized for 512x512). If runtime uses a different size, clear
            # the optimized artifact and retry once with the requested shape.
            msg = str(exc).lower()
            can_recover = (
                ("optimized" in msg and "resolution" in msg)
                or ("optimized" in msg and "got" in msg)
                or ("optimized for" in msg)
            )
            if can_recover and hasattr(self._torch_model, "remove_optimized_model"):
                try:
                    self._torch_model.remove_optimized_model()
                    dets = self._torch_model.predict(src, threshold=conf, shape=(side, side))
                except Exception:
                    raise exc
            else:
                raise
        if dets.xyxy is None or len(dets.xyxy) == 0:
            names = self._resolve_names_map()
            empty = torch.zeros((0, 6), dtype=torch.float32)
            return [Results(src, path="crop", names=names, boxes=empty)]

        xyxy = torch.from_numpy(dets.xyxy).float()
        scores = torch.from_numpy(dets.confidence).float()
        cls_ids = torch.as_tensor(dets.class_id, dtype=torch.float32)
        boxes_t = torch.cat([xyxy, scores[:, None], cls_ids[:, None]], dim=1)
        names_map = self._resolve_names_map()
        return [Results(src, path="crop", names=names_map, boxes=boxes_t)]

    def predict_batch(self, crops: List[np.ndarray], conf: float = 0.25, imgsz: Optional[int] = None) -> List:
        """
        Run defect inference on N BGR crops in one TRT call.
        Returns a list of Results lists (one per crop), same format as predict().
        Falls back to sequential if engine is not dynamic-batch.
        """
        from ultralytics.engine.results import Results

        if not crops:
            return []
        if not self.supports_batch:
            return [self.predict(source=c, conf=conf, imgsz=imgsz) for c in crops]

        side = int(imgsz) if imgsz is not None else 640
        if side % 32 != 0:
            side = int(max(32, (side // 32) * 32))

        max_b = self._ort._max_batch
        names = self._resolve_names_map()
        results_all: List = []

        # Process in sub-batches if N > max_batch
        for start in range(0, len(crops), max_b):
            batch_crops = crops[start:start + max_b]
            n = len(batch_crops)
            hw0s = [(c.shape[0], c.shape[1]) for c in batch_crops]

            chw01_list = []
            for crop in batch_crops:
                chw01_list.append(preprocess_bgr_to_chw01_square(crop, side)[0])
            batch_chw01 = np.stack(chw01_list, axis=0)  # [n, 3, side, side]

            dets_batch, labels_batch = self._ort.predict_arrays_batch(batch_chw01)

            for idx, (h0, w0) in enumerate(hw0s):
                crop = batch_crops[idx]
                d0 = dets_batch[idx]
                l0 = labels_batch[idx]

                if d0.ndim == 2 and d0.shape[-1] >= 5 and l0.ndim in (1, 2):
                    xyxy = d0[:, :4].astype(np.float32, copy=False)
                    scores = d0[:, 4].astype(np.float32, copy=False)
                    cls_ids = (l0[:, 0] if l0.ndim == 2 else l0).astype(np.int64, copy=False)
                else:
                    xyxy, scores, cls_ids = rfdetr_postprocess_topk(
                        l0[None], d0[None], orig_hw=(side, side), num_select=300
                    )

                keep = scores >= conf
                xyxy = xyxy[keep]; scores = scores[keep]; cls_ids = cls_ids[keep]
                sx = float(w0) / float(side)
                sy = float(h0) / float(side)
                xyxy[:, [0, 2]] *= sx
                xyxy[:, [1, 3]] *= sy
                xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0.0, max(0.0, float(w0 - 1)))
                xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0.0, max(0.0, float(h0 - 1)))
                ww = xyxy[:, 2] - xyxy[:, 0]; hh = xyxy[:, 3] - xyxy[:, 1]
                valid = (ww >= 1.0) & (hh >= 1.0)
                xyxy = xyxy[valid]; scores = scores[valid]; cls_ids = cls_ids[valid]

                if len(xyxy):
                    boxes_t = torch.from_numpy(
                        np.concatenate([xyxy, scores[:, None], cls_ids[:, None].astype(np.float32)], axis=1)
                    )
                else:
                    boxes_t = torch.zeros((0, 6), dtype=torch.float32)
                results_all.append([Results(crop, path="crop", names=names, boxes=boxes_t)])

        return results_all
