"""
Inference entrypoint — the main file that server.py imports.

This file contains:
  1. CONFIG: Loading/caching infer_config.json (model paths, thresholds, colors).
  2. MODELS: Loading/unloading YOLO TensorRT models to/from GPU memory.
  3. PIPELINE: The main infer_frame_leaf_grouped_tracked() function that
     orchestrates the full two-stage inference pipeline per frame.

How server.py uses this:
    import infer_main as infer
    infer.load_models()                           # load TensorRT engines
    annotated, summary = infer.infer_frame_leaf_grouped_tracked(...)  # per frame
    infer.unload_models(main, defect)             # free GPU memory

File structure:
    infer_main.py   <- YOU ARE HERE (config + models + pipeline entry point)
    infer_detect.py <- Detection logic (extract, filter, dedup, defect-on-leaf)
    infer_draw.py   <- Drawing (boxes, masks, defects) + tracking stats
"""

import json
import logging
import os
from typing import Optional

import cv2
import torch
import gc
from ultralytics import YOLO

import config_store
from stem_detector import detect_stem

from infer_detect import (
    LEAF_STEMS_OPENCV_MARKER,
    class_name,
    get_class_config,
    get_leaf_aliases,
    get_main_defect_aliases,
    get_stem_aliases,
    _build_alias_map,
    resolve_class,
    extract_main_detections,
    dedup_leaf_boxes,
    dedup_indices_keep_biggest_overlap,
    mask_to_box,
    poly_to_box,
    tighten_boxes_with_masks,
    filter_indices_by_min_hw,
    filter_indices_by_roi_overlap,
    roi_bounds_from_frame,
    run_defect_on_leaf,
    extract_masked_leaf_crop,
)
from infer_rfdetr import RFDETRDefectWrapper, RFDETRTrackedWrapper
from infer_draw import (
    draw_main_boxes,
    draw_main_masks,
    draw_defects,
    update_tracking_stats,
    compute_tracking_summary,
    compute_size_summary,
    draw_size_debug,
)

import time as _time
import psutil as _psutil

# Debug print state — memory printed once per second, frame counter per session
_dbg_last_mem_ts: float = 0.0
_dbg_frame_count: int = 0
_dbg_proc = _psutil.Process()
_dbg_log_path = os.path.join(os.path.dirname(__file__), "infer_debug.log")
_dbg_log_file = open(_dbg_log_path, "w", buffering=1)

# Optional monitor hook — set via set_monitor() so all debug/detection lines
# also appear in monitor.log alongside the system-stats rows.
_monitor_log_fn = None


def set_monitor(fn):
    """Pass SmartMonitor.log (or any callable) to route all infer logs to monitor.log."""
    global _monitor_log_fn
    _monitor_log_fn = fn


def _dbg_print(msg: str):
    # Keep stdout quiet during runtime; logs are persisted to files/monitor.
    try:
        _dbg_log_file.write(msg + "\n")
    except Exception:
        pass
    if _monitor_log_fn is not None:
        try:
            _monitor_log_fn(msg)
        except Exception:
            pass


def _dbg_stats_sizes(stats) -> str:
    """Return compact stats container sizes for debug logs."""
    if not isinstance(stats, dict):
        return "tracks=- defsets=- defcnt=- m2sizes=-"
    tracks = stats.get("tracks", {})
    defsets = stats.get("leaf_defects", {})
    defcnt = stats.get("leaf_defect_counts", {})
    m2sizes = stats.get("model2_track_sizes", {})
    try:
        return (
            f"tracks={len(tracks) if isinstance(tracks, dict) else -1} "
            f"defsets={len(defsets) if isinstance(defsets, dict) else -1} "
            f"defcnt={len(defcnt) if isinstance(defcnt, dict) else -1} "
            f"m2sizes={len(m2sizes) if isinstance(m2sizes, dict) else -1}"
        )
    except Exception:
        return "tracks=- defsets=- defcnt=- m2sizes=-"


def _dbg_mem_rss_mb() -> int:
    try:
        return int(_dbg_proc.memory_info().rss / (1024 * 1024))
    except Exception:
        return -1


def _dbg_mem_ram_str() -> str:
    try:
        vm = _psutil.virtual_memory()
        used = int(vm.used / (1024 * 1024))
        total = int(vm.total / (1024 * 1024))
        pct = vm.percent
        return f"ram={used}/{total}MB({pct:.0f}%)"
    except Exception:
        return "ram=?MB"


# ===========================================================================
# CONFIGURATION
# ===========================================================================
# Reads infer_config.json which controls model paths, confidence thresholds,
# box colors, debug logging, and vote thresholds.

DEFAULT_CONFIG = {
    "models": {
        "main": "models/best.engine",
        "defect": "models/defect.engine",
    },
    "main": {
        "backend": "yolo",
        "model_type": "detect",
        "render": "box",
        "conf_leaf": 0.7,
        "box_thickness": 4,
        "mask_alpha": 0.35,
        "roi_enabled": False,
        "roi_margin_pct": 0,
        "roi_min_overlap": 0.5,
        "roi_filter_mode": "overlap",
        "roi_debug_draw": False,
        "roi_debug_color": [200, 200, 200],
        "roi_debug_thickness": 1,
    },
    "defect": {
        "backend": "rfdetr",
        "conf_stem": 0.30,
        "conf_other": 0.30,
        # Stem length when server stem_size is enabled: "model" needs a stem class in
        # class_config defect_model (role ignore); else use "opencv" (stem_detector.py).
        "stem_size_source": "opencv",
        "opencv_stem_min_length_px": 12,
        "debug_log_detections": False,
        "debug_log_every_n_frames": 15,
        "debug_log_raw_candidates": False,
        "debug_log_filter_reasons": False,
        "box_thickness": 2,
        "vote_thresholds": {
            "min_count": 1,
        },
    },
    "warmup_main_image": "",
    "warmup_defect_image": "",
}


def load_config():
    """
    Load infer_config.json from disk, merging with defaults.

    Why merge: If the user adds a new config key in a code update, old config
    files on deployed Jetsons won't have it. Merging ensures defaults fill gaps.

    If the file doesn't exist, creates it with DEFAULT_CONFIG.

    Returns:
        Dict with all config keys guaranteed to exist.
    """
    data = config_store.load_infer_config()
    if not isinstance(data, dict):
        data = {}
    merged = DEFAULT_CONFIG.copy()
    merged.update(data)
    # Ensure unified and legacy files both contain the merged shape.
    save_config(merged)
    return merged


def save_config(config):
    """Write config dict through the central config store."""
    config_store.save_infer_config(config)


# File-change-aware config cache.
# Why: get_config() is called every frame during inference. Reading the file
# every frame would be wasteful. Instead we cache and only reload when the
# file's mtime changes (allowing live config edits without restart).
_cfg_cache = None
_cfg_mtime: float = 0.0


def get_config():
    """
    Return the cached config, reloading only if infer_config.json was modified.

    Why this exists:
    During inference, this is called every single frame to read thresholds,
    colors, etc. File I/O every frame (30fps) would be wasteful. The mtime
    check is a single stat() syscall which is essentially free.

    Returns:
        Config dict (same as load_config output).
    """
    global _cfg_cache, _cfg_mtime
    try:
        mtime = config_store.infer_config_mtime()
    except Exception:
        return _cfg_cache if _cfg_cache is not None else load_config()
    if _cfg_cache is None or mtime != _cfg_mtime:
        _cfg_cache = load_config()
        _cfg_mtime = mtime
    return _cfg_cache


# ===========================================================================
# MODEL LOADING / UNLOADING
# ===========================================================================

# Read model paths and type from config at import time (also refreshed via get_config in load_models).
_CFG = load_config()
ENGINE_PATH = _CFG.get("models", {}).get("main", "models/best.engine")
DEFECT_ENGINE_PATH = _CFG.get("models", {}).get("defect", "models/defect.engine")
MAIN_MODEL_TYPE = str(_CFG.get("main", {}).get("model_type", "detect")).strip().lower()

_SERVER2_DIR = os.path.dirname(os.path.abspath(__file__))


def _abs_model_path(path: str) -> str:
    """Resolve model path relative to Server2/ so loading works regardless of process cwd."""
    p = (path or "").strip()
    if not p:
        raise FileNotFoundError("empty model path in infer_config.json")
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(_SERVER2_DIR, p))


def _infer_backend_from_path(path: str) -> str:
    ext = os.path.splitext((path or "").lower())[1]
    # Heuristic only used when config doesn't specify backend.
    if ext in (".engine", ".xml"):
        return "yolo"
    if ext == ".pt":
        return "yolo"
    if ext == ".pth":
        return "rfdetr"
    # `.onnx` is ambiguous (YOLO ONNX vs RF-DETR ONNX). Require explicit `backend` in config.
    return ""


def _normalize_backend(name: str | None) -> str:
    b = str(name or "").strip().lower()
    if b in ("yolo", "ultralytics", "yolov8", "yolo11"):
        return "yolo"
    if b in ("rfdetr", "rf-detr", "rf_detr"):
        return "rfdetr"
    return b


def load_model(engine_path: str = ENGINE_PATH, model_type: str = MAIN_MODEL_TYPE):
    """
    Load a single YOLO model from a TensorRT engine file.

    Why TensorRT: On Jetson, TensorRT engines are 5-10x faster than PyTorch.
    The .engine files are pre-compiled from .pt weights using ultralytics export.

    Args:
        engine_path: Path to the .engine file.
        model_type: "detect" for object detection, "seg" for instance segmentation.

    Returns:
        YOLO model object ready for inference.
    """
    if str(model_type).strip().lower() == "seg":
        return YOLO(engine_path, task="segment")
    return YOLO(engine_path, task="detect")


def load_models(
    main_path: Optional[str] = None,
    defect_path: Optional[str] = None,
    main_type: Optional[str] = None,
):
    """
    Load both the main (leaf detection) and defect models into GPU memory.

    Paths default from the latest infer_config.json (get_config), resolved under Server2/.

    Args:
        main_path: Path to the main model .engine file (optional).
        defect_path: Path to the defect model .engine file (optional).
        main_type: "detect" or "seg" for the main model (optional).

    Returns:
        Tuple of (main_model, defect_model) YOLO objects.
    """
    cfg = get_config()
    if main_path is None:
        main_path = cfg.get("models", {}).get("main", ENGINE_PATH)
    if defect_path is None:
        defect_path = cfg.get("models", {}).get("defect", DEFECT_ENGINE_PATH)
    if main_type is None:
        main_type = str(cfg.get("main", {}).get("model_type", MAIN_MODEL_TYPE)).strip().lower()

    main_cfg = cfg.get("main", {}) if isinstance(cfg.get("main"), dict) else {}
    defect_cfg = cfg.get("defect", {}) if isinstance(cfg.get("defect"), dict) else {}

    main_backend = (
        _normalize_backend(main_cfg.get("backend"))
        or _infer_backend_from_path(str(main_path))
        or "yolo"
    )
    defect_backend = (
        _normalize_backend(defect_cfg.get("backend"))
        or _infer_backend_from_path(str(defect_path))
        or "yolo"
    )

    main_abs = _abs_model_path(main_path)
    defect_abs = _abs_model_path(defect_path)
    for label, abs_p in (("main", main_abs), ("defect", defect_abs)):
        if not os.path.isfile(abs_p):
            raise FileNotFoundError(
                f"{label} model not found: {abs_p} (check infer_config.json models.{label})"
            )

    if main_backend == "rfdetr":
        if str(main_type).strip().lower() == "seg":
            raise ValueError("main.backend=rfdetr does not support model_type=seg in this integration")
        imgsz = main_cfg.get("imgsz", None)
        try:
            side = int(imgsz) if imgsz is not None else 640
        except Exception:
            side = 640
        if side % 32 != 0:
            side = int(max(32, (side // 32) * 32))
        num_classes = int(main_cfg.get("rfdetr_num_classes", main_cfg.get("num_classes", 90)))
        main_model = RFDETRTrackedWrapper(
            weights_path=main_abs,
            num_classes=num_classes,
            shape_hw=(side, side),
            tracker_yaml=config_store.TRACKER_CONFIG_PATH,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    else:
        main_model = YOLO(
            main_abs, task="segment" if str(main_type).strip().lower() == "seg" else "detect"
        )

    if defect_backend == "rfdetr":
        num_classes = int(defect_cfg.get("rfdetr_num_classes", defect_cfg.get("num_classes", 90)))
        cc = get_class_config()
        defect_section = cc.get("defect_model", {})
        configured_names = [k for k in defect_section.keys() if not str(k).startswith("_")]
        # Enforce stable semantic class order matching training order, not JSON key order.
        if int(num_classes) == 6:
            preferred_order = ["mud", "bruise", "ipd", "whitespot", "torn", "leaf stem"]
            if all(name in defect_section for name in preferred_order):
                configured_names = preferred_order
        elif int(num_classes) == 4:
            preferred_order = ["mud", "bruise", "ipd", "whitespot"]
            if all(name in defect_section for name in preferred_order):
                configured_names = preferred_order
        if configured_names:
            configured_names = configured_names[: max(1, num_classes)]
        # RF-DETR normalization constants are stable across RF-DETRSmall; keep explicit for ORT path.
        means = (0.485, 0.456, 0.406)
        stds = (0.229, 0.224, 0.225)
        defect_model = RFDETRDefectWrapper(
            weights_or_engine_path=defect_abs,
            num_classes=num_classes,
            means=means,
            stds=stds,
            device=0 if torch.cuda.is_available() else -1,
            class_names=configured_names if configured_names else None,
        )
    else:
        defect_model = YOLO(defect_abs, task="detect")

    logging.info(
        "Models constructed: main=%s (%s) path=%s | defect=%s (%s) path=%s",
        type(main_model).__name__,
        main_backend,
        main_abs,
        type(defect_model).__name__,
        defect_backend,
        defect_abs,
    )
    if hasattr(defect_model, "runtime_summary"):
        logging.info("Defect runtime: %s", defect_model.runtime_summary())

    return main_model, defect_model


def unload_models(main_model, defect_model):
    """
    Completely unload models from GPU memory including TensorRT caches.

    Why aggressive cleanup:
    On Jetson with limited GPU memory (4-8GB shared with CPU), failing to
    fully release model memory causes OOM errors when reloading models.
    Simple `del model` doesn't release TensorRT engine memory. We need to:
    1. Clear the YOLO predictor's internal model reference.
    2. Move tensors to CPU (releases GPU allocations).
    3. Delete the Python objects.
    4. Run gc.collect() multiple times (Python GC is generational).
    5. Call torch.cuda.empty_cache() to release the CUDA allocator's pool.

    Args:
        main_model: YOLO main model (or None).
        defect_model: YOLO defect model (or None).
    """
    try:
        if torch.cuda.is_available():
            mem_before = torch.cuda.memory_allocated() / (1024**2)
            _dbg_print(f"[Model Unload] GPU memory before cleanup: {mem_before:.1f} MiB")

        def _destroy_trt_engine(backend, name):
            """Destroy TensorRT engine and context inside an AutoBackend to free GPU memory."""
            if backend is None:
                return
            try:
                # AutoBackend stores TRT objects as:
                #   backend.context  = execution context
                #   backend.model    = the TRT engine (ICudaEngine)
                #   backend.runtime  = TRT runtime
                #   backend.bindings = allocated GPU tensors
                # Order matters: context first, then engine, then runtime.
                for attr in ('context', 'model', 'runtime'):
                    obj = getattr(backend, attr, None)
                    if obj is not None:
                        try:
                            setattr(backend, attr, None)
                        except Exception:
                            pass
                # Clear GPU tensor bindings
                for attr in ('bindings', 'binding_addrs', 'output_shapes', 'fp16',
                             'dynamic', 'is_trt10'):
                    if hasattr(backend, attr):
                        try:
                            setattr(backend, attr, None)
                        except Exception:
                            pass
            except Exception as e:
                _dbg_print(f"[Model Unload] TRT cleanup error for {name}: {e}")

        def _clear_model(model, name):
            """Clear a single YOLO model's internal references."""
            if model is None:
                return
            try:
                # RF-DETR wrappers (non-YOLO)
                if model.__class__.__name__ in ("RFDETRTrackedWrapper", "RFDETRDefectWrapper"):
                    try:
                        if hasattr(model, "_ort") and getattr(model, "_ort", None) is not None:
                            ort = getattr(model, "_ort")
                            sess = getattr(ort, "session", None)
                            if sess is not None:
                                try:
                                    del sess
                                except Exception:
                                    pass
                            setattr(model, "_ort", None)
                    except Exception:
                        pass
                    try:
                        if hasattr(model, "_torch_model") and getattr(model, "_torch_model", None) is not None:
                            setattr(model, "_torch_model", None)
                    except Exception:
                        pass
                    try:
                        if hasattr(model, "_rfdetr") and getattr(model, "_rfdetr", None) is not None:
                            setattr(model, "_rfdetr", None)
                    except Exception:
                        pass
                    return

                # Destroy TensorRT engine inside predictor's model
                if hasattr(model, 'predictor') and model.predictor is not None:
                    if hasattr(model.predictor, 'model') and model.predictor.model is not None:
                        _destroy_trt_engine(model.predictor.model, f"{name}.predictor.model")
                        model.predictor.model = None
                    model.predictor = None
                # Destroy TensorRT engine inside model.model
                if hasattr(model, 'model') and model.model is not None:
                    _destroy_trt_engine(model.model, f"{name}.model")
                    if hasattr(model.model, 'cpu'):
                        try:
                            model.model.cpu()
                        except Exception:
                            pass
                    model.model = None
                if hasattr(model, 'session'):
                    model.session = None
                # Clear any YOLO-level caches
                if hasattr(model, 'ckpt'):
                    model.ckpt = None
            except Exception as e:
                _dbg_print(f"[Model Unload] Error clearing {name}: {e}")

        _clear_model(main_model, "main_model")
        del main_model
        _clear_model(defect_model, "defect_model")
        del defect_model

        # Multiple GC passes because Python uses generational garbage collection
        for _ in range(3):
            gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            mem_after = torch.cuda.memory_allocated() / (1024**2)
            _dbg_print(f"[Model Unload] GPU memory after cleanup: {mem_after:.1f} MiB "
                       f"(freed ~{mem_before - mem_after:.1f} MiB)")

        _dbg_print("[Model Unload] Cleanup complete: models destroyed, GPU memory freed")
    except Exception as e:
        _dbg_print(f"[Model Unload Error] {e}")


# ===========================================================================
# DEFAULT CALIBRATION CONSTANTS
# ===========================================================================
# These are exported so server.py can use them as defaults for size config.

# Calibration: how many centimeters one pixel represents.
# Must be calibrated for your specific camera setup and distance.
CM_PER_PX = 0.01

# Size thresholds for leaf quality checks (in centimeters).
# Leaves outside these ranges are marked "out of spec".
SIZE_THRESHOLDS = {
    "leaf": {
        "length": {"min": 10.0, "max": 12.0},
        "width": {"min": 4.0, "max": 6.0},
    },
    "stem": {
        "length": {"min": 1.0, "max": 4.0},
    },
}


# ===========================================================================
# MAIN INFERENCE PIPELINE
# ===========================================================================

def infer_frame_leaf_grouped_tracked(
    main_model,
    defect_model,
    frame_bgr,
    stats,
    imgsz: int | None = None,
    conf: float = 0.45,
    device: int = 0,
    size_cfg=SIZE_THRESHOLDS,
    cm_per_px: float = CM_PER_PX,
    stem_size: bool = False,
    size_debug: bool = False,
    size_debug_color: tuple = (255, 150, 0),
    max_full_leaf: int = 20,
):
    """
    Run the full two-stage inference pipeline on a single frame.

    This is the main function called by server.py's pipeline_loop every frame.
    It orchestrates the entire detection → tracking → defect → drawing → stats flow.

    ┌─────────────────────────────────────────────────────────────┐
    │  STAGE 1: Main model (leaf detection + tracking)            │
    │  - model.track() with ByteTrack for persistent IDs          │
    │  - Extract detections → filter to leaves only               │
    │  - Dedup overlapping boxes (IoU + containment)              │
    │  - If seg model: tighten boxes using mask outlines          │
    ├─────────────────────────────────────────────────────────────┤
    │  RENDERING: Draw leaf boxes or masks on frame               │
    ├─────────────────────────────────────────────────────────────┤
    │  STAGE 2: Defect model (per-leaf crop)                      │
    │  - For each detected leaf:                                   │
    │    - Crop leaf region, apply mask                            │
    │    - Run defect model → filter by conf/area                 │
    │    - Update vote counts per (leaf_id, defect_type)          │
    │    - Draw defects only if vote count >= min_count           │
    │    - Red box if defected, green "G" if healthy              │
    ├─────────────────────────────────────────────────────────────┤
    │  STATS: Update tracking stats → compute summary             │
    │  - Leaf count, defect count, size spec summary              │
    └─────────────────────────────────────────────────────────────┘

    Args:
        main_model: YOLO model for leaf detection (with tracking enabled).
        defect_model: YOLO model for defect detection on leaf crops.
        frame_bgr: Input frame as numpy array (H, W, 3) in BGR format.
        stats: Mutable dict for tracking state across frames. Pass {} initially.
               This dict grows as it accumulates track data, defect votes, etc.
        imgsz: Override inference input size for both models (None = per-model config/default).
        conf: Base confidence threshold (overridden per-type by config).
        device: GPU device index (0 for single-GPU Jetson).
        size_cfg: Dict with leaf/stem size thresholds for quality checks.
        cm_per_px: Calibration scalar (centimeters per pixel).
        stem_size: If True, include stem length in the in-spec/out-spec decision.

    Returns:
        Tuple of (annotated_frame, summary_dict):
        - annotated_frame: BGR frame with all boxes/masks/defects drawn.
        - summary_dict: Dict with total_leaf, total_defects, defect_counts,
                        size_summary, etc. (sent to frontend via /infer/stats).
    """
    # -----------------------------------------------------------------------
    # Read config (cached, reloads only if file changed)
    # -----------------------------------------------------------------------
    cfg = get_config()
    main_cfg = cfg.get("main", {})
    defect_cfg = cfg.get("defect", {})
    def _parse_imgsz(val):
        try:
            if val is None:
                return None
            v = int(val)
            return v if v > 0 else None
        except (TypeError, ValueError):
            return None
    # Priority:
    # 1) function arg imgsz (forces both models)
    # 2) infer_config per-model values (main.imgsz / defect.imgsz)
    # 3) engine default
    if imgsz is not None:
        main_imgsz = _parse_imgsz(imgsz)
        defect_imgsz = _parse_imgsz(imgsz)
    else:
        main_imgsz = _parse_imgsz(main_cfg.get("imgsz"))
        defect_imgsz = _parse_imgsz(defect_cfg.get("imgsz"))
    main_model_type = str(main_cfg.get("model_type", "detect")).strip().lower()
    main_thickness = int(main_cfg.get("box_thickness", 2))
    main_render = str(main_cfg.get("render", "box")).strip().lower()
    mask_alpha = float(main_cfg.get("mask_alpha", 0.35))
    defect_thickness = int(defect_cfg.get("box_thickness", 2))
    conf_leaf = float(main_cfg.get("conf_leaf", conf))
    debug_log = bool(defect_cfg.get("debug_log_detections", False))
    debug_every_n = int(defect_cfg.get("debug_log_every_n_frames", 15))
    if debug_every_n <= 0:
        debug_every_n = 1
    debug_log_raw = bool(defect_cfg.get("debug_log_raw_candidates", False))
    debug_log_drop = bool(defect_cfg.get("debug_log_filter_reasons", False))
    vote_cfg = defect_cfg.get("vote_thresholds", {})
    vote_min_count = int(vote_cfg.get("min_count", 5))
    stem_size_source = str(defect_cfg.get("stem_size_source", "model")).strip().lower()
    opencv_stem_min_px = int(defect_cfg.get("opencv_stem_min_length_px", 12))
    m2_size_enabled = bool(defect_cfg.get("size_measure_enabled", False))
    _m2_size_cls_raw = defect_cfg.get("size_measure_classes", [])
    if isinstance(_m2_size_cls_raw, str):
        _m2_size_cls = [x.strip().lower() for x in _m2_size_cls_raw.split(",") if x.strip()]
    elif isinstance(_m2_size_cls_raw, (list, tuple)):
        _m2_size_cls = [str(x).strip().lower() for x in _m2_size_cls_raw if str(x).strip()]
    else:
        _m2_size_cls = []
    m2_size_class_set = set(_m2_size_cls)
    m2_size_all = "all" in m2_size_class_set
    m2_size_cfg = defect_cfg.get("size_measure_thresholds_cm", {})
    m2_size_logic = str(defect_cfg.get("size_final_logic", "auto")).strip().lower()
    roi_enabled = bool(main_cfg.get("roi_enabled", False))
    roi_debug_draw = bool(main_cfg.get("roi_debug_draw", False))
    try:
        roi_margin_pct = float(main_cfg.get("roi_margin_pct", 0))
    except (TypeError, ValueError):
        roi_margin_pct = 0.0
    try:
        roi_min_overlap = float(main_cfg.get("roi_min_overlap", 0.5))
    except (TypeError, ValueError):
        roi_min_overlap = 0.5
    roi_filter_mode = str(main_cfg.get("roi_filter_mode", "overlap")).strip().lower()
    _rdc = main_cfg.get("roi_debug_color", [200, 200, 200])
    if isinstance(_rdc, (list, tuple)) and len(_rdc) >= 3:
        roi_debug_color = tuple(max(0, min(255, int(_rdc[i]))) for i in range(3))
    else:
        roi_debug_color = (200, 200, 200)
    try:
        roi_debug_thickness = max(1, min(4, int(main_cfg.get("roi_debug_thickness", 1))))
    except (TypeError, ValueError):
        roi_debug_thickness = 1

    cc = get_class_config()
    defect_section = cc.get("defect_model", {})
    defect_alias_map = _build_alias_map(defect_section)

    def _bgr_tuple(val):
        if isinstance(val, (list, tuple)) and len(val) >= 3:
            return tuple(max(0, min(255, int(val[i]))) for i in range(3))
        return None

    # Draw colors: class_config is canonical; infer_config main.colors / defect.colors optional overrides only.
    main_colors = {}
    main_section = cc.get("main_model", {})
    main_default_entry = main_section.get("_default", {})
    for _key, _entry in main_section.items():
        if _key.startswith("_"):
            continue
        _c = _bgr_tuple(_entry.get("color"))
        if _c:
            main_colors[_key] = _c
            for _alias in _entry.get("aliases", []):
                main_colors[_alias.strip().lower()] = _c
    main_alias_map = _build_alias_map(main_section)
    # Model1 -> Model2 routing classes (from model1 config).
    # Examples:
    #   ["all"]                -> all model1 classes go to stage-2
    #   ["good","torn","cut"]  -> only those classes go to stage-2
    _m2_pass_raw = main_cfg.get("model2_pass_classes", ["all"])
    if isinstance(_m2_pass_raw, str):
        _m2_pass = [x.strip().lower() for x in _m2_pass_raw.split(",") if x.strip()]
    elif isinstance(_m2_pass_raw, (list, tuple)):
        _m2_pass = [str(x).strip().lower() for x in _m2_pass_raw if str(x).strip()]
    else:
        _m2_pass = ["all"]
    _m2_pass_set = set(_m2_pass) if _m2_pass else {"all"}
    _m2_pass_all = "all" in _m2_pass_set
    _obj_raw = main_cfg.get("object_classes", ["all"])
    if isinstance(_obj_raw, str):
        _obj_list = [x.strip().lower() for x in _obj_raw.split(",") if x.strip()]
    elif isinstance(_obj_raw, (list, tuple)):
        _obj_list = [str(x).strip().lower() for x in _obj_raw if str(x).strip()]
    else:
        _obj_list = ["all"]
    _obj_set = set(_obj_list) if _obj_list else {"all"}
    _obj_all = "all" in _obj_set
    _main_def_raw = main_cfg.get("main_defect_classes", [])
    if isinstance(_main_def_raw, str):
        _main_def_list = [x.strip().lower() for x in _main_def_raw.split(",") if x.strip()]
    elif isinstance(_main_def_raw, (list, tuple)):
        _main_def_list = [str(x).strip().lower() for x in _main_def_raw if str(x).strip()]
    else:
        _main_def_list = []
    _main_def_set_cfg = set(_main_def_list)
    _main_defect_keys_runtime = _main_def_set_cfg if _main_def_set_cfg else get_main_defect_aliases()
    _m1_size_raw = main_cfg.get("size_measure_classes", ["all"])
    if isinstance(_m1_size_raw, str):
        _m1_size_list = [x.strip().lower() for x in _m1_size_raw.split(",") if x.strip()]
    elif isinstance(_m1_size_raw, (list, tuple)):
        _m1_size_list = [str(x).strip().lower() for x in _m1_size_raw if str(x).strip()]
    else:
        _m1_size_list = ["all"]
    _m1_size_set = set(_m1_size_list) if _m1_size_list else {"all"}
    _m1_size_all = "all" in _m1_size_set
    # For model1, use the lowest class confidence at model call so YOLO doesn't
    # pre-drop detections before per-class confidence filters are applied.
    main_track_conf = conf_leaf
    for _key, _entry in main_section.items():
        if _key.startswith("_"):
            continue
        _c = float(_entry.get("confidence", main_default_entry.get("confidence", conf_leaf)))
        if _c < main_track_conf:
            main_track_conf = _c
    _main_ov = main_cfg.get("colors")
    if isinstance(_main_ov, dict):
        for _k, _v in _main_ov.items():
            _t = _bgr_tuple(_v)
            if _t:
                main_colors[_k] = _t

    defect_colors = {}
    _def_ov = defect_cfg.get("colors")
    if isinstance(_def_ov, dict):
        for _k, _v in _def_ov.items():
            _t = _bgr_tuple(_v)
            if _t:
                defect_colors[_k] = _t

    _min_area_cfg = defect_cfg.get("min_area_px")
    if isinstance(_min_area_cfg, dict) and len(_min_area_cfg) > 0:
        min_area_px = _min_area_cfg
    else:
        min_area_px = None

    def _normalize_defect_key(name: str) -> tuple:
        """Map defect names to (config_key, label) using class_config.json.
        config_key = full name e.g. 'yellow spots', label = short e.g. 'YL'."""
        n = str(name).strip().lower()
        cls_key = defect_alias_map.get(n, n)
        entry = defect_section.get(cls_key, {})
        return cls_key, entry.get("label", cls_key)

    # -----------------------------------------------------------------------
    # STAGE 1: Run main model — with or without ByteTrack tracking
    # -----------------------------------------------------------------------
    _tracking_enabled = bool(cfg.get("tracking_enabled", False))

    base_kwargs = {
        "source": frame_bgr,
        "verbose": False,
        "conf": main_track_conf,
        "iou": 0.8,
        "device": device,
    }
    if main_model_type == "seg":
        base_kwargs["retina_masks"] = True
    if main_imgsz is not None:
        base_kwargs["imgsz"] = main_imgsz

    with torch.inference_mode():
        if _tracking_enabled:
            base_kwargs["persist"] = True
            base_kwargs["tracker"] = config_store.TRACKER_CONFIG_PATH
            results = main_model.track(**base_kwargs)
        else:
            results = main_model.predict(**base_kwargs)
    result = results[0]

    # Extract detections into numpy arrays
    xyxy, confs, cls_ids, track_ids, names, masks, masks_xy = extract_main_detections(result)

    # ── Debug prints ────────────────────────────────────────────────────────
    global _dbg_last_mem_ts, _dbg_frame_count
    _dbg_frame_count += 1
    _now = _time.time()
    _rss = _dbg_mem_rss_mb()
    _ram = _dbg_mem_ram_str()
    _dbg_frame_log = bool(debug_log) and (_dbg_frame_count % max(1, int(debug_every_n)) == 0)
    # Memory once per second + periodic CUDA cache flush.
    # On Jetson, CUDA memory is shared system RAM. PyTorch caches tensors after each
    # inference call and never returns them to the OS between frames. Without explicit
    # flushing the cache grows ~30-120MB/frame and eventually triggers the RAM kill.
    # empty_cache() returns all unused cached blocks back to the OS; active tensors
    # (model weights, current-frame outputs) are NOT affected.
    if (_now - _dbg_last_mem_ts) >= 1.0:
        _swap = int(_psutil.swap_memory().used / (1024 * 1024))
        _sizes = _dbg_stats_sizes(stats)
        _dbg_print(f"[MEM] frame={_dbg_frame_count} rss={_rss}MB {_ram} swap={_swap}MB {_sizes}")
        _dbg_last_mem_ts = _now
    _gc_every_n = max(1, int(cfg.get("gc_every_n_frames", 10)))
    if _dbg_frame_count % _gc_every_n == 0:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    # Model 1 detailed logs are optional to avoid per-frame I/O overhead.
    if _dbg_frame_log:
        if cls_ids is not None and len(cls_ids) > 0:
            _m1_items = []
            for _i in range(len(cls_ids)):
                _cname = str(names.get(int(cls_ids[_i]), cls_ids[_i]))
                _conf = float(confs[_i]) if confs is not None else 0.0
                _box = [int(v) for v in xyxy[_i]] if xyxy is not None else []
                _tid = int(track_ids[_i]) if track_ids is not None else -1
                _m1_items.append(f"{_cname}({_conf:.2f}) tid={_tid} box={_box}")
            _dbg_print(f"[M1] frame={_dbg_frame_count} rss={_rss}MB {_ram} n={len(cls_ids)} | {' | '.join(_m1_items)}")
        else:
            _dbg_print(f"[M1] frame={_dbg_frame_count} rss={_rss}MB {_ram} n=0 (no detections)")
    # ── End debug prints ─────────────────────────────────────────────────────

    # Some deployments run seg-type configs against detect-only exports intentionally.
    # Treat missing masks as acceptable and continue with box-only behavior silently.

    # Start from all model1 detections, then route only selected classes to stage-2.
    keep_indices = [] if cls_ids is None else list(range(len(cls_ids)))
    # Legacy main-model defect bucket is kept for compatibility, but not used when
    # model2_pass_classes controls stage-2 routing.
    main_defect_indices = []

    # Apply per-class confidence thresholds from class config on model1 outputs.
    if confs is not None and cls_ids is not None:
        def _filter_main_by_class_conf(indices):
            out = []
            for i in indices:
                try:
                    cls_id = int(cls_ids[i])
                    cconf = float(confs[i])
                except Exception:
                    continue
                raw_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
                _, cls_entry = resolve_class(raw_name, main_section, main_alias_map)
                conf_threshold = float(
                    cls_entry.get("confidence", main_default_entry.get("confidence", conf))
                )
                if cconf >= conf_threshold:
                    out.append(i)
            return out

        keep_indices = _filter_main_by_class_conf(keep_indices)

    # Remove duplicate/contained boxes across all kept model1 detections.
    all_main_indices = keep_indices.copy()
    all_main_indices = dedup_leaf_boxes(all_main_indices, xyxy, confs)
    all_main_indices = dedup_indices_keep_biggest_overlap(all_main_indices, xyxy, overlap_thresh=0.8)
    keep_indices = all_main_indices.copy()
    main_defect_indices = []

    # If seg model, tighten boxes using mask outlines for better crops
    xyxy_use = xyxy
    if main_model_type == "seg" and xyxy is not None:
        xyxy_use = tighten_boxes_with_masks(keep_indices, xyxy, masks, masks_xy, frame_bgr.shape)

    # Apply per-class min_w/min_h filter for main model detections (pixels)
    # This removes tiny boxes early so they are not rendered, tracked, or passed to stage-2.
    if xyxy_use is not None and cls_ids is not None:
        main_section = cc.get("main_model", {})
        keep_indices = filter_indices_by_min_hw(keep_indices, xyxy_use, cls_ids, names, main_section)

    roi_xyxy = None
    if roi_enabled or roi_debug_draw:
        roi_xyxy = roi_bounds_from_frame(frame_bgr.shape, roi_margin_pct)
    if roi_enabled and roi_xyxy is not None and xyxy_use is not None:
        keep_indices = filter_indices_by_roi_overlap(
            keep_indices, xyxy_use, roi_xyxy, roi_min_overlap, roi_filter_mode
        )

    # These are the final model1 detections after filtering (for render/tracking stats).
    all_main_indices = keep_indices.copy()

    # Stage-2 routing: keep only configured classes for model2 pass.
    if not _m2_pass_all and keep_indices and cls_ids is not None:
        routed = []
        for i in keep_indices:
            if i < 0 or i >= len(cls_ids):
                continue
            raw_name = str(class_name(names, int(cls_ids[i]))).strip().lower()
            cls_key, _ = resolve_class(raw_name, main_section, main_alias_map)
            if raw_name in _m2_pass_set or cls_key in _m2_pass_set:
                routed.append(i)
        keep_indices = routed

    # -----------------------------------------------------------------------
    # RENDERING: Draw leaf boundaries (boxes or masks)
    # -----------------------------------------------------------------------
    # Render all model1 detections that survived filters (regardless of stage-2 routing).
    render_indices = all_main_indices

    if main_render == "mask" and (masks is not None or masks_xy is not None):
        annotated = draw_main_masks(
            frame_bgr, xyxy_use, cls_ids, names, render_indices,
            main_colors, masks, alpha=mask_alpha, masks_xy=masks_xy,
        )
    else:
        annotated = draw_main_boxes(
            frame_bgr, xyxy_use, confs, cls_ids, names, render_indices,
            main_colors, main_thickness,
        )

    if roi_debug_draw and roi_xyxy is not None:
        ix1, iy1, ix2, iy2 = [int(round(v)) for v in roi_xyxy]
        cv2.rectangle(
            annotated,
            (ix1, iy1),
            (ix2, iy2),
            roi_debug_color,
            roi_debug_thickness,
            lineType=cv2.LINE_AA,
        )

    # -----------------------------------------------------------------------
    # STAGE 2: Defect detection per leaf
    # -----------------------------------------------------------------------
    m2_calls_frame = 0
    if keep_indices and xyxy_use is not None:
        # Debug logging: only log every N frames to avoid flooding stdout
        stage2_debug_should_log = False
        if debug_log:
            counter = int(getattr(infer_frame_leaf_grouped_tracked, "_debug_counter", 0)) + 1
            setattr(infer_frame_leaf_grouped_tracked, "_debug_counter", counter)
            stage2_debug_should_log = (counter % debug_every_n == 0)

        for i in keep_indices:
            m2_calls_frame += 1
            x1, y1, x2, y2 = xyxy_use[i]
            raw_main_name = str(class_name(names, int(cls_ids[i]))).strip().lower()
            main_cls_key, main_cls_entry = resolve_class(raw_main_name, main_section, main_alias_map)
            is_main_defect = (
                main_cls_key in _main_defect_keys_runtime
                or str(main_cls_entry.get("role", "")).strip().lower() == "defect"
            )

            # Get leaf mask for masked cropping (seg model only)
            leaf_mask = None
            if main_model_type == "seg" and masks is not None:
                if i < len(masks):
                    leaf_mask = masks[i]

            leaf_tid_dbg = int(track_ids[i]) if track_ids is not None else None

            # Run defect model on this leaf's crop
            defects = run_defect_on_leaf(
                defect_model, frame_bgr, (x1, y1, x2, y2), leaf_mask,
                defect_imgsz, conf, device,
                min_area_px=min_area_px,
                debug_log_raw=stage2_debug_should_log and debug_log_raw,
                debug_log_filter_reasons=stage2_debug_should_log and debug_log_drop,
                debug_leaf_tid=leaf_tid_dbg,
            )

            # Model 2 detailed logs are optional to avoid per-frame I/O overhead.
            if _dbg_frame_log:
                _leaf_tid_pr = leaf_tid_dbg if leaf_tid_dbg is not None else -1
                _m2_rss = _dbg_mem_rss_mb()
                _m2_ram = _dbg_mem_ram_str()
                if defects:
                    _m2_items = [f"{d['name']}({float(d['conf']):.2f})" for d in defects]
                    _dbg_print(f"[M2] frame={_dbg_frame_count} rss={_m2_rss}MB {_m2_ram} leaf_tid={_leaf_tid_pr} defects={', '.join(_m2_items)}")
                else:
                    _dbg_print(f"[M2] frame={_dbg_frame_count} rss={_m2_rss}MB {_m2_ram} leaf_tid={_leaf_tid_pr} defects=none")
            if stage2_debug_should_log:
                leaf_tid = leaf_tid_dbg if leaf_tid_dbg is not None else -1
                raw_items = []
                for d in defects:
                    bx1, by1, bx2, by2 = d["box"]
                    raw_items.append(
                        f"{d['name']}@{float(d['conf']):.2f}[{int(bx1)},{int(by1)},{int(bx2)},{int(by2)}]"
                    )
                kept_text = ", ".join(raw_items) if raw_items else "none"
                _dbg_print(f"[debug][defect_model][kept] leaf_tid={leaf_tid} kept={kept_text}")

            # Separate stem detections from other defects using class_config roles
            stem_detections = []
            non_stem_defects = []
            for d in defects:
                d_key, d_entry = resolve_class(d["name"], defect_section, defect_alias_map)
                if d_entry.get("role") == "ignore":
                    stem_detections.append(d)
                else:
                    non_stem_defects.append(d)

            # Track stem length for size checks: model boxes vs OpenCV geometry (infer_config stem_size_source)
            if stats is not None and track_ids is not None:
                leaf_tid = int(track_ids[i])
                if stem_size_source == "opencv" and stem_size:
                    crop_ocv = extract_masked_leaf_crop(
                        frame_bgr, (x1, y1, x2, y2), leaf_mask
                    )
                    stem_px_frame = 0.0
                    if crop_ocv is not None and crop_ocv.size > 0:
                        try:
                            stem_ocv = detect_stem(
                                crop_ocv, min_stem_length_px=opencv_stem_min_px
                            )
                            if stem_ocv.get("has_stem"):
                                stem_px_frame = float(
                                    stem_ocv.get("stem_euclidean_length", 0.0)
                                )
                        except Exception:
                            stem_px_frame = 0.0
                    stems = stats.setdefault("leaf_stems", {})
                    prev = stems.get(leaf_tid)
                    if prev is None:
                        stems[leaf_tid] = (
                            stem_px_frame,
                            LEAF_STEMS_OPENCV_MARKER,
                            1,
                        )
                    else:
                        sum_px, sum_m, cnt = prev
                        stems[leaf_tid] = (
                            sum_px + stem_px_frame,
                            sum_m + LEAF_STEMS_OPENCV_MARKER,
                            cnt + 1,
                        )
                elif stem_size_source != "opencv" and stem_detections:
                    # Pick the largest stem detection in this frame (aspect-ratio length estimate)
                    best_len = 0.0
                    best_w = 0.0
                    for d in stem_detections:
                        sx1, sy1, sx2, sy2 = d["box"]
                        sw = max(1.0, float(sx2) - float(sx1))
                        sh = max(1.0, float(sy2) - float(sy1))
                        s_length = max(sw, sh)
                        if s_length > best_len:
                            best_len = s_length
                            best_w = min(sw, sh)
                    if best_len > 0:
                        s_area = best_len * best_w
                        s_ar = best_len / max(1.0, best_w)
                        stems = stats.setdefault("leaf_stems", {})
                        prev = stems.get(leaf_tid)
                        if prev is None:
                            stems[leaf_tid] = (s_area, s_ar, 1)
                        else:
                            sum_a, sum_r, cnt = prev
                            stems[leaf_tid] = (
                                sum_a + s_area,
                                sum_r + s_ar,
                                cnt + 1,
                            )

            # Vote counters for /infer/stats (summary uses vote_min_count).
            # Leaf border red/green and defect boxes use this frame's detections immediately.
            if stats is not None and track_ids is not None:
                leaf_tid = int(track_ids[i])
                stats.setdefault("leaf_defect_total", {}).setdefault(leaf_tid, 0)
                stats["leaf_defect_total"][leaf_tid] += 1
                for d in non_stem_defects:
                    stats.setdefault("leaf_defect_counts", {}).setdefault(leaf_tid, {})
                    curr = stats["leaf_defect_counts"][leaf_tid].get(d["name"], 0)
                    stats["leaf_defect_counts"][leaf_tid][d["name"]] = curr + 1
                if m2_size_enabled:
                    candidates = []
                    for d in non_stem_defects:
                        dname = str(d.get("name", "")).strip().lower()
                        dkey, _ = resolve_class(dname, defect_section, defect_alias_map)
                        if m2_size_all or dname in m2_size_class_set or dkey in m2_size_class_set:
                            candidates.append(d)
                    if candidates:
                        best = max(
                            candidates,
                            key=lambda dd: max(1.0, float(dd["box"][2]) - float(dd["box"][0]))
                            * max(1.0, float(dd["box"][3]) - float(dd["box"][1])),
                        )
                        bx1, by1, bx2, by2 = best["box"]
                        bw = max(1.0, float(bx2) - float(bx1))
                        bh = max(1.0, float(by2) - float(by1))
                        bl = max(bw, bh)
                        bwd = min(bw, bh)
                        bar = bl / max(1.0, bwd)
                        barea = bl * bwd
                        m2s = stats.setdefault("model2_track_sizes", {})
                        prev = m2s.get(leaf_tid)
                        if prev is None:
                            m2s[leaf_tid] = (bl, bwd)
                        else:
                            prev_l, prev_w = prev
                            if barea > prev_l * prev_w:
                                m2s[leaf_tid] = (bl, bwd)
            # Keep only the biggest stem per leaf (by box area)
            best_stem = None
            if stem_detections:
                best_stem = max(stem_detections, key=lambda d: (
                    (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1])
                ))
            # Draw + leaf border: all non-stem defects seen this frame (after model filters)
            display_defects = ([best_stem] if best_stem else []) + non_stem_defects
            defect_found = bool(non_stem_defects) or is_main_defect

            # So summaries treat this track as defected (same idea as torn/cut + leaf_defect_counts)
            if defect_found and stats is not None and track_ids is not None:
                leaf_tid = int(track_ids[i])
                dset = stats.setdefault("leaf_defects", {}).setdefault(leaf_tid, set())
                for d in non_stem_defects:
                    dset.add(d.get("name", "defect"))

            # Draw defect boxes on the annotated frame
            draw_defects(annotated, display_defects, defect_colors, defect_thickness,
                         draw_stem=bool(defect_cfg.get("draw_stem", True)))

            # Verdict border + label policy:
            # - If model2 finds nothing: keep model1 class label as-is.
            # - If model1 is defect + model2 defects exist: "<MODEL1_LABEL> DEFECTED".
            # - If model1 is good + model2 defects exist: "DEFECTED".
            if defect_found:
                m1_label = str(main_cls_entry.get("label", main_cls_key)).strip() or str(main_cls_key).strip()
                if non_stem_defects:
                    verdict_label = f"{m1_label} DEFECTED" if is_main_defect else "DEFECTED"
                else:
                    verdict_label = m1_label
                cv2.rectangle(
                    annotated,
                    (int(x1), int(y1)), (int(x2), int(y2)),
                    (0, 0, 255),
                    main_thickness,
                )
                if main_render != "mask":
                    # Cover previous draw_main_boxes label above the box.
                    wx1, wy2 = int(x1), int(y1)
                    (txt_w, txt_h), _ = cv2.getTextSize(
                        verdict_label,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        2,
                    )
                    wx2 = min(annotated.shape[1] - 1, wx1 + txt_w + 12)
                    wy1 = max(0, wy2 - 24)
                    cv2.rectangle(annotated, (wx1, wy1), (wx2, wy2), (0, 0, 0), -1)
                cv2.putText(
                    annotated,
                    verdict_label,
                    (int(x1) + 4, int(y1) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            else:
                cv2.rectangle(
                    annotated,
                    (int(x1), int(y1)), (int(x2), int(y2)),
                    (0, 255, 0),
                    main_thickness,
                )

    # -----------------------------------------------------------------------
    # MAIN-MODEL DEFECTS: torn, cut from stage 1 (skip stage 2)
    # -----------------------------------------------------------------------
    if main_defect_indices and xyxy_use is not None:
        for i in main_defect_indices:
            x1, y1, x2, y2 = xyxy_use[i]
            raw_cname = str(names.get(int(cls_ids[i]), str(cls_ids[i]))).strip().lower()
            cls_key, cls_entry = resolve_class(raw_cname, main_section, main_alias_map)
            defect_label = cls_entry.get("label", cls_key)

            # Draw red border to mark as defected
            cv2.rectangle(
                annotated,
                (int(x1), int(y1)), (int(x2), int(y2)),
                (0, 0, 255),  # Red = defected
                main_thickness,
            )
            cv2.putText(
                annotated, defect_label,
                (int(x1) + 4, int(y1) + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA,
            )

            # Update stats: count as defected leaf with this defect type
            if stats is not None and track_ids is not None:
                leaf_tid = int(track_ids[i])
                stats.setdefault("leaf_defect_total", {}).setdefault(leaf_tid, 0)
                stats["leaf_defect_total"][leaf_tid] += 1
                stats.setdefault("leaf_defect_counts", {}).setdefault(leaf_tid, {})
                curr = stats["leaf_defect_counts"][leaf_tid].get(cls_key, 0)
                stats["leaf_defect_counts"][leaf_tid][cls_key] = curr + 1

    # -----------------------------------------------------------------------
    # STATS: Update tracking stats and compute summary
    # -----------------------------------------------------------------------
    # Include both good leaves and main-model defects in tracking
    all_indices = all_main_indices
    if stats is not None and xyxy_use is not None and cls_ids is not None:
        update_tracking_stats(stats, xyxy_use, cls_ids, track_ids, names, all_indices)

    summary = compute_tracking_summary(stats) if stats is not None else {}
    if isinstance(summary, dict):
        summary["m2_calls_frame"] = int(m2_calls_frame)
        summary["m1_det_frame"] = int(len(all_main_indices)) if all_main_indices is not None else 0
    summary["frame_leaf_count"] = len(all_indices) if all_indices is not None else 0

    if stats is not None:
        m1_size_aliases = None
        if not _m1_size_all:
            m1_size_aliases = set(_m1_size_set)
        summary["size_summary"] = compute_size_summary(
            stats, size_cfg=size_cfg, cm_per_px=cm_per_px, stem_size=stem_size,
            leaf_aliases=m1_size_aliases, max_full_leaf=max_full_leaf,
        )
        # Final size pass/fail per object:
        # - default auto rule: if model2 size config exists, require model1 AND model2
        # - else only model1.
        _m2_cfg_exists = bool(m2_size_enabled and (m2_size_all or len(m2_size_class_set) > 0))
        if m2_size_logic == "auto":
            final_logic = "model1_and_model2" if _m2_cfg_exists else "model1_only"
        elif m2_size_logic in ("model1_only", "model1_and_model2"):
            final_logic = m2_size_logic
        else:
            final_logic = "model1_only"

        def _get_range_cm(cfg, key):
            node = cfg.get(key, {}) if isinstance(cfg, dict) else {}
            try:
                mn = float(node.get("min", 0.0))
            except Exception:
                mn = 0.0
            try:
                mx = float(node.get("max", 0.0))
            except Exception:
                mx = 0.0
            return mn, mx

        m2_len_min, m2_len_max = _get_range_cm(m2_size_cfg, "length")
        m2_w_min, m2_w_max = _get_range_cm(m2_size_cfg, "width")
        m1_leaf_cfg = size_cfg.get("leaf", {}) if isinstance(size_cfg, dict) else {}
        m1_len_min, m1_len_max = _get_range_cm(m1_leaf_cfg, "length")
        m1_w_min, m1_w_max = _get_range_cm(m1_leaf_cfg, "width")
        m1_stem_cfg = size_cfg.get("stem", {}) if isinstance(size_cfg, dict) else {}
        m1_stem_min, m1_stem_max = _get_range_cm(m1_stem_cfg, "length")

        full_ids = stats.get("size_full_leaf_ids", []) if isinstance(stats, dict) else []
        track_sizes = stats.get("track_sizes", {}) if isinstance(stats, dict) else {}
        leaf_stems = stats.get("leaf_stems", {}) if isinstance(stats, dict) else {}
        m2_track_sizes = stats.get("model2_track_sizes", {}) if isinstance(stats, dict) else {}
        final_in = 0
        final_out = 0
        final_count = 0
        for tid in full_ids:
            if tid not in track_sizes:
                continue
            m1_len_px, m1_w_px = track_sizes[tid]
            m1_len_cm = m1_len_px * float(cm_per_px)
            m1_w_cm = m1_w_px * float(cm_per_px)
            m1_ok = (m1_len_min <= m1_len_cm <= m1_len_max) and (m1_w_min <= m1_w_cm <= m1_w_max)
            if stem_size and tid in leaf_stems:
                stem_sum_area, stem_sum_ar, stem_cnt = leaf_stems.get(tid)
                if stem_cnt > 0:
                    stem_avg_area = stem_sum_area / stem_cnt
                    stem_avg_ar = stem_sum_ar / stem_cnt
                    if abs(stem_avg_ar - LEAF_STEMS_OPENCV_MARKER) < 1e-3:
                        stem_len_px = stem_avg_area
                    else:
                        stem_len_px = (stem_avg_area * stem_avg_ar) ** 0.5
                    stem_cm = stem_len_px * float(cm_per_px)
                    m1_ok = m1_ok and (m1_stem_min <= stem_cm <= m1_stem_max)

            m2_ok = True
            if final_logic == "model1_and_model2":
                if tid not in m2_track_sizes:
                    m2_ok = False
                else:
                    m2_len_px, m2_w_px = m2_track_sizes[tid]
                    m2_len_cm = m2_len_px * float(cm_per_px)
                    m2_w_cm = m2_w_px * float(cm_per_px)
                    m2_ok = (m2_len_min <= m2_len_cm <= m2_len_max) and (m2_w_min <= m2_w_cm <= m2_w_max)

            final_count += 1
            if m1_ok and m2_ok:
                final_in += 1
            else:
                final_out += 1

        final_summary = {
            "logic": final_logic,
            "count": final_count,
            "in_spec": final_in,
            "out_spec": final_out,
            "in_spec_pct": round((100.0 * final_in / final_count), 2) if final_count > 0 else 0.0,
            "out_spec_pct": round((100.0 * final_out / final_count), 2) if final_count > 0 else 0.0,
        }
        summary["size_summary"]["final"] = final_summary

        # Draw size debug overlay if enabled
        if size_debug and xyxy_use is not None and track_ids is not None:
            draw_size_debug(
                annotated, xyxy_use, track_ids, stats,
                size_cfg=size_cfg, cm_per_px=cm_per_px,
                stem_size=stem_size, color=size_debug_color,
            )

        # Final defect summary by track-level final class + votes.
        # Why: avoid overcounting a leaf as torn just because it had a brief torn hit
        # in a few frames. Main-model defects (torn/cut) are only counted when that
        # track's final class is defect; other defects still use vote threshold.
        # Main-model classes that should be counted as direct defects.
        main_defect_keys = _main_def_set_cfg if _main_def_set_cfg else get_main_defect_aliases()
        raw_counts = {}  # config_key -> count
        total_defects = 0
        total_defected_objects = 0
        leaf_defect_counts = stats.get("leaf_defect_counts", {})
        tracks = stats.get("tracks", {})

        # Majority/final class for each track from accumulated class_counts.
        final_class = {}
        for tid, info in tracks.items():
            counts = info.get("class_counts", {})
            if not counts:
                continue
            final_class[tid] = max(counts.items(), key=lambda kv: kv[1])[0]

        total_objects = 0
        for leaf_tid, fc in final_class.items():
            kept = []
            fc_key_raw = str(fc).strip().lower()
            fc_key, _ = resolve_class(fc_key_raw, main_section, main_alias_map)
            is_object = _obj_all or fc_key in _obj_set or fc_key_raw in _obj_set
            if not is_object:
                continue
            total_objects += 1

            # Count main-model defect when final track class is a main defect.
            # Also count stage-2 defects for the same leaf (if present by votes).
            if fc_key in main_defect_keys:
                cfg_key = fc_key
                kept.append(cfg_key)
                raw_counts[cfg_key] = raw_counts.get(cfg_key, 0) + 1

            # For stage-2 defects, require vote threshold.
            # This applies to both good leaves and main-defect leaves.
            counts = leaf_defect_counts.get(leaf_tid, {})
            stage2_kept = []
            for name, cnt in counts.items():
                if name in main_defect_keys:
                    continue
                if cnt >= vote_min_count:
                    cfg_key, _ = _normalize_defect_key(name)
                    stage2_kept.append(cfg_key)
            if stage2_kept:
                for cfg_key in stage2_kept:
                    if cfg_key not in kept:
                        kept.append(cfg_key)
                        raw_counts[cfg_key] = raw_counts.get(cfg_key, 0) + 1

            if kept:
                total_defected_objects += 1
                total_defects += len(kept)

        # Include all configured defect classes (even 0 count) for consistent report
        # This includes defects from both the main model and defect model
        defect_counts = {}
        # Main-model defect classes (torn, cut)
        for cfg_key, entry in main_section.items():
            if cfg_key.startswith("_"):
                continue
            if entry.get("role") != "defect":
                continue
            defect_counts[cfg_key] = {
                "label": entry.get("label", cfg_key),
                "count": raw_counts.get(cfg_key, 0),
            }
        # Defect-model defect classes (yellow spots, white spots, etc.)
        for cfg_key, entry in defect_section.items():
            if cfg_key.startswith("_"):
                continue
            if entry.get("role") != "defect":
                continue
            defect_counts[cfg_key] = {
                "label": entry.get("label", cfg_key),
                "count": raw_counts.get(cfg_key, 0),
            }
        # Include any observed defects not in config
        for cfg_key, cnt in raw_counts.items():
            if cfg_key not in defect_counts:
                _, lbl = _normalize_defect_key(cfg_key)
                defect_counts[cfg_key] = {"label": lbl, "count": cnt}

        total_good_objects = max(0, int(total_objects) - int(total_defected_objects))
        good_pct = (100.0 * total_good_objects / total_objects) if total_objects > 0 else 0.0
        defected_pct = (100.0 * total_defected_objects / total_objects) if total_objects > 0 else 0.0
        defect_distribution = {
            k: int(v.get("count", 0))
            for k, v in defect_counts.items()
            if int(v.get("count", 0)) > 0
        }
        summary["defect_counts"] = defect_counts
        summary["defect_distribution"] = defect_distribution
        summary["total_defects"] = total_defects
        summary["total_objects"] = total_objects
        summary["total_defected_objects"] = total_defected_objects
        summary["total_good_objects"] = total_good_objects
        summary["good_percentage"] = round(good_pct, 2)
        summary["defected_percentage"] = round(defected_pct, 2)
        # Backward compatibility
        summary["total_leaf"] = total_objects
        summary["total_defected_leaf"] = total_defected_objects

    # Robust fallback when tracker IDs are temporarily missing:
    # keep frame-level infer stats non-zero from observed detections so /infer/stats
    # is still meaningful after stop/persist.
    if track_ids is None:
        observed_leaf = len(all_indices) if all_indices is not None else 0
        observed_main_defects = len(main_defect_indices) if main_defect_indices is not None else 0
        try:
            summary["frame_leaf_count"] = max(int(summary.get("frame_leaf_count", 0)), int(observed_leaf))
            summary["total_objects"] = max(int(summary.get("total_objects", 0)), int(observed_leaf))
            summary["total_leaf"] = max(int(summary.get("total_leaf", 0)), int(observed_leaf))
            summary["tracks_seen"] = max(int(summary.get("tracks_seen", 0)), int(observed_leaf))
            summary["total_defected_objects"] = max(
                int(summary.get("total_defected_objects", 0)),
                int(observed_main_defects),
            )
            summary["total_defected_leaf"] = max(
                int(summary.get("total_defected_leaf", 0)),
                int(observed_main_defects),
            )
            summary["total_defects"] = max(
                int(summary.get("total_defects", 0)),
                int(observed_main_defects),
            )
            _tot = int(summary.get("total_objects", 0))
            _def = int(summary.get("total_defected_objects", 0))
            _good = max(0, _tot - _def)
            summary["total_good_objects"] = max(int(summary.get("total_good_objects", 0)), _good)
            if _tot > 0:
                summary["good_percentage"] = round(100.0 * _good / _tot, 2)
                summary["defected_percentage"] = round(100.0 * _def / _tot, 2)
        except Exception:
            pass

    # Break references to large per-frame tensors/results so Python/torch can
    # reclaim them earlier instead of carrying them across many frames.
    try:
        del results
    except Exception:
        pass
    try:
        del result
    except Exception:
        pass
    try:
        del masks
    except Exception:
        pass
    try:
        del masks_xy
    except Exception:
        pass
    return annotated, summary


# ===========================================================================
# PUBLIC API
# ===========================================================================

__all__ = [
    # Constants (used by server.py for defaults)
    "ENGINE_PATH",
    "DEFECT_ENGINE_PATH",
    "CM_PER_PX",
    "SIZE_THRESHOLDS",
    # Class config (from class_config.json)
    "get_class_config",
    "get_leaf_aliases",
    "get_main_defect_aliases",
    "get_stem_aliases",
    # Model management
    "load_model",
    "load_models",
    "unload_models",
    # Config
    "load_config",
    "get_config",
    # Main pipeline
    "infer_frame_leaf_grouped_tracked",
    # Stats (re-exported for server.py)
    "compute_tracking_summary",
    "compute_size_summary",
    "draw_size_debug",
]
