import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import asyncio

import os
import copy
import time
import json
import gc
import sys
from datetime import datetime, timezone
import logging
import logging.handlers
import threading
import queue
import shutil
import subprocess
import cv2
import numpy as np
import config_store
import pipeline
from smart_monitor import SmartMonitor
from utils import deep_merge_dict

# Set environment variables for better GPU memory management
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")  # Load CUDA modules lazily
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Suppress TF logs
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:128,garbage_collection_threshold:0.6,expandable_segments:True",
)

import infer_main as infer
import infer_detect

Gst.init(None)

app = FastAPI()

# Max concurrent MJPEG viewers per camera. (Jetson can get overloaded if too many browsers connect.)
STREAM_MAX_CLIENTS = int(os.environ.get("STREAM_MAX_CLIENTS", "4"))

# Configure CORS - add your frontend origin(s) here
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve saved media files from /media
app.mount("/media/local_video", StaticFiles(directory="media/videos/local"), name="local_video")
app.mount("/media", StaticFiles(directory="media"), name="media")

capture_dir = "./media/images"
video_dir = "./media/videos"
os.makedirs(capture_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)
MAX_SAVED_INFER_ARCHIVE_VIDEOS = 10

# UI setup persistence (frontend boot/save settings blob)
UISETUP_DATA_DIR = os.environ.get("GOMICRO_DATA_DIR", "./data")
UISETUP_FILE = os.path.join(UISETUP_DATA_DIR, "uisetup.json")
RAM_SHUTDOWN_ASSESSMENT_FILE = os.path.join(UISETUP_DATA_DIR, "ram_shutdown_assessment.json")
os.makedirs(UISETUP_DATA_DIR, exist_ok=True)
UISETUP_LOCK = threading.Lock()
_RAM_KILL_SHUTDOWN_LOCK = threading.Lock()
_RAM_KILL_SHUTDOWN_DONE = False

# Fixed 1080p resolution — single pipeline, single buffer
CAPTURE_WIDTH, CAPTURE_HEIGHT = 1920, 1080
STREAM_WIDTH, STREAM_HEIGHT = 1920, 1080
PROCESS_WIDTH, PROCESS_HEIGHT = 1920, 1080
CAMERA_FPS = 30

DEFAULT_SETTINGS = {
    "wbmode": 3,
    "exposure_us": 10000000,
    "gain_min": 3.0,
    "gain_max": 3.0,
    "digital_gain_min": 1.0,
    "digital_gain_max": 1.0,
    "aelock": True,
    "camera_mode": "single",
}

DEFAULT_CONFIG = {
    "exceed_enabled": False,
    # If False, Argus/GStreamer camera pipelines are not started at boot (or after
    # video EOF). Use for video-file inference when nvarguscamerasrc crashes or
    # Argus times out after TensorRT loads. Live camera: set True when hardware works.
    "start_camera_on_boot": True,
    # If False, do NOT auto-restart camera when leaving video inference sessions.
    # This avoids repeated NVMM/NvMap allocation spikes on low-memory Jetson setups.
    "auto_resume_camera_after_video_infer": False,
    # Stable runtime mode: keep camera/model runtime steady and avoid source switching.
    # In this mode, source is forced to camera and camera pipeline is not restarted on
    # /stream/mode calls, reducing Argus/NVMM churn and fragmentation risk.
    "stable_runtime_mode": False,
    # Enable smart diagnostics logging (monitor.log only).
    "monitor_enabled": True,
    # Sampling interval for monitor logs.
    "monitor_interval_sec": 5,
    # Print periodic RAM/RSS usage to main server logs (terminal + exced.log).
    "ram_log_enabled": True,
    "ram_log_interval_sec": 5,
    # Contiguous-memory diagnostics log (NvMap/CMA/buddyinfo snapshots).
    "contig_diag_enabled": True,
    "contig_diag_interval_sec": 2,
    # External realtime monitor GUI toggle (scripts/realtime_monitor_gui.py).
    "live_plot_gui_enabled": False,
    # Default MJPEG streaming framerate for /stream endpoint.
    "stream_fps": 30,
    # Inference processing FPS target across camera/video workers.
    "infer_fps": 20,
    # Realtime mode switch:
    # - True: minimum latency, drops older frames under load
    # - False: keeps frame order, allows delay growth
    "realtime_mode": True,
    # Buffer sizes used when realtime_mode=False.
    "buffer_sizes": {
        "gst_queue": 30,
        "appsink": 30,
        "infer_queue": 30,
    },
    # Processing behavior: legacy realtime-only mode.
    "processing_mode": "live_process",
}


def _group_server_config(flat_cfg: dict) -> dict:
    """Group flat server config into 3 sections for app_config readability."""
    f = flat_cfg or {}
    return {
        "model_related": {
            "exceed_enabled": bool(f.get("exceed_enabled", False)),
        },
        "mode_related": {
            "start_camera_on_boot": bool(f.get("start_camera_on_boot", True)),
            "auto_resume_camera_after_video_infer": bool(
                f.get("auto_resume_camera_after_video_infer", False)
            ),
            "stable_runtime_mode": bool(f.get("stable_runtime_mode", False)),
            "stream_fps": f.get("stream_fps", 30),
            "infer_fps": f.get("infer_fps", 20),
            "realtime_mode": bool(f.get("realtime_mode", True)),
            "buffer_sizes": f.get(
                "buffer_sizes",
                {"gst_queue": 30, "appsink": 30, "infer_queue": 30},
            ),
            "processing_mode": str(f.get("processing_mode", "live_process")),
            "record_mode_process_fps": f.get("record_mode_process_fps", 20.0),
            "record_mode_capture_fps": f.get("record_mode_capture_fps", 20.0),
            "record_mode_unload_models_on_complete": bool(
                f.get("record_mode_unload_models_on_complete", True)
            ),
        },
        "others": {
            "monitor_enabled": bool(f.get("monitor_enabled", True)),
            "monitor_interval_sec": f.get("monitor_interval_sec", 5),
            "ram_log_enabled": bool(f.get("ram_log_enabled", True)),
            "ram_log_interval_sec": f.get("ram_log_interval_sec", 5),
            "contig_diag_enabled": bool(f.get("contig_diag_enabled", True)),
            "contig_diag_interval_sec": f.get("contig_diag_interval_sec", 2),
            "live_plot_gui_enabled": bool(f.get("live_plot_gui_enabled", False)),
        },
    }


def _flatten_server_config(raw_cfg: dict) -> dict:
    """Flatten grouped or legacy server config to runtime flat keys."""
    out = DEFAULT_CONFIG.copy()
    raw = raw_cfg if isinstance(raw_cfg, dict) else {}

    # New grouped layout
    model_related = raw.get("model_related", {})
    mode_related = raw.get("mode_related", {})
    others = raw.get("others", {})
    if isinstance(model_related, dict):
        out.update({k: v for k, v in model_related.items() if k in out})
    if isinstance(mode_related, dict):
        out.update({k: v for k, v in mode_related.items() if k in out})
    if isinstance(others, dict):
        out.update({k: v for k, v in others.items() if k in out})

    # Backward-compatible flat keys (legacy format)
    out.update({k: v for k, v in raw.items() if k in out})
    # Realtime legacy-only: ignore any persisted record_mode setting.
    out["processing_mode"] = "live_process"
    return out

DEFAULT_INFER_VIDEO_PATH = os.path.join(
    os.path.dirname(__file__), "media", "videos", "local", "sample.mp4"
)
RECORD_MODE_CAM0_INFER_PATH = os.path.join(
    os.path.dirname(__file__), "media", "videos", "top_infer.mp4"
)

def _load_infer_config_flags():
    """Read preload_models and unload_on_stop from app_config infer section."""
    try:
        data = config_store.load_infer_config()
        preload = data.get("preload_models", True)
        unload = data.get("unload_on_stop", True)
        return bool(preload), bool(unload)
    except Exception:
        return True, True

PRELOAD_MODELS, UNLOAD_ON_STOP = _load_infer_config_flags()

def _cleanup_old_infer_archive_videos(max_keep: int = MAX_SAVED_INFER_ARCHIVE_VIDEOS) -> None:
    """Keep only the latest N timestamped DeepStream infer archives."""
    try:
        keep = max(0, int(max_keep))
    except Exception:
        keep = 10
    if keep <= 0:
        return
    try:
        entries = []
        for name in os.listdir(video_dir):
            if not (name.startswith("top_infer_ds_") and name.endswith(".mp4")):
                continue
            full_path = os.path.join(video_dir, name)
            if not os.path.isfile(full_path):
                continue
            entries.append((os.path.getmtime(full_path), full_path))
        if len(entries) <= keep:
            return
        entries.sort(key=lambda item: item[0], reverse=True)
        for _, stale_path in entries[keep:]:
            try:
                os.remove(stale_path)
            except Exception as exc:
                logging.warning("infer archive cleanup failed for %s: %s", stale_path, exc)
    except Exception as exc:
        logging.warning("infer archive cleanup skipped: %s", exc)

def _resolve_infer_video_path(path_value):
    if path_value is None:
        return os.path.abspath(DEFAULT_INFER_VIDEO_PATH)
    p = str(path_value).strip()
    if not p:
        return os.path.abspath(DEFAULT_INFER_VIDEO_PATH)
    if not os.path.isabs(p):
        p = os.path.join(os.path.dirname(__file__), p)
    return os.path.abspath(p)


def _resolve_server_local_path(path_value: str | None) -> str:
    """Absolute path for a value relative to Server2/ (same rule as infer video)."""
    p = str(path_value or "").strip()
    if not p:
        return ""
    if os.path.isabs(p):
        return os.path.abspath(p)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), p))


def _warmup_bgr_square(path_value: str | None, side: int, role: str) -> tuple[np.ndarray, str]:
    """BGR (side×side) for model warmup: load from disk or black placeholder."""
    p = _resolve_server_local_path(path_value)
    if p and os.path.isfile(p):
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is not None and bgr.size > 0:
            h, w = bgr.shape[:2]
            if w != side or h != side:
                interp = cv2.INTER_AREA if w * h > side * side else cv2.INTER_LINEAR
                bgr = cv2.resize(bgr, (side, side), interpolation=interp)
            return bgr, f"{role}=file:{p}"
    z = np.zeros((side, side, 3), dtype=np.uint8)
    return z, f"{role}=zeros"


def _is_readable_video_file(path_value: str | None) -> bool:
    path = str(path_value or "").strip()
    if not path or not os.path.isfile(path):
        return False
    cap = None
    try:
        cap = cv2.VideoCapture(path)
        if cap is None or not cap.isOpened():
            return False
        ok, frame = cap.read()
        return bool(ok and frame is not None)
    except Exception:
        return False
    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass


def _candidate_infer_video_paths() -> list[str]:
    local_dir = os.path.join(os.path.dirname(__file__), "media", "videos", "local")
    candidates = [_resolve_infer_video_path(_load_infer_video_path()), os.path.abspath(DEFAULT_INFER_VIDEO_PATH)]
    try:
        for name in sorted(os.listdir(local_dir)):
            if name.lower().endswith((".mp4", ".mov", ".mkv", ".avi")):
                candidates.append(os.path.abspath(os.path.join(local_dir, name)))
    except Exception:
        pass
    seen = set()
    ordered = []
    for path in candidates:
        norm = os.path.normpath(path)
        if norm in seen:
            continue
        seen.add(norm)
        ordered.append(path)
    return ordered


def _resolve_working_infer_video_path() -> str:
    preferred = os.path.abspath(DEFAULT_INFER_VIDEO_PATH)
    if _is_readable_video_file(preferred):
        return preferred
    for candidate in _candidate_infer_video_paths():
        if _is_readable_video_file(candidate):
            return candidate
    return preferred

def _load_infer_video_path():
    """Read video_source_path from app_config infer section."""
    try:
        data = config_store.load_infer_config()
        if not isinstance(data, dict):
            return _resolve_working_infer_video_path()
        configured = _resolve_infer_video_path(data.get("video_source_path"))
        if _is_readable_video_file(configured):
            return configured
        fallback = _resolve_working_infer_video_path()
        if os.path.normpath(fallback) != os.path.normpath(configured):
            logging.warning(
                "Configured infer video is unreadable (%s), using fallback %s",
                configured,
                fallback,
            )
        return fallback
    except Exception:
        return _resolve_working_infer_video_path()

DEFAULT_SIZE_CONFIG = copy.deepcopy(infer.SIZE_THRESHOLDS)
DEFAULT_CM_PER_PX = infer.CM_PER_PX
DEFAULT_STEM_SIZE = False
DEFAULT_SIZE_DEBUG = False
DEFAULT_SIZE_DEBUG_COLOR = [255, 150, 0]  # BGR orange
DEFAULT_SELECTED_CATEGORY = "Spinach"
DEFAULT_MAX_FULL_LEAF = 20

def _coerce_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        txt = value.strip().lower()
        if txt in ("1", "true", "yes", "on"):
            return True
        if txt in ("0", "false", "no", "off"):
            return False
    return bool(default)

def _sanitize_size_config(cfg):
    if not isinstance(cfg, dict):
        return None
    base = copy.deepcopy(DEFAULT_SIZE_CONFIG)
    leaf = cfg.get("leaf")
    if isinstance(leaf, dict):
        for dim in ("length", "width"):
            dim_cfg = leaf.get(dim)
            if isinstance(dim_cfg, dict):
                for bound in ("min", "max"):
                    if bound in dim_cfg:
                        try:
                            base["leaf"][dim][bound] = float(dim_cfg[bound])
                        except Exception:
                            return None
    stem = cfg.get("stem")
    if isinstance(stem, dict):
        dim_cfg = stem.get("length")
        if isinstance(dim_cfg, dict):
            for bound in ("min", "max"):
                if bound in dim_cfg:
                    try:
                        base["stem"]["length"][bound] = float(dim_cfg[bound])
                    except Exception:
                        return None
    return base

def _validate_min_max(cfg):
    if cfg["leaf"]["length"]["min"] > cfg["leaf"]["length"]["max"]:
        return False
    if cfg["leaf"]["width"]["min"] > cfg["leaf"]["width"]["max"]:
        return False
    if cfg["stem"]["length"]["min"] > cfg["stem"]["length"]["max"]:
        return False
    return True


def _default_other_profile() -> dict:
    stem_len = DEFAULT_SIZE_CONFIG.get("stem", {}).get("length", {})
    return {
        "length": {
            "min": float(stem_len.get("min", 1.0)),
            "max": float(stem_len.get("max", 4.0)),
        },
        "width": {"min": 0.0, "max": 2.0},
    }


def _default_size_doc() -> dict:
    leaf = DEFAULT_SIZE_CONFIG.get("leaf", {})
    return {
        "cm_per_px": float(DEFAULT_CM_PER_PX),
        "selected_category": DEFAULT_SELECTED_CATEGORY,
        "categories": {
            DEFAULT_SELECTED_CATEGORY: {
                "default": {
                    "length": {
                        "min": float(leaf.get("length", {}).get("min", 10.0)),
                        "max": float(leaf.get("length", {}).get("max", 12.0)),
                    },
                    "width": {
                        "min": float(leaf.get("width", {}).get("min", 4.0)),
                        "max": float(leaf.get("width", {}).get("max", 6.0)),
                    },
                },
                "other": _default_other_profile(),
            }
        },
    }


def _sanitize_profile(profile: dict) -> dict | None:
    if not isinstance(profile, dict):
        return None
    out = {
        "length": {"min": 0.0, "max": 0.0},
        "width": {"min": 0.0, "max": 0.0},
    }
    for dim in ("length", "width"):
        node = profile.get(dim)
        if not isinstance(node, dict):
            return None
        try:
            mn = float(node.get("min"))
            mx = float(node.get("max"))
        except Exception:
            return None
        if mn > mx:
            return None
        out[dim]["min"] = mn
        out[dim]["max"] = mx
    return out


def _normalize_size_payload_to_v2(payload: dict) -> dict | None:
    """
    Normalize supported size-config payload shapes into canonical v2 schema.

    Canonical schema:
      {
        cm_per_px: float,
        selected_category: str,
        categories: { "<category>": { default: profile, other: profile } }
      }
    """
    base = _default_size_doc()
    if not isinstance(payload, dict):
        return None

    # New format (v2)
    if "categories" in payload:
        cats_raw = payload.get("categories")
        if not isinstance(cats_raw, dict) or len(cats_raw) == 0:
            return None
        cats_out = {}
        for cat, cfg in cats_raw.items():
            if not isinstance(cat, str) or not cat.strip() or not isinstance(cfg, dict):
                return None
            dflt = _sanitize_profile(cfg.get("default"))
            other = _sanitize_profile(cfg.get("other"))
            if dflt is None or other is None:
                return None
            cats_out[cat] = {"default": dflt, "other": other}
        try:
            cm = float(payload.get("cm_per_px", base["cm_per_px"]))
        except Exception:
            return None
        if cm <= 0:
            return None
        sel = str(payload.get("selected_category", "")).strip()
        if not sel or sel not in cats_out:
            sel = next(iter(cats_out.keys()))
        return {"cm_per_px": cm, "selected_category": sel, "categories": cats_out}

    # Legacy wrapper: { cm_per_px, size_config: {leaf, stem} }
    if "size_config" in payload:
        sc = payload.get("size_config")
        if isinstance(sc, str):
            try:
                sc = json.loads(sc)
            except Exception:
                return None
        if not isinstance(sc, dict):
            return None
        payload = {"cm_per_px": payload.get("cm_per_px", base["cm_per_px"]), **sc}

    # Legacy flat: {leaf, stem, cm_per_px}
    if "leaf" in payload or "stem" in payload:
        leaf = payload.get("leaf", {})
        stem = payload.get("stem", {})
        default_profile = _sanitize_profile({
            "length": (leaf or {}).get("length", {}),
            "width": (leaf or {}).get("width", {}),
        })
        stem_len = (stem or {}).get("length", {}) if isinstance(stem, dict) else {}
        other_profile = _sanitize_profile({
            "length": stem_len,
            "width": {"min": 0.0, "max": 2.0},
        })
        if default_profile is None or other_profile is None:
            return None
        try:
            cm = float(payload.get("cm_per_px", base["cm_per_px"]))
        except Exception:
            return None
        if cm <= 0:
            return None
        sel = DEFAULT_SELECTED_CATEGORY
        return {
            "cm_per_px": cm,
            "selected_category": sel,
            "categories": {sel: {"default": default_profile, "other": other_profile}},
        }

    return None


def _size_doc_to_legacy_thresholds(size_doc: dict) -> dict:
    """Convert canonical v2 active category profiles into infer legacy shape."""
    out = copy.deepcopy(DEFAULT_SIZE_CONFIG)
    cats = size_doc.get("categories", {}) if isinstance(size_doc, dict) else {}
    sel = str(size_doc.get("selected_category", "")).strip() if isinstance(size_doc, dict) else ""
    if not isinstance(cats, dict) or not cats:
        return out
    if sel not in cats:
        sel = next(iter(cats.keys()))
    chosen = cats.get(sel, {})
    dflt = chosen.get("default", {})
    other = chosen.get("other", {})
    try:
        out["leaf"]["length"]["min"] = float(dflt.get("length", {}).get("min", out["leaf"]["length"]["min"]))
        out["leaf"]["length"]["max"] = float(dflt.get("length", {}).get("max", out["leaf"]["length"]["max"]))
        out["leaf"]["width"]["min"] = float(dflt.get("width", {}).get("min", out["leaf"]["width"]["min"]))
        out["leaf"]["width"]["max"] = float(dflt.get("width", {}).get("max", out["leaf"]["width"]["max"]))
        out["stem"]["length"]["min"] = float(other.get("length", {}).get("min", out["stem"]["length"]["min"]))
        out["stem"]["length"]["max"] = float(other.get("length", {}).get("max", out["stem"]["length"]["max"]))
    except Exception:
        return copy.deepcopy(DEFAULT_SIZE_CONFIG)
    return out

def _parse_color(val, default):
    """Parse a BGR color from a list/tuple of 3 ints, return as tuple."""
    if isinstance(val, (list, tuple)) and len(val) == 3:
        try:
            return tuple(int(c) for c in val)
        except Exception:
            pass
    return tuple(default)

def load_size_config():
    data = config_store.load_size_config_raw()
    if not data:
        doc = _default_size_doc()
        legacy = _size_doc_to_legacy_thresholds(doc)
        return (
            legacy,
            float(doc["cm_per_px"]),
            bool(DEFAULT_STEM_SIZE),
            bool(DEFAULT_SIZE_DEBUG),
            tuple(DEFAULT_SIZE_DEBUG_COLOR),
            doc,
        )
    try:
        size_debug = data.get("size_debug", DEFAULT_SIZE_DEBUG) if isinstance(data, dict) else DEFAULT_SIZE_DEBUG
        size_debug_color = data.get("size_debug_color", DEFAULT_SIZE_DEBUG_COLOR) if isinstance(data, dict) else DEFAULT_SIZE_DEBUG_COLOR
        doc = _normalize_size_payload_to_v2(data if isinstance(data, dict) else {})
        if doc is None:
            doc = _default_size_doc()
        legacy = _size_doc_to_legacy_thresholds(doc)
        return (
            legacy,
            float(doc.get("cm_per_px", DEFAULT_CM_PER_PX)),
            bool(DEFAULT_STEM_SIZE),
            _coerce_bool(size_debug, DEFAULT_SIZE_DEBUG),
            _parse_color(size_debug_color, DEFAULT_SIZE_DEBUG_COLOR),
            doc,
        )
    except Exception:
        doc = _default_size_doc()
        legacy = _size_doc_to_legacy_thresholds(doc)
        return (
            legacy,
            float(doc["cm_per_px"]),
            bool(DEFAULT_STEM_SIZE),
            bool(DEFAULT_SIZE_DEBUG),
            tuple(DEFAULT_SIZE_DEBUG_COLOR),
            doc,
        )


def save_size_config(size_cfg, cm_per_px, stem_size, size_debug=False, size_debug_color=None, size_doc=None):
    if size_debug_color is None:
        size_debug_color = list(DEFAULT_SIZE_DEBUG_COLOR)
    if size_doc is None:
        # Backward-safe fallback.
        size_doc = _default_size_doc()
        size_doc["cm_per_px"] = float(cm_per_px)
    # Preserve max_full_leaf from existing config so it is not wiped on every save.
    existing = config_store.load_size_config_raw()
    payload = {
        "cm_per_px": float(size_doc.get("cm_per_px", cm_per_px)),
        "selected_category": str(size_doc.get("selected_category", DEFAULT_SELECTED_CATEGORY)),
        "categories": size_doc.get("categories", _default_size_doc()["categories"]),
        "max_full_leaf": int(existing.get("max_full_leaf", DEFAULT_MAX_FULL_LEAF)),
        "size_debug": bool(size_debug),
        "size_debug_color": list(size_debug_color),
    }
    config_store.save_size_config_raw(payload)

def load_config():
    data = config_store.load_server_config()
    return _flatten_server_config(data)


def _resolve_frame_mode(cfg: dict) -> str:
    """Resolve effective frame mode from server config.

    Preferred key: realtime_mode (bool).
    Backward compatibility: frame_mode ("realtime" | "buffer").
    """
    if isinstance(cfg, dict) and "realtime_mode" in cfg:
        return "realtime" if _coerce_bool(cfg.get("realtime_mode"), True) else "buffer"
    # Legacy fallback
    mode = str((cfg or {}).get("frame_mode", "realtime")).strip().lower()
    return mode if mode in ("realtime", "buffer") else "realtime"


def _coerce_buffer_size(value, default: int = 30) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        n = int(default)
    return max(1, min(300, n))


def _coerce_stream_fps(value, default: float = 30.0) -> float:
    try:
        fps = float(value)
    except (TypeError, ValueError):
        fps = float(default)
    return max(1.0, min(60.0, fps))


def _coerce_interval_sec(value, default: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = float(default)
    return max(0.5, min(60.0, v))

def save_config(config):
    config_store.save_server_config(_group_server_config(config))

SERVER_CONFIG = load_config()
save_config(SERVER_CONFIG)
EXCEED_ENABLED = bool(SERVER_CONFIG.get("exceed_enabled", False))
STABLE_RUNTIME_MODE = bool(SERVER_CONFIG.get("stable_runtime_mode", False))
PROCESSING_MODE = "live_process"
MONITOR_ENABLED = bool(SERVER_CONFIG.get("monitor_enabled", True))
MONITOR_INTERVAL_SEC = float(SERVER_CONFIG.get("monitor_interval_sec", 5) or 5)
RAM_LOG_ENABLED = bool(SERVER_CONFIG.get("ram_log_enabled", True))
RAM_LOG_INTERVAL_SEC = float(SERVER_CONFIG.get("ram_log_interval_sec", 5) or 5)
CONTIG_DIAG_ENABLED = bool(SERVER_CONFIG.get("contig_diag_enabled", True))
CONTIG_DIAG_INTERVAL_SEC = _coerce_interval_sec(
    SERVER_CONFIG.get("contig_diag_interval_sec", 2),
    2.0,
)
LIVE_PLOT_GUI_ENABLED = bool(SERVER_CONFIG.get("live_plot_gui_enabled", False))
STREAM_FPS = _coerce_stream_fps(SERVER_CONFIG.get("stream_fps", 30), 30.0)
INFER_FPS = _coerce_stream_fps(SERVER_CONFIG.get("infer_fps", 20), 20.0)
RECORD_MODE_PROCESS_FPS = 20.0
RECORD_MODE_CAPTURE_FPS = 20.0
RECORD_MODE_UNLOAD_MODELS_ON_COMPLETE = False
CAPTURE_MODE = _resolve_frame_mode(SERVER_CONFIG)
FRAME_MODE = CAPTURE_MODE
_buf_cfg = SERVER_CONFIG.get("buffer_sizes", {})
if not isinstance(_buf_cfg, dict):
    _buf_cfg = {}
BUFFER_GST_QUEUE = _coerce_buffer_size(_buf_cfg.get("gst_queue", 30), 30)
BUFFER_APPSINK = _coerce_buffer_size(_buf_cfg.get("appsink", 30), 30)
BUFFER_INFER_QUEUE = _coerce_buffer_size(_buf_cfg.get("infer_queue", 30), 30)
# Optional RAM "kill switch". When enabled, the server will gracefully shutdown
# if total RAM usage exceeds this percentage. Default: disabled (0).
RAM_KILL_PCT = float(os.environ.get("RAM_KILL_PCT", SERVER_CONFIG.get("ram_kill_pct", 0) or 0) or 0)
SMART_MONITOR = SmartMonitor(
    os.path.dirname(__file__),
    interval_sec=MONITOR_INTERVAL_SEC,
    ram_kill_pct=(RAM_KILL_PCT if RAM_KILL_PCT > 0 else 1000.0),
)
LIVE_PLOT_GUI_PROCESS = None

# Route all detection/memory debug prints from infer_main and infer_detect
# into monitor.log so every log line appears in one place.
infer.set_monitor(SMART_MONITOR.log)
infer_detect.set_monitor(SMART_MONITOR.log)


class monitor_task:
    """Context manager to log task start/end resource deltas when enabled."""

    def __init__(self, name: str, extra: str = ""):
        self.name = name
        self.extra = extra
        self.task_id = None

    def __enter__(self):
        if MONITOR_ENABLED:
            self.task_id = SMART_MONITOR.task_start(self.name, self.extra)
        return self

    def __exit__(self, exc_type, exc, tb):
        if MONITOR_ENABLED and self.task_id is not None:
            extra = self.extra
            if exc is not None:
                extra = f"{extra} | error={exc.__class__.__name__}"
            SMART_MONITOR.task_end(self.task_id, extra=extra)

def load_settings():
    data = config_store.load_camera_settings()
    if not data:
        return DEFAULT_SETTINGS.copy()
    try:
        # Backward compatibility for legacy single-value gain fields.
        if isinstance(data, dict):
            if "gain" in data and ("gain_min" not in data and "gain_max" not in data):
                try:
                    gain_val = float(data["gain"])
                    data["gain_min"] = gain_val
                    data["gain_max"] = gain_val
                except Exception:
                    pass
            if "digital_gain" in data and ("digital_gain_min" not in data and "digital_gain_max" not in data):
                try:
                    dg_val = float(data["digital_gain"])
                    data["digital_gain_min"] = dg_val
                    data["digital_gain_max"] = dg_val
                except Exception:
                    pass
        settings = DEFAULT_SETTINGS.copy()
        settings.update({k: v for k, v in data.items() if k in settings})
        settings.pop("capture_resolution", None)  # No longer used, fixed at 1080p
        return settings
    except Exception:
        return DEFAULT_SETTINGS.copy()

def save_settings(settings):
    config_store.save_camera_settings(settings)

CAMERA_SETTINGS = load_settings()

save_settings(CAMERA_SETTINGS)

def load_infer_state():
    return config_store.load_infer_state()

def save_infer_state(state):
    config_store.save_infer_state(state)

def _read_uisetup_settings():
    if not os.path.exists(UISETUP_FILE):
        return None
    try:
        with open(UISETUP_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        settings = data.get("settings")
        return settings if isinstance(settings, dict) else None
    except Exception:
        # Corrupted file: treat as first run so frontend can seed defaults.
        return None

def _write_uisetup_settings(settings: dict) -> str:
    updated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    payload = {"updated_at": updated_at, "settings": settings}
    tmp_path = f"{UISETUP_FILE}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, UISETUP_FILE)
    return updated_at

INFER_STATE = load_infer_state()

def reset_infer_state_file():
    """Clear infer_state.json so stats are per-session and fully runtime-driven."""
    global INFER_STATE
    # Do not seed hardcoded camera/video entries.
    # last_video is populated only when a real recording session writes it.
    INFER_STATE = {"stats": {}, "last_video": {}}
    save_infer_state(INFER_STATE)


def clear_infer_video_artifacts():
    """Remove stale infer videos so new infer session shows only new outputs."""
    paths = set()
    try:
        last_video = INFER_STATE.get("last_video", {}) if isinstance(INFER_STATE, dict) else {}
        if isinstance(last_video, dict):
            for item in last_video.values():
                if isinstance(item, dict):
                    fp = item.get("file")
                    if isinstance(fp, str) and fp.strip():
                        paths.add(os.path.abspath(fp))
    except Exception:
        pass

    # Common infer outputs.
    paths.add(os.path.abspath(os.path.join(video_dir, "top_infer.mp4")))
    paths.add(os.path.abspath(os.path.join(video_dir, "bottom_infer.mp4")))
    paths.add(os.path.abspath(RECORD_MODE_CAM0_INFER_PATH))

    # DeepStream infer archive outputs.
    try:
        if os.path.isdir(video_dir):
            for name in os.listdir(video_dir):
                if name.startswith("top_infer_ds_") and name.endswith(".mp4"):
                    paths.add(os.path.abspath(os.path.join(video_dir, name)))
    except Exception:
        pass

    for p in sorted(paths):
        try:
            if os.path.isfile(p):
                os.remove(p)
        except Exception:
            pass

def stop_all_infer_workers():
    """Stop all inference and clean up inference state."""
    for cam_state in CAMERAS.values():
        cam_state.shared.inferring = False
        cam_state.last_infer_error_ts = 0.0
    
    # Completely unload models from GPU memory
    unload_all_models()

AVAILABLE_CAMS = {
    0: os.path.exists("/dev/video0"),
    1: os.path.exists("/dev/video1"),
}

def get_active_cams():
    mode = CAMERA_SETTINGS.get("camera_mode", "single")
    if mode == "dual":
        return [cam_id for cam_id, ok in AVAILABLE_CAMS.items() if ok]
    # single mode defaults to cam 0 only
    return [0] if AVAILABLE_CAMS.get(0) else []

def _gst_element_has_property(factory_name: str, prop_name: str) -> bool:
    """Check whether a GStreamer element factory exposes a property."""
    element = None
    try:
        factory = Gst.ElementFactory.find(factory_name)
        if factory is None:
            return False
        element = factory.create(None)
        if element is None:
            return False
        return element.find_property(prop_name) is not None
    except Exception:
        return False
    finally:
        element = None

def build_pipeline(cam_id: int, fps_override: int | None = None, simple_pipeline: bool = False) -> str:
    wbmode = CAMERA_SETTINGS["wbmode"]
    # Argus exposuretimerange expects nanoseconds; value stored is treated as ns.
    exposure_ns = int(CAMERA_SETTINGS["exposure_us"])
    gain_min = float(CAMERA_SETTINGS["gain_min"])
    gain_max = float(CAMERA_SETTINGS["gain_max"])
    digital_gain_min = float(CAMERA_SETTINGS["digital_gain_min"])
    digital_gain_max = float(CAMERA_SETTINGS["digital_gain_max"])
    aelock = "true" if CAMERA_SETTINGS["aelock"] else "false"
    target_fps = int(fps_override or CAMERA_FPS)
    source_props = [
        f"sensor-id={cam_id}",
        "ee-mode=0", "ee-strength=0", "tnr-mode=0", "tnr-strength=0",
        f"ispdigitalgainrange=\"{digital_gain_min} {digital_gain_max}\"",
        f"aelock={aelock}", f"wbmode={wbmode}",
        f"exposuretimerange=\"{exposure_ns} {exposure_ns}\"",
        f"gainrange=\"{gain_min} {gain_max}\"",
    ]
    if FRAME_MODE == "buffer":
        appsink_props = [
            f"name=appsink{cam_id}",
            f"max-buffers={BUFFER_APPSINK}",
            "drop=false",
            "sync=false",
        ]
    else:
        appsink_props = [f"name=appsink{cam_id}", "max-buffers=1", "drop=true", "sync=false"]
    if _gst_element_has_property("appsink", "wait-on-eos"):
        appsink_props.append("wait-on-eos=false")
    pipeline_parts = [
        f"nvarguscamerasrc {' '.join(source_props)}",
        "!",
        f"video/x-raw(memory:NVMM),width={CAPTURE_WIDTH},height={CAPTURE_HEIGHT},format=NV12,framerate={target_fps}/1",
        "!",
    ]
    if not simple_pipeline:
        if FRAME_MODE == "buffer":
            pipeline_parts.extend(["queue", f"max-size-buffers={BUFFER_GST_QUEUE}", "!"])
        else:
            pipeline_parts.extend(["queue", "max-size-buffers=1", "leaky=downstream", "!"])
    pipeline_parts.extend([
        "nvvidconv",
        "output-buffers=1",
        "flip-method=0",
        "!",
        f"video/x-raw,width={PROCESS_WIDTH},height={PROCESS_HEIGHT},format=BGRx",
        "!",
        "videoconvert",
        "!",
        "video/x-raw,format=BGR",
        "!",
        f"appsink {' '.join(appsink_props)}",
    ])
    return " ".join(pipeline_parts)

def _log_pipeline_bus(cam_state: "CameraState", context: str):
    """Drain and log useful bus messages around camera startup failures."""
    if cam_state.pipeline is None:
        return
    bus = cam_state.pipeline.get_bus()
    if bus is None:
        return
    while True:
        msg = bus.pop_filtered(
            Gst.MessageType.ERROR | Gst.MessageType.WARNING | Gst.MessageType.EOS
        )
        if msg is None:
            break
        if msg.type == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            logging.error(
                "Camera %s GStreamer error during %s: %s%s",
                cam_state.cam_id,
                context,
                err,
                f" (debug: {debug})" if debug else "",
            )
        elif msg.type == Gst.MessageType.WARNING:
            err, debug = msg.parse_warning()
            logging.warning(
                "Camera %s GStreamer warning during %s: %s%s",
                cam_state.cam_id,
                context,
                err,
                f" (debug: {debug})" if debug else "",
            )
        elif msg.type == Gst.MessageType.EOS:
            logging.warning(
                "Camera %s pipeline reached EOS during %s",
                cam_state.cam_id,
                context,
            )

class SharedState:
    """Shared state for pipeline output, controlled by inference and recording flags."""
    def __init__(self):
        self.latest: np.ndarray | None = None    # Always the current output frame (raw or inferred)
        self.latest_raw: np.ndarray | None = None  # Always the latest raw camera frame
        # Frame publication tracking (used by MJPEG stream to avoid stale/frozen frames)
        self.latest_seq: int = 0
        self.latest_ts: float = 0.0
        # Incremented on mode/source switches to invalidate previous stream state.
        self.frame_generation: int = 0
        self.frame_mode: str = FRAME_MODE
        self.capture_mode: str = CAPTURE_MODE
        # Used only when frame_mode="buffer".
        self.infer_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=BUFFER_INFER_QUEUE)
        self.inferring: bool = False      # True only when inference running
        self.recording: bool = False      # True when recording active
        self.session_mode: str = "live_process"
        self.active_category: str = DEFAULT_SELECTED_CATEGORY
        self.cm_per_px: float = float(DEFAULT_CM_PER_PX)
        self.size_thresholds: dict | None = None
        self.processing_active: bool = False
        self.processing_status: str = "idle"
        self.processing_error: str | None = None
        self.processing_total_frames: int = 0
        self.processing_done_frames: int = 0
        self.processing_started_ts: float = 0.0

class CameraState:
    def __init__(self, cam_id: int):
        
        self.cam_id = cam_id
        self.pipeline = None
        self.appsink = None
        self.running = False
        self.pipeline_thread = None
        self.shared = SharedState()
        
        # Recording management
        self.ffmpeg_process = None
        self.current_video_path = None
        self.record_lock = threading.Lock()
        
        # Stream management
        self.stream_running = False
        self.stream_clients = 0
        self.stream_token = 0
        self.stream_lock = threading.Lock()
        
        # Inference tracking
        self.last_infer_error_ts = 0.0

CAMERAS = {}

def init_cameras():
    global CAMERAS
    physical = get_active_cams()
    # Always expose at least cam 0 so video-file inference (/stream/mode source=video)
    # works when no /dev/video* nodes exist or Argus is disabled at boot.
    active = physical if physical else [0]
    if not physical:
        logging.warning(
            "No /dev/video* for current camera_mode — using logical camera 0 only "
            "(video file inference). Live Argus pipeline will not start without devices."
        )
    CAMERAS = {cam_id: CameraState(cam_id) for cam_id in active}
    init_track_stats(active)
    return active, physical

def init_track_stats(active_cams):
    global TRACK_STATS
    with TRACK_STATS_LOCK:
        TRACK_STATS = {
            cam_id: {
                "tracks": {},
                "leaf_defects": {},
                "summary": {},
                "track_sizes": {},
                "leaf_stems": {},
                "leaf_defect_counts": {},
                "leaf_defect_total": {},
                "size_full_leaf_ids": [],
            }
            for cam_id in active_cams
        }
        for cam_id in active_cams:
            saved = INFER_STATE.get("stats", {}).get(str(cam_id))
            if isinstance(saved, dict):
                TRACK_STATS[cam_id]["summary"] = saved


# Hook smart_monitor to log per-track container sizes so we can correlate
# detection activity with RAM growth.
def _monitor_extra_fields():
    try:
        with TRACK_STATS_LOCK:
            s0 = TRACK_STATS.get(0, {}) if isinstance(TRACK_STATS, dict) else {}
            tracks = s0.get("tracks", {}) if isinstance(s0.get("tracks", {}), dict) else {}
            stems = s0.get("leaf_stems", {}) if isinstance(s0.get("leaf_stems", {}), dict) else {}
            m2s = (
                s0.get("model2_track_sizes", {})
                if isinstance(s0.get("model2_track_sizes", {}), dict)
                else {}
            )
            defects = s0.get("leaf_defects", {}) if isinstance(s0.get("leaf_defects", {}), dict) else {}
        qsz = -1
        try:
            cs = CAMERAS.get(0)
            if cs is not None and hasattr(cs, "shared") and hasattr(cs.shared, "infer_queue"):
                qsz = int(cs.shared.infer_queue.qsize())
        except Exception:
            qsz = -1
        return {
            "cam0_tracks": len(tracks),
            "cam0_stems": len(stems),
            "cam0_m2sizes": len(m2s),
            "cam0_defsets": len(defects),
            "cam0_defcnt": len(s0.get("leaf_defect_counts", {}))
            if isinstance(s0.get("leaf_defect_counts", {}), dict)
            else 0,
            "infer_q": qsz,
        }
    except Exception:
        return {}


try:
    SMART_MONITOR.set_extra_provider(_monitor_extra_fields)
except Exception:
    pass

def reset_track_stats(cam_id=None):
    clear_keys = (
        "tracks",
        "leaf_defects",
        "summary",
        "track_sizes",
        "leaf_stems",
        "leaf_defect_counts",
        "leaf_defect_total",
        "model2_track_sizes",
    )
    with TRACK_STATS_LOCK:
        if cam_id is None:
            for stats in TRACK_STATS.values():
                for key in clear_keys:
                    stats[key] = {}
                stats["size_full_leaf_ids"] = []
                stats.pop("frame_idx", None)
                stats.pop("peak_summary", None)
        else:
            stats = TRACK_STATS.get(cam_id)
            if stats is not None:
                for key in clear_keys:
                    stats[key] = {}
                stats["size_full_leaf_ids"] = []
                stats.pop("frame_idx", None)
                stats.pop("peak_summary", None)

def _build_fallback_summary_from_stats(stats):
    """Best-effort summary when per-frame summary is missing."""
    tracks = stats.get("tracks", {}) if isinstance(stats, dict) else {}
    leaf_defect_total = stats.get("leaf_defect_total", {}) if isinstance(stats, dict) else {}
    leaf_defects = stats.get("leaf_defects", {}) if isinstance(stats, dict) else {}
    if not isinstance(tracks, dict):
        tracks = {}
    if not isinstance(leaf_defect_total, dict):
        leaf_defect_total = {}
    if not isinstance(leaf_defects, dict):
        leaf_defects = {}

    tracks_seen = len(tracks)
    defected_leaf = set()
    total_defects = 0
    for tid, cnt in leaf_defect_total.items():
        try:
            c = int(cnt)
        except Exception:
            c = 0
        if c > 0:
            defected_leaf.add(tid)
            total_defects += c
    for tid, defs in leaf_defects.items():
        if isinstance(defs, (set, list, tuple, dict)) and len(defs) > 0:
            defected_leaf.add(tid)
            if isinstance(defs, (set, list, tuple)):
                total_defects += len(defs)

    return {
        "size_summary": {"leaf": _default_size_leaf_bucket()},
        "total_leaf": tracks_seen,
        "total_defects": int(total_defects),
        "total_defected_leaf": len(defected_leaf),
        "defect_counts": {},
        "tracks_seen": tracks_seen,
        "frame_leaf_count": tracks_seen,
    }

def maybe_persist_stats(cam_id, stats):
    if stats is None:
        return
    now = time.time()
    last_persist = stats.get("last_persist_ts", 0.0)
    if now - last_persist < 1.0:
        return
    summary = stats.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}
    if not summary or (int(summary.get("tracks_seen", 0)) == 0 and int(summary.get("total_leaf", 0)) == 0):
        summary = stats.get("peak_summary") or _build_fallback_summary_from_stats(stats)
    summary = _normalize_infer_summary(summary)
    INFER_STATE.setdefault("stats", {})
    INFER_STATE["stats"][str(cam_id)] = summary
    save_infer_state(INFER_STATE)
    stats["last_persist_ts"] = now

def persist_stats_now(cam_id):
    with TRACK_STATS_LOCK:
        stats = TRACK_STATS.get(cam_id)
        if stats is None:
            return
        summary = stats.get("summary", {})
        if not isinstance(summary, dict):
            summary = {}
        if not summary or (int(summary.get("tracks_seen", 0)) == 0 and int(summary.get("total_leaf", 0)) == 0):
            summary = stats.get("peak_summary") or _build_fallback_summary_from_stats(stats)
        summary = _normalize_infer_summary(summary)
        INFER_STATE.setdefault("stats", {})
        INFER_STATE["stats"][str(cam_id)] = summary
        save_infer_state(INFER_STATE)
        stats["last_persist_ts"] = time.time()

def _default_size_leaf_bucket():
    return {
        "count": 0,
        "in_spec": 0,
        "out_spec": 0,
        "in_spec_pct": 0,
        "out_spec_pct": 0,
    }

def _normalize_infer_summary(summary):
    if not isinstance(summary, dict):
        summary = {}
    size_summary = summary.get("size_summary")
    if not isinstance(size_summary, dict):
        size_summary = {}
    leaf_bucket = size_summary.get("leaf")
    if not isinstance(leaf_bucket, dict):
        leaf_bucket = {}
    leaf_defaults = _default_size_leaf_bucket()
    leaf_defaults.update({k: v for k, v in leaf_bucket.items() if k in leaf_defaults})
    size_summary["leaf"] = leaf_defaults
    summary["size_summary"] = size_summary
    summary.setdefault("total_leaf", 0)
    summary.setdefault("total_defects", 0)
    summary.setdefault("total_defected_leaf", 0)
    summary.setdefault("defect_counts", {})
    summary.setdefault("tracks_seen", 0)
    summary.setdefault("frame_leaf_count", 0)
    return summary


def _is_effectively_empty_summary(summary: dict) -> bool:
    """
    True when summary contains only zero/default values.

    A normalized summary may be structurally non-empty but still represent "no detections".
    """
    s = _normalize_infer_summary(summary if isinstance(summary, dict) else {})
    try:
        if int(s.get("total_leaf", 0)) != 0:
            return False
        if int(s.get("total_defects", 0)) != 0:
            return False
        if int(s.get("total_defected_leaf", 0)) != 0:
            return False
        if int(s.get("tracks_seen", 0)) != 0:
            return False
        if int(s.get("frame_leaf_count", 0)) != 0:
            return False
    except Exception:
        return False

    size_leaf = ((s.get("size_summary") or {}).get("leaf") or {})
    try:
        if int(size_leaf.get("count", 0)) != 0:
            return False
    except Exception:
        return False

    defect_counts = s.get("defect_counts", {})
    if isinstance(defect_counts, dict):
        for v in defect_counts.values():
            c = v.get("count", 0) if isinstance(v, dict) else v
            try:
                if int(c) != 0:
                    return False
            except Exception:
                return False
    elif defect_counts:
        return False

    return True

RECORD_FPS = CAMERA_FPS if not EXCEED_ENABLED else 30
# INFER_FPS is set from SERVER_CONFIG (mode_related.infer_fps) near the top of this file.
SIZE_CONFIG_LOCK = threading.Lock()
SIZE_CONFIG, SIZE_CM_PER_PX, SIZE_STEM_SIZE, SIZE_DEBUG, SIZE_DEBUG_COLOR, SIZE_CONFIG_DOC = load_size_config()
save_size_config(SIZE_CONFIG, SIZE_CM_PER_PX, SIZE_STEM_SIZE, SIZE_DEBUG, SIZE_DEBUG_COLOR, SIZE_CONFIG_DOC)
_size_raw = config_store.load_size_config_raw()
SIZE_MAX_FULL_LEAF = int(_size_raw.get("max_full_leaf", DEFAULT_MAX_FULL_LEAF))

MAIN_MODEL = None
DEFECT_MODEL = None
MODEL_LOCK = threading.Lock()
INFER_LOCK = threading.Lock()
MODEL_READY = threading.Event()
JPEG_LOCK = threading.Lock()

def require_exceed():
    if not EXCEED_ENABLED:
        raise HTTPException(status_code=403, detail="exceed_disabled")

def get_models():
    global MAIN_MODEL, DEFECT_MODEL
    with monitor_task("get_models"):
        if MAIN_MODEL is None or DEFECT_MODEL is None:
            with MODEL_LOCK:
                if MAIN_MODEL is None or DEFECT_MODEL is None:
                    monitor_event("MODEL_LOADING — allocating GPU+RAM for YOLO engines")
                    logging.info("Loading YOLO models...")
                    import psutil, os as _os
                    proc = psutil.Process(_os.getpid())
                    rss_before = proc.memory_info().rss / (1024**2)

                    MAIN_MODEL, DEFECT_MODEL = infer.load_models()
                    MODEL_READY.set()

                    rss_after = proc.memory_info().rss / (1024**2)
                    _def_tag = type(DEFECT_MODEL).__name__
                    if hasattr(DEFECT_MODEL, "runtime_summary"):
                        _def_tag = f"{_def_tag} ({DEFECT_MODEL.runtime_summary()})"
                    _msg = (
                        f"[Model Load] RSS before: {rss_before:.0f} MiB, after: {rss_after:.0f} MiB "
                        f"(allocated ~{rss_after - rss_before:.0f} MiB) | "
                        f"main={type(MAIN_MODEL).__name__} defect={_def_tag}"
                    )
                    print(_msg)
                    SMART_MONITOR.log(_msg)
                    monitor_event(f"MODEL_LOADED — RSS: {rss_after:.0f}MiB (+{rss_after - rss_before:.0f}MiB)")
                    logging.info("YOLO models loaded successfully and ready for inference")
                else:
                    logging.info("Models already loaded, reusing...")
        else:
            logging.debug("Models already available in memory")
    return MAIN_MODEL, DEFECT_MODEL

def unload_all_models():
    """Completely unload all models from GPU memory."""
    global MAIN_MODEL, DEFECT_MODEL
    with monitor_task("unload_all_models"):
        with MODEL_LOCK:
            if MAIN_MODEL is not None or DEFECT_MODEL is not None:
                logging.info("Unloading YOLO models from GPU memory...")
                infer.unload_models(MAIN_MODEL, DEFECT_MODEL)
                MAIN_MODEL = None
                DEFECT_MODEL = None
                MODEL_READY.clear()
                # Small delay to ensure CUDA cache is cleared
                time.sleep(0.1)
                logging.info("YOLO models unloaded successfully")

TRACK_STATS = {}
TRACK_STATS_LOCK = threading.Lock()

STREAM_MODE = "raw"
STREAM_MODE_LOCK = threading.Lock()
INFER_SOURCE = "camera"
INFER_SOURCE_LOCK = threading.Lock()
INFER_VIDEO_PATH = _load_infer_video_path()
INFER_VIDEO_CAP = None
VIDEO_ENDED = False
VIDEO_READ_FAILS = 0
# Do not treat a single OpenCV read failure as EOF; Jetson+MP4 often returns (ok=False)
# once before resuming. Real EOF produces many consecutive failures.
INFER_VIDEO_READ_FAIL_MAX = 60
VIDEO_CAP_LOCK = threading.Lock()


def _open_infer_video_capture(path: str) -> cv2.VideoCapture | None:
    """Open a file for infer mode. On Jetson, OpenCV+FFmpeg often decodes one frame then fails;
    try GStreamer (decodebin, then nvv4l2) before falling back to FFmpeg.
    """
    abs_path = os.path.abspath(path)
    if not os.path.isfile(abs_path):
        return None
    gst_api = getattr(cv2, "CAP_GSTREAMER", None)
    attempts: list[tuple[str, object]] = []
    if gst_api is not None and sys.platform.startswith("linux"):
        loc = abs_path.replace("\\", "\\\\").replace('"', '\\"')
        # Software path — works for most MP4/MOV in OpenCV+GStreamer builds
        pipe_decode = (
            f'filesrc location="{loc}" ! decodebin ! queue ! videoconvert ! '
            f"video/x-raw,format=BGR ! appsink drop=true sync=false max-buffers=2"
        )
        attempts.append(("gstreamer_decodebin", lambda: cv2.VideoCapture(pipe_decode, gst_api)))
        # Jetson HW decode for H.264 in MP4 (falls through if codec/container mismatches)
        pipe_nv = (
            f'filesrc location="{loc}" ! qtdemux ! queue ! h264parse ! nvv4l2decoder ! '
            f"nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! "
            f"appsink drop=true sync=false max-buffers=2"
        )
        attempts.append(("gstreamer_nvv4l2_h264", lambda: cv2.VideoCapture(pipe_nv, gst_api)))
    api_ff = getattr(cv2, "CAP_FFMPEG", None)
    if api_ff is not None:
        attempts.append(("ffmpeg", lambda: cv2.VideoCapture(abs_path, api_ff)))
    attempts.append(("default", lambda: cv2.VideoCapture(abs_path)))

    for name, factory in attempts:
        cap = None
        try:
            cap = factory()
            if cap is not None and cap.isOpened():
                logging.info("Infer video capture backend=%s path=%s", name, abs_path)
                if name.startswith("ffmpeg") or name == "default":
                    try:
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception:
                        pass
                return cap
        except Exception as exc:
            logging.warning("Infer video backend %s failed: %s", name, exc)
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
    return None


def _recover_infer_video_capture(path: str) -> tuple[bool, np.ndarray] | None:
    """Release and re-open the file (may pick a different backend). Jetson often needs this after the first read."""
    global INFER_VIDEO_CAP, VIDEO_READ_FAILS
    with VIDEO_CAP_LOCK:
        try:
            if INFER_VIDEO_CAP is not None:
                INFER_VIDEO_CAP.release()
        except Exception:
            pass
        INFER_VIDEO_CAP = None
        INFER_VIDEO_CAP = _open_infer_video_capture(os.path.abspath(path))
        if INFER_VIDEO_CAP is None:
            return None
        ok, fr = INFER_VIDEO_CAP.read()
        if ok and fr is not None:
            VIDEO_READ_FAILS = 0
            return True, fr
    return None


def _loop_infer_video_from_start(path: str) -> tuple[bool, np.ndarray] | None:
    """After EOF or sustained read errors: seek 0 or reopen so infer/video sessions do not fall back to camera."""
    global INFER_VIDEO_CAP, VIDEO_READ_FAILS
    abs_path = os.path.abspath(path)
    with VIDEO_CAP_LOCK:
        if INFER_VIDEO_CAP is not None and INFER_VIDEO_CAP.isOpened():
            try:
                INFER_VIDEO_CAP.set(cv2.CAP_PROP_POS_FRAMES, 0)
            except Exception:
                pass
            ok, fr = INFER_VIDEO_CAP.read()
            if ok and fr is not None:
                VIDEO_READ_FAILS = 0
                logging.info("Infer video looped (seek start): %s", abs_path)
                return True, fr
        try:
            if INFER_VIDEO_CAP is not None:
                INFER_VIDEO_CAP.release()
        except Exception:
            pass
        INFER_VIDEO_CAP = None
        INFER_VIDEO_CAP = _open_infer_video_capture(abs_path)
        if INFER_VIDEO_CAP is None:
            return None
        ok, fr = INFER_VIDEO_CAP.read()
        if ok and fr is not None:
            VIDEO_READ_FAILS = 0
            logging.info("Infer video looped (reopen): %s", abs_path)
            return True, fr
    return None


MODE_CHANGE_LOCK = threading.Lock()
LAST_MODE_CHANGE_TS = 0.0
LAST_MODE = STREAM_MODE
LAST_SOURCE = INFER_SOURCE
MODE_CHANGE_COOLDOWN = 0.75
CAMERA_RESTART_LOCK = threading.Lock()
LAST_CAMERA_RESTART_TS = 0.0
CAMERA_RESTART_COOLDOWN = 1.0
DEEPSTREAM_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deepstream_runtime.log")


class DeepStreamSession:
    """Legacy-only stub kept for backward compatibility."""

    def __init__(self):
        self.source = None
        self.command = None
        self.record_output_path = None

    def is_running(self) -> bool:
        return False

    def start(
        self,
        source: str,
        video_path: str | None = None,
        record_output_path: str | None = None,
        publish_frame_fn=None,
    ) -> dict:
        raise RuntimeError("deepstream_removed_use_legacy_runtime")

    def stop(self):
        self.source = None
        self.command = None
        self.record_output_path = None


DEEPSTREAM_SESSION = DeepStreamSession()


def _is_deepstream_enabled() -> bool:
    # Legacy-only runtime: DeepStream has been removed from active use.
    return False

def _build_output_path(base_dir: str, input_path: str, prefix: str, ext: str) -> str:
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    ts = int(time.time() * 1000)
    filename = f"{prefix}_{base_name}_{ts}.{ext}"
    return os.path.join(base_dir, filename)

def start_ffmpeg_raw(output_path, width: int = 1920, height: int = 1080, fps: float = 30.0):
    """Start FFmpeg for raw video encoding (non-blocking write).

    yuv420p + faststart: HTML5 video in browsers often shows black without a web-safe pixel format.
    """
    w = int(max(1, width))
    h = int(max(1, height))
    try:
        fr = float(fps)
    except Exception:
        fr = 30.0
    if not (fr > 0.0):
        fr = 30.0
    if not shutil.which("ffmpeg"):
        raise HTTPException(
            status_code=503,
            detail="ffmpeg_not_found: install ffmpeg on the system (e.g. sudo apt install -y ffmpeg)",
        )
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "bgr24",
        "-r", f"{fr:.6f}",
        "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def stop_ffmpeg(proc):
    if proc and proc.stdin:
        try:
            proc.stdin.close()
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

def recorder_write(ffmpeg_proc, frame: np.ndarray) -> bool:
    """Write frame to FFmpeg stdin non-blocking. Returns False if pipe broken."""
    if not ffmpeg_proc or ffmpeg_proc.poll() is not None:
        return False
    try:
        ffmpeg_proc.stdin.write(frame.tobytes())
        return True
    except (BrokenPipeError, OSError):
        return False


def _record_frame_due(cam_state: CameraState, fps: float) -> bool:
    """Throttle recording writes for DeepStream frame bridge."""
    try:
        target_fps = float(max(fps, 1.0))
    except Exception:
        target_fps = 20.0
    interval = 1.0 / target_fps
    now = time.time()
    last = float(getattr(cam_state.shared, "last_record_ts", 0.0))
    if (now - last) < interval:
        return False
    cam_state.shared.last_record_ts = now
    return True

def grab_1080p(cam_state: CameraState) -> np.ndarray | None:
    """Grab frame from camera and ensure 1080p output."""
    if cam_state.appsink is None:
        return None
    
    sample = cam_state.appsink.emit("pull-sample")
    if sample is None:
        return None

    buf = sample.get_buffer()
    ok, mapinfo = buf.map(Gst.MapFlags.READ)
    if not ok:
        sample = None
        return None

    raw = np.frombuffer(mapinfo.data, dtype=np.uint8).copy()
    buf.unmap(mapinfo)
    sample = None
    
    try:
        frame_bgr = raw.reshape((PROCESS_HEIGHT, PROCESS_WIDTH, 3))
    except ValueError:
        return None
    
    # Always resize to 1080p for downstream processing
    if frame_bgr.shape[:2] != (1080, 1920):
        frame_bgr = cv2.resize(frame_bgr, (1920, 1080), interpolation=cv2.INTER_LINEAR)
    
    return frame_bgr

def run_inference(frame: np.ndarray, cam_state: CameraState) -> np.ndarray:
    """Run inference on frame and return annotated result."""
    if not cam_state.shared.inferring:
        return frame
    # record_mode capture phase: keep stream/record flow alive but skip model inference.
    if (
        getattr(cam_state.shared, "session_mode", "") == "record_mode"
        and STREAM_MODE == "infer"
        and bool(getattr(cam_state.shared, "recording", False))
    ):
        return frame
    try:
        main_model, defect_model = get_models()
    except Exception as exc:
        if not getattr(run_inference, "_err_logged", False):
            print(f"[DEBUG] run_inference: FAILED to load models: {exc}")
            run_inference._err_logged = True
        _log_infer_error(cam_state, f"cam {cam_state.cam_id} failed to load models", exc)
        return frame
    
    with TRACK_STATS_LOCK:
        stats = TRACK_STATS.get(cam_state.cam_id)
    
    with SIZE_CONFIG_LOCK:
        size_cfg = SIZE_CONFIG
        cm_per_px = SIZE_CM_PER_PX
        stem_size = SIZE_STEM_SIZE
        size_debug = SIZE_DEBUG
        size_debug_color = SIZE_DEBUG_COLOR
        max_full_leaf = SIZE_MAX_FULL_LEAF
    # Session overrides from /stream/mode infer payload (category + thresholds)
    active_thresholds = getattr(cam_state.shared, "size_thresholds", None)
    if isinstance(active_thresholds, dict):
        size_cfg = _size_doc_to_legacy_thresholds({
            "selected_category": str(getattr(cam_state.shared, "active_category", DEFAULT_SELECTED_CATEGORY)),
            "categories": {
                str(getattr(cam_state.shared, "active_category", DEFAULT_SELECTED_CATEGORY)): active_thresholds
            },
        })
    try:
        shared_cm = float(getattr(cam_state.shared, "cm_per_px", cm_per_px))
        if shared_cm > 0:
            cm_per_px = shared_cm
    except Exception:
        pass

    try:
        with INFER_LOCK:
            annotated, summary = infer.infer_frame_leaf_grouped_tracked(
                main_model,
                defect_model,
                frame,
                stats,
                size_cfg=size_cfg,
                cm_per_px=cm_per_px,
                stem_size=stem_size,
                size_debug=size_debug,
                size_debug_color=size_debug_color,
                max_full_leaf=SIZE_MAX_FULL_LEAF,
            )
    except Exception as exc:
        _log_infer_error(cam_state, f"cam {cam_state.cam_id} inference failed", exc)
        return frame
    
    with TRACK_STATS_LOCK:
        if stats is not None:
            stats["summary"] = summary
            _peak = stats.get("peak_summary") or {}
            if (int(summary.get("tracks_seen", 0)) + int(summary.get("total_defects", 0))) > \
               (int(_peak.get("tracks_seen", 0)) + int(_peak.get("total_defects", 0))):
                stats["peak_summary"] = dict(summary)
            maybe_persist_stats(cam_state.cam_id, stats)
            # Periodic diagnostics + bounded cleanup of long-lived tracking maps.
            # This helps separate Python container growth from allocator-retained memory.
            fidx = int(stats.get("frame_idx", 0))
            if fidx > 0 and (fidx % 120 == 0):
                tracks_n = len(stats.get("tracks", {})) if isinstance(stats.get("tracks", {}), dict) else 0
                defsets_n = len(stats.get("leaf_defects", {})) if isinstance(stats.get("leaf_defects", {}), dict) else 0
                defcnt_n = (
                    len(stats.get("leaf_defect_counts", {}))
                    if isinstance(stats.get("leaf_defect_counts", {}), dict)
                    else 0
                )
                m2sizes_n = (
                    len(stats.get("model2_track_sizes", {}))
                    if isinstance(stats.get("model2_track_sizes", {}), dict)
                    else 0
                )
                SMART_MONITOR.log(
                    f"[STATS] frame_idx={fidx} cam={cam_state.cam_id} "
                    f"tracks={tracks_n} defsets={defsets_n} defcnt={defcnt_n} m2sizes={m2sizes_n}"
                )
                if max(tracks_n, defsets_n, defcnt_n, m2sizes_n) >= 300:
                    monitor_event(
                        f"STATS_RESET cam{cam_state.cam_id} frame_idx={fidx} "
                        f"(tracks={tracks_n},defsets={defsets_n},defcnt={defcnt_n},m2sizes={m2sizes_n})"
                    )
                    reset_track_stats(cam_state.cam_id)
    
    return annotated

def _build_pipeline_hooks() -> pipeline.PipelineHooks:
    """Build callbacks used by pipeline.py so flow remains centralized and readable."""

    def _should_record_raw() -> bool:
        with STREAM_MODE_LOCK:
            mode = STREAM_MODE
        if mode != "infer":
            return True
        # In record_mode, infer sessions record raw stream first and process on stop.
        return PROCESSING_MODE == "record_mode"

    def _get_infer_source() -> str:
        with INFER_SOURCE_LOCK:
            return INFER_SOURCE

    def _is_video_worker_running() -> bool:
        with VIDEO_WORKER_LOCK:
            return VIDEO_WORKER_RUNNING

    def _set_video_worker_running(value: bool):
        global VIDEO_WORKER_RUNNING
        with VIDEO_WORKER_LOCK:
            VIDEO_WORKER_RUNNING = bool(value)

    def _get_video_slow_level() -> int:
        with VIDEO_SPEED_LOCK:
            return int(VIDEO_SLOW_LEVEL)

    def _get_infer_fps() -> float:
        return float(INFER_FPS)

    def _get_record_fps() -> float:
        if PROCESSING_MODE == "record_mode":
            return float(RECORD_MODE_CAPTURE_FPS)
        return 30.0

    def _get_frame_generation(cam_state: CameraState) -> int:
        return int(getattr(cam_state.shared, "frame_generation", 0))

    def _read_video_frame() -> tuple[bool, np.ndarray | None]:
        global INFER_VIDEO_CAP, VIDEO_READ_FAILS, VIDEO_ENDED

        with VIDEO_CAP_LOCK:
            if INFER_VIDEO_CAP is None or not INFER_VIDEO_CAP.isOpened():
                INFER_VIDEO_CAP = _open_infer_video_capture(INFER_VIDEO_PATH)
                if INFER_VIDEO_CAP is None:
                    logging.error("Failed to open video file: %s", INFER_VIDEO_PATH)
                    time.sleep(1.0)
                    return False, None
            ok, vframe = INFER_VIDEO_CAP.read()

        if not ok or vframe is None:
            VIDEO_READ_FAILS += 1
            # Jetson: FFmpeg backend often breaks after frame 1 — force full reopen with GStreamer candidates.
            if VIDEO_READ_FAILS in (2, 4, 8):
                recovered = _recover_infer_video_capture(INFER_VIDEO_PATH)
                if recovered is not None:
                    return True, recovered[1]

            if VIDEO_READ_FAILS >= INFER_VIDEO_READ_FAIL_MAX:
                looped = _loop_infer_video_from_start(INFER_VIDEO_PATH)
                if looped is not None:
                    return True, looped[1]
                VIDEO_ENDED = True
                logging.error(
                    "Infer video stopped after %s failures (cannot reopen): %s",
                    VIDEO_READ_FAILS,
                    INFER_VIDEO_PATH,
                )
                return False, None

            time.sleep(0.02)
            return True, None

        VIDEO_READ_FAILS = 0
        return True, vframe

    return pipeline.PipelineHooks(
        run_inference=run_inference,
        recorder_write=recorder_write,
        should_record_raw=_should_record_raw,
        monitor_event=monitor_event,
        get_infer_source=_get_infer_source,
        read_video_frame=_read_video_frame,
        handle_video_eof=handle_video_eof,
        is_video_worker_running=_is_video_worker_running,
        set_video_worker_running=_set_video_worker_running,
        get_video_slow_level=_get_video_slow_level,
        get_infer_fps=_get_infer_fps,
        get_record_fps=_get_record_fps,
        get_frame_generation=_get_frame_generation,
    )


def _infer_worker(cam_state: CameraState):
    """Legacy wrapper; inference loop now lives in pipeline.py."""
    pipeline.infer_worker_loop(cam_state, _build_pipeline_hooks())

def pipeline_loop(cam_state: CameraState):
    """Grab loop wrapper; detailed flow is in pipeline.py::camera_pipeline_loop."""
    pipeline.camera_pipeline_loop(cam_state, grab_1080p, _build_pipeline_hooks())

VIDEO_WORKER_RUNNING = False
VIDEO_WORKER_THREAD = None
VIDEO_WORKER_STARTED_TS = 0.0
VIDEO_WORKER_LOCK = threading.Lock()
VIDEO_SPEED_LOCK = threading.Lock()
# 0 = normal speed, 10 = slowest (more delay between frames)
VIDEO_SLOW_LEVEL = 0
PROCESSING_WORKERS: dict[int, threading.Thread] = {}
PROCESSING_WORKERS_LOCK = threading.Lock()

def video_pipeline_loop(cam_state: CameraState):
    """Video-source loop wrapper; detailed flow is in pipeline.py::video_pipeline_loop."""
    pipeline.video_pipeline_loop(cam_state, _build_pipeline_hooks())


@app.get("/infer/video/speed")
def get_infer_video_speed():
    require_exceed()
    with VIDEO_SPEED_LOCK:
        slow = int(VIDEO_SLOW_LEVEL)
    if slow < 0:
        slow = 0
    if slow > 10:
        slow = 10
    return {"status": "ok", "slow": slow}


@app.post("/infer/video/speed")
async def set_infer_video_speed(request: Request):
    require_exceed()
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        payload = await request.json()
    else:
        form = await request.form()
        payload = dict(form)
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="invalid_payload")
    if "slow" not in payload:
        raise HTTPException(status_code=400, detail="missing_slow")
    try:
        slow_val = int(float(payload["slow"]))
    except Exception:
        raise HTTPException(status_code=400, detail="invalid_slow")
    if slow_val < 0:
        slow_val = 0
    if slow_val > 10:
        slow_val = 10
    with VIDEO_SPEED_LOCK:
        global VIDEO_SLOW_LEVEL
        VIDEO_SLOW_LEVEL = slow_val
    return {"status": "ok", "slow": slow_val}

def start_video_worker():
    """Start a video inference worker on cam 0."""
    global VIDEO_WORKER_RUNNING, VIDEO_WORKER_THREAD, VIDEO_WORKER_STARTED_TS
    with INFER_SOURCE_LOCK:
        infer_source = INFER_SOURCE
    if infer_source != "video":
        logging.warning("start_video_worker skipped: infer_source=%s", infer_source)
        return False
    video_path = os.path.abspath(INFER_VIDEO_PATH)
    if not os.path.isfile(video_path):
        logging.error("start_video_worker failed: missing video source %s", video_path)
        return False
    cam_state = CAMERAS.get(0)
    if cam_state is None:
        logging.error("start_video_worker failed: cam0 state unavailable")
        return False
    with VIDEO_WORKER_LOCK:
        if VIDEO_WORKER_RUNNING:
            t = VIDEO_WORKER_THREAD
            if t is not None and t.is_alive():
                return True
        VIDEO_WORKER_RUNNING = True
        VIDEO_WORKER_STARTED_TS = time.time()
    t = threading.Thread(target=video_pipeline_loop, args=(cam_state,), daemon=True)
    with VIDEO_WORKER_LOCK:
        VIDEO_WORKER_THREAD = t
    t.start()
    logging.info("start_video_worker launched: path=%s", video_path)
    return True

def stop_video_worker():
    """Signal the video worker to stop."""
    global VIDEO_WORKER_RUNNING
    with VIDEO_WORKER_LOCK:
        VIDEO_WORKER_RUNNING = False


def _wait_for_video_output(cam_state: CameraState, timeout_sec: float = 15.0) -> bool:
    """Wait briefly for the video worker to publish a fresh frame."""
    baseline_seq = int(getattr(cam_state.shared, "latest_seq", 0))
    deadline = time.time() + max(timeout_sec, 0.1)
    while time.time() < deadline:
        with VIDEO_WORKER_LOCK:
            running = VIDEO_WORKER_RUNNING
            worker = VIDEO_WORKER_THREAD
        if not running:
            return False
        if worker is not None and not worker.is_alive():
            return False
        if int(getattr(cam_state.shared, "latest_seq", 0)) > baseline_seq and cam_state.shared.latest is not None:
            return True
        time.sleep(0.05)
    return False


def _run_record_mode_processing(
    cam_id: int,
    video_path: str,
    category: str,
    cm_per_px: float,
    thresholds: dict,
):
    """Run offline video processing for record_mode and persist final infer state."""
    cam_state = CAMERAS.get(cam_id)
    if cam_state is None:
        return
    try:
        processing_generation = int(getattr(cam_state.shared, "frame_generation", 0))
        cam_state.shared.processing_active = True
        cam_state.shared.processing_status = "processing"
        cam_state.shared.processing_error = None
        cam_state.shared.processing_started_ts = time.time()
        cam_state.shared.processing_total_frames = 0
        cam_state.shared.processing_done_frames = 0
        try:
            get_models()
        except Exception as exc:
            raise RuntimeError(f"model_load_failed: {exc}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"cannot_open_video: {video_path}")
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        except Exception:
            total_frames = 0
        try:
            src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        except Exception:
            src_fps = 0.0
        if not (src_fps > 0.0):
            src_fps = 30.0
        target_fps = float(max(RECORD_MODE_PROCESS_FPS, 1.0))
        frame_stride = int(max(1, round(src_fps / target_fps)))
        cam_state.shared.processing_total_frames = max(0, total_frames)
        if cam_state.shared.processing_total_frames > 0 and frame_stride > 1:
            cam_state.shared.processing_total_frames = max(
                1, int(cam_state.shared.processing_total_frames // frame_stride)
            )
        local_stats = {}
        summary = {}
        failed_frames = 0
        processed_frames = 0
        aborted_due_to_generation = False
        writer = None
        processed_name = os.path.splitext(os.path.basename(video_path))[0] + "_processed.mp4"
        processed_path = os.path.join(video_dir, processed_name)
        size_cfg = _size_doc_to_legacy_thresholds({
            "selected_category": category,
            "categories": {category: thresholds},
        })
        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                frame_idx = cam_state.shared.processing_done_frames + 1
                # Record-mode processing cap: process only every Nth frame.
                if frame_stride > 1 and ((frame_idx - 1) % frame_stride) != 0:
                    cam_state.shared.processing_done_frames += 1
                    continue
                cam_state.shared.processing_done_frames += 1
                processed_frames += 1
                if int(getattr(cam_state.shared, "frame_generation", 0)) != processing_generation:
                    aborted_due_to_generation = True
                    logging.info(
                        "record_mode processing publish stopped due to generation change cam=%s video=%s",
                        cam_id,
                        video_path,
                    )
                    break
                if frame.shape[:2] != (1080, 1920):
                    frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LINEAR)
                try:
                    annotated, summary = infer.infer_frame_leaf_grouped_tracked(
                        MAIN_MODEL,
                        DEFECT_MODEL,
                        frame,
                        local_stats,
                        size_cfg=size_cfg,
                        cm_per_px=cm_per_px,
                        stem_size=SIZE_STEM_SIZE,
                        size_debug=False,
                        size_debug_color=SIZE_DEBUG_COLOR,
                    )
                    # Publish progress frames so /stream shows processing live.
                    if int(getattr(cam_state.shared, "frame_generation", 0)) == processing_generation:
                        cam_state.shared.latest = annotated
                        try:
                            cam_state.shared.latest_seq = int(getattr(cam_state.shared, "latest_seq", 0)) + 1
                            cam_state.shared.latest_ts = time.time()
                        except Exception:
                            pass
                    if writer is None:
                        h, w = annotated.shape[:2]
                        writer = start_ffmpeg_raw(
                            processed_path,
                            width=int(w),
                            height=int(h),
                            fps=float(src_fps),
                        )
                        if writer is None or writer.poll() is not None:
                            writer = None
                            raise RuntimeError(f"cannot_open_processed_video_writer: {processed_path}")
                    if not recorder_write(writer, annotated):
                        raise RuntimeError(f"processed_video_writer_failed: {processed_path}")
                except Exception as frame_exc:
                    # Continue processing to avoid hard-stop on bad frame.
                    failed_frames += 1
                    if failed_frames <= 5:
                        logging.exception(
                            "record_mode frame processing failed cam=%s frame=%s video=%s",
                            cam_id,
                            cam_state.shared.processing_done_frames,
                            video_path,
                        )
                    elif failed_frames == 6:
                        logging.error(
                            "record_mode frame processing has many failures cam=%s video=%s; suppressing further per-frame logs",
                            cam_id,
                            video_path,
                        )
                    continue
        finally:
            cap.release()
            if writer is not None:
                stop_ffmpeg(writer)
        # Normalize progress for skipped-frame processing
        cam_state.shared.processing_done_frames = int(processed_frames)

        if aborted_due_to_generation:
            cam_state.shared.processing_status = "cancelled"
            return
        if not isinstance(summary, dict) or not summary:
            try:
                summary = infer.compute_tracking_summary(local_stats)
            except Exception:
                summary = {}
        normalized = _normalize_infer_summary(summary if isinstance(summary, dict) else {})
        if cam_state.shared.processing_done_frames > 0 and failed_frames >= cam_state.shared.processing_done_frames:
            raise RuntimeError("all_frames_failed_during_record_mode_processing")
        INFER_STATE.setdefault("stats", {})
        INFER_STATE["stats"][str(cam_id)] = normalized
        INFER_STATE.setdefault("last_video", {})
        chosen_video_path = processed_path if os.path.isfile(processed_path) else video_path
        INFER_STATE["last_video"][str(cam_id)] = {
            "file": os.path.abspath(chosen_video_path),
            "url": f"/media/videos/{os.path.basename(chosen_video_path)}",
            "ts": time.time(),
        }
        save_infer_state(INFER_STATE)
        cam_state.shared.processing_status = "completed"
    except Exception as exc:
        cam_state.shared.processing_status = "error"
        cam_state.shared.processing_error = str(exc)
    finally:
        cam_state.shared.processing_active = False
        if (
            RECORD_MODE_UNLOAD_MODELS_ON_COMPLETE
            and PROCESSING_MODE == "record_mode"
            and (MAIN_MODEL is not None or DEFECT_MODEL is not None)
        ):
            try:
                unload_all_models()
            except Exception:
                pass


def _start_record_mode_processing_worker(
    cam_id: int,
    video_path: str,
    category: str,
    cm_per_px: float,
    thresholds: dict,
):
    """Run offline video processing for record_mode in a background worker."""
    cam_state = CAMERAS.get(cam_id)
    if cam_state is None:
        return

    def _worker():
        with PROCESSING_WORKERS_LOCK:
            if cam_id in PROCESSING_WORKERS:
                # another processing run already active for this camera
                return
            PROCESSING_WORKERS[cam_id] = threading.current_thread()
        try:
            _run_record_mode_processing(cam_id, video_path, category, cm_per_px, thresholds)
        finally:
            with PROCESSING_WORKERS_LOCK:
                PROCESSING_WORKERS.pop(cam_id, None)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

def _start_camera(cam_state: CameraState):
    monitor_event(f"CAMERA_START cam{cam_state.cam_id}")
    _task = monitor_task("camera_start", extra=f"cam={cam_state.cam_id}")
    _task.__enter__()
    def _start_once(fps_override: int | None = None, simple_pipeline: bool = False, profile: str = "default") -> bool:
        cam_state.running = True
        try:
            pipeline_desc = build_pipeline(
                cam_state.cam_id,
                fps_override=fps_override,
                simple_pipeline=simple_pipeline,
            )
            logging.info(
                "Starting camera %s with pipeline profile=%s fps=%s: %s",
                cam_state.cam_id,
                profile,
                int(fps_override or CAMERA_FPS),
                pipeline_desc,
            )
            cam_state.pipeline = Gst.parse_launch(pipeline_desc)
            if cam_state.pipeline is None:
                logging.error(f"Failed to parse GStreamer pipeline for camera {cam_state.cam_id}")
                _stop_camera(cam_state)
                return False
            cam_state.appsink = cam_state.pipeline.get_by_name(f"appsink{cam_state.cam_id}")
            if cam_state.appsink is None:
                logging.error(f"Failed to get appsink from pipeline for camera {cam_state.cam_id}")
                _stop_camera(cam_state)
                return False
            change_ret = cam_state.pipeline.set_state(Gst.State.PLAYING)
            if change_ret == Gst.StateChangeReturn.FAILURE:
                _log_pipeline_bus(cam_state, "set_state")
                logging.error(f"Failed to start pipeline state for camera {cam_state.cam_id}")
                _stop_camera(cam_state)
                return False

            # Give pipeline time to settle, then confirm state.
            state_ret, _, _ = cam_state.pipeline.get_state(2 * Gst.SECOND)
            if state_ret == Gst.StateChangeReturn.FAILURE:
                _log_pipeline_bus(cam_state, "get_state")
                logging.error(f"Pipeline state check failed for camera {cam_state.cam_id}")
                _stop_camera(cam_state)
                return False

            # Start single pipeline thread (grab → infer → stream → record)
            cam_state.pipeline_thread = threading.Thread(target=pipeline_loop, args=(cam_state,), daemon=True)
            cam_state.pipeline_thread.start()
            
            # Wait for at least one frame to confirm healthy startup
            deadline = time.time() + 2.0
            got_frame = False
            while time.time() < deadline:
                if cam_state.shared.latest is not None:
                    got_frame = True
                    break
                time.sleep(0.05)
            
            if not got_frame:
                _log_pipeline_bus(cam_state, "startup_frame_wait")
                logging.error(
                    "Camera %s pipeline started but no frames arrived within timeout",
                    cam_state.cam_id,
                )
                _stop_camera(cam_state)
                return False
            logging.info(f"Camera {cam_state.cam_id} started successfully")
            return True
        except Exception as e:
            _log_pipeline_bus(cam_state, "startup_exception")
            logging.error(f"Error starting camera {cam_state.cam_id}: {e}")
            _stop_camera(cam_state)
            return False

    try:
        if _start_once(profile="default"):
            return True

        # Retry with simplified pipeline (no queue) for transient Argus/NVMM failures.
        for attempt in range(3):
            time.sleep(0.5)
            logging.warning(
                "Camera %s retrying start (simple pipeline, attempt %s/3)",
                cam_state.cam_id,
                attempt + 1,
            )
            if _start_once(simple_pipeline=True, profile="noqueue"):
                return True

        return False
    finally:
        _task.__exit__(None, None, None)

def _stop_camera(cam_state: CameraState):
    monitor_event(f"CAMERA_STOP cam{cam_state.cam_id}")
    with monitor_task("camera_stop", extra=f"cam={cam_state.cam_id}"):
        cam_state.running = False
        if cam_state.pipeline_thread and cam_state.pipeline_thread.is_alive():
            cam_state.pipeline_thread.join(timeout=1.0)
        cam_state.pipeline_thread = None
        if cam_state.pipeline:
            cam_state.pipeline.set_state(Gst.State.NULL)
            cam_state.pipeline = None
        cam_state.appsink = None

def rebuild_cameras():
    for cam_state in CAMERAS.values():
        _stop_camera(cam_state)
    active, physical = init_cameras()
    if not physical:
        logging.warning("No V4L devices for current mode; only logical camera slots exist")
    elif len(physical) == 1 and CAMERA_SETTINGS.get("camera_mode") == "dual":
        logging.warning("Only one camera available; dual camera mode is not possible")
    phys_set = set(physical)
    for cam_state in CAMERAS.values():
        if cam_state.cam_id in phys_set:
            _start_camera(cam_state)

def restart_cameras_in_place():
    for cam_state in CAMERAS.values():
        _stop_camera(cam_state)
    for cam_state in CAMERAS.values():
        _start_camera(cam_state)

def _camera_restart_cooldown():
    global LAST_CAMERA_RESTART_TS
    with CAMERA_RESTART_LOCK:
        now = time.time()
        wait = CAMERA_RESTART_COOLDOWN - (now - LAST_CAMERA_RESTART_TS)
        if wait > 0:
            time.sleep(wait)
        LAST_CAMERA_RESTART_TS = time.time()

def safe_stop_all_cameras():
    _camera_restart_cooldown()
    stop_all_cameras()

def safe_start_all_cameras():
    _camera_restart_cooldown()
    start_all_cameras()

def stop_all_cameras():
    for cam_state in CAMERAS.values():
        _stop_camera(cam_state)

def start_all_cameras():
    physical = set(get_active_cams())
    for cam_state in CAMERAS.values():
        if cam_state.pipeline is not None:
            continue
        if physical and cam_state.cam_id not in physical:
            continue
        _start_camera(cam_state)

def _has_active_camera_pipeline() -> bool:
    return any(cs.pipeline is not None for cs in CAMERAS.values())


def _wait_for_camera_output(timeout_sec: float = 2.0) -> bool:
    """Wait briefly for any camera pipeline to publish a fresh frame."""
    baselines = {
        cam_id: int(getattr(cs.shared, "latest_seq", 0))
        for cam_id, cs in CAMERAS.items()
    }
    deadline = time.time() + max(timeout_sec, 0.1)
    while time.time() < deadline:
        for cam_id, cs in CAMERAS.items():
            if not bool(cs.pipeline is not None and cs.running):
                continue
            if (
                cs.shared.latest is not None
                and int(getattr(cs.shared, "latest_seq", 0)) > baselines.get(cam_id, 0)
            ):
                return True
        time.sleep(0.05)
    return False


def ensure_live_camera_resumed(reason: str, *, force_restart: bool = False) -> bool:
    """
    Best-effort helper to restore a live camera producer after video mode.
    This intentionally does not depend on auto-resume flags because explicit UI
    switches back to camera should always try to recover live preview.
    """
    start_allowed = bool(SERVER_CONFIG.get("start_camera_on_boot", True))
    if not start_allowed:
        logging.info("camera resume skipped (%s): start_camera_on_boot=false", reason)
        return False
    if STABLE_RUNTIME_MODE:
        logging.info("camera resume skipped (%s): stable_runtime_mode=true", reason)
        return _has_active_camera_pipeline()
    if force_restart and _has_active_camera_pipeline():
        safe_stop_all_cameras()
        time.sleep(0.2)
    elif _has_active_camera_pipeline():
        return _wait_for_camera_output(timeout_sec=1.0)
    time.sleep(0.5)
    safe_start_all_cameras()
    ok = _wait_for_camera_output(timeout_sec=2.0)
    logging.info("camera resume result (%s): %s", reason, ok)
    return ok

def get_latest_frame(cam_state: CameraState):
    """Get latest output frame (raw or inferred) for capture/streaming."""
    output = cam_state.shared.latest
    if output is None:
        return None
    
    # Encode to JPEG for backward compatibility with streaming
    with JPEG_LOCK:
        ok, jpeg = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return None
    return jpeg.tobytes()

def _log_infer_error(cam_state: CameraState, message: str, exc: Exception):
    now = time.time()
    if (now - cam_state.last_infer_error_ts) >= 5.0:
        logging.exception("%s: %s", message, exc)
        cam_state.last_infer_error_ts = now

def start_recording(cam_state: CameraState, mode: str) -> bool:
    """Start recording. Sets recording flags in shared state."""
    monitor_event(f"RECORD_START cam{cam_state.cam_id} mode={mode}")
    with monitor_task("record_start", extra=f"cam={cam_state.cam_id} mode={mode}"):
        with cam_state.record_lock:
            suffix = "infer" if mode == "infer" else "raw"
            filename = "top.mp4" if cam_state.cam_id == 0 else "bottom.mp4"
            if suffix == "infer":
                name, ext = os.path.splitext(filename)
                filename = f"{name}_infer{ext}"
            if PROCESSING_MODE == "record_mode" and mode == "infer" and cam_state.cam_id == 0:
                cam_state.current_video_path = os.path.abspath(RECORD_MODE_CAM0_INFER_PATH)
            else:
                cam_state.current_video_path = f"{video_dir}/{filename}"
            
            # Only start FFmpeg if not already running
            if cam_state.ffmpeg_process is None or cam_state.ffmpeg_process.poll() is not None:
                cam_state.ffmpeg_process = start_ffmpeg_raw(cam_state.current_video_path)
            
            cam_state.shared.recording = True
    
    return True

def stop_recording(cam_state: CameraState) -> bool:
    """Stop recording. Clears recording flags and stops FFmpeg."""
    monitor_event(f"RECORD_STOP cam{cam_state.cam_id}")
    with monitor_task("record_stop", extra=f"cam={cam_state.cam_id}"):
        with cam_state.record_lock:
            if cam_state.ffmpeg_process is None:
                return False
            
            cam_state.shared.recording = False
            stop_ffmpeg(cam_state.ffmpeg_process)
            cam_state.ffmpeg_process = None
        
        persist_stats_now(cam_state.cam_id)
    return True

def stop_all_recordings():
    for cam_state in CAMERAS.values():
        stop_recording(cam_state)

def handle_video_eof():
    global INFER_VIDEO_CAP, STREAM_MODE, INFER_SOURCE
    with VIDEO_CAP_LOCK:
        if INFER_VIDEO_CAP:
            INFER_VIDEO_CAP.release()
            INFER_VIDEO_CAP = None
    for cam_state in CAMERAS.values():
        cam_state.shared.inferring = False
        try:
            cam_state.shared.frame_generation = int(getattr(cam_state.shared, "frame_generation", 0)) + 1
            cam_state.shared.latest = None
            cam_state.shared.latest_raw = None
            cam_state.shared.latest_seq = int(getattr(cam_state.shared, "latest_seq", 0)) + 1
            cam_state.shared.latest_ts = time.time()
        except Exception:
            pass
    stop_all_recordings()
    # Persist stats on video end
    last_video = INFER_STATE.get("last_video", {})
    reset_infer_state_file()
    INFER_STATE["last_video"] = last_video
    save_infer_state(INFER_STATE)
    for cam_id in CAMERAS.keys():
        persist_stats_now(cam_id)
    reset_track_stats()
    # Reset mode and source back to camera
    with STREAM_MODE_LOCK:
        STREAM_MODE = "raw"
    with INFER_SOURCE_LOCK:
        INFER_SOURCE = "camera"
    resumed = ensure_live_camera_resumed(
        "video_eof",
        force_restart=not bool(SERVER_CONFIG.get("auto_resume_camera_after_video_infer", False)),
    )
    if not resumed:
        logging.info("Video EOF: live camera not resumed immediately")


def resume_live_camera_after_video_infer_source():
    """If infer used a video file, switch back to camera and restart Argus.

    Otherwise no thread updates shared.latest and the MJPEG stream stays frozen on the last
    infer frame (Exceed shows Start Assessment but the picture never returns to live).
    """
    global INFER_SOURCE, INFER_VIDEO_CAP, VIDEO_ENDED, VIDEO_READ_FAILS, LAST_SOURCE
    with INFER_SOURCE_LOCK:
        if INFER_SOURCE != "video":
            return
        with VIDEO_CAP_LOCK:
            if INFER_VIDEO_CAP:
                INFER_VIDEO_CAP.release()
                INFER_VIDEO_CAP = None
        INFER_SOURCE = "camera"
        VIDEO_ENDED = False
        VIDEO_READ_FAILS = 0
    with MODE_CHANGE_LOCK:
        LAST_SOURCE = "camera"
    for _cs in CAMERAS.values():
        _cs.shared.latest = None
        _cs.shared.latest_raw = None
    ensure_live_camera_resumed("switch_back_from_video", force_restart=True)


def mjpeg_generator(cam_state: CameraState, stream_token: int):
    """MJPEG wrapper; detailed flow is in pipeline.py::mjpeg_stream_generator."""

    def _on_disconnect(replaced_by_newer: bool):
        # NOTE: client accounting is handled by the StreamingResponse BackgroundTask
        # in /stream. This hook is only for diagnostics.
        if replaced_by_newer:
            monitor_event("STREAM_DISCONNECT (client replaced)")
        else:
            monitor_event("STREAM_DISCONNECT (client left)")

    yield from pipeline.mjpeg_stream_generator(
        cam_state=cam_state,
        stream_token=stream_token,
        on_disconnect=_on_disconnect,
        jpeg_quality=80,
        stream_fps=STREAM_FPS,
    )


async def mjpeg_async_generator(request: Request, cam_state: "CameraState", stream_token: int):
    """Async MJPEG generator that reliably detects client disconnects."""
    interval = 1.0 / float(max(STREAM_FPS, 1.0))
    try:
        while True:
            try:
                if await request.is_disconnected():
                    monitor_event(f"STREAM_DISCONNECT cam{cam_state.cam_id} (request disconnected)")
                    break
            except Exception:
                pass

            with cam_state.stream_lock:
                replaced = cam_state.stream_token != stream_token
                active = cam_state.stream_running
            if replaced or not active:
                if replaced:
                    monitor_event(f"STREAM_DISCONNECT cam{cam_state.cam_id} (token replaced)")
                break

            output = cam_state.shared.latest
            if output is None:
                await asyncio.sleep(0.01)
                continue

            ok, jpeg = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                await asyncio.sleep(0.01)
                continue

            payload = jpeg.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"
            )
            await asyncio.sleep(interval)
    finally:
        with cam_state.stream_lock:
            cam_state.stream_clients = max(0, int(cam_state.stream_clients) - 1)
            cam_state.stream_running = cam_state.stream_clients > 0
        monitor_event(f"STREAM_CLOSED cam{cam_state.cam_id} (clients={cam_state.stream_clients})")
# ── System monitor ──────────────────────────────────────────────────────────
import psutil
import collections

MONITOR_LOG_PATH = os.path.join(os.path.dirname(__file__), "monitor.log")
CONTIG_DIAG_LOG_PATH = os.path.join(os.path.dirname(__file__), "contig_diag.log")

# Event log: other parts of the server append events here, monitor thread writes them out
_monitor_events = collections.deque(maxlen=100)

def monitor_event(event: str):
    """Call from anywhere in server to log an event with next monitor tick."""
    import datetime
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    _monitor_events.append(f"[{ts}] {event}")
    if MONITOR_ENABLED:
        SMART_MONITOR.event(event)

def _monitor_loop():
    """Log system metrics + server state every 5 seconds to monitor.log."""
    import datetime
    monitor_logger = logging.getLogger("monitor")
    monitor_logger.setLevel(logging.INFO)
    monitor_logger.propagate = False
    # Clear on each server start so log only contains current run
    open(MONITOR_LOG_PATH, "w").close()
    fh = logging.handlers.RotatingFileHandler(MONITOR_LOG_PATH, maxBytes=50*1024*1024, backupCount=1)
    fh.setFormatter(logging.Formatter("%(message)s"))
    monitor_logger.addHandler(fh)
    monitor_logger.info("=" * 80)
    monitor_logger.info("SERVER START — " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    monitor_logger.info("=" * 80)

    proc = psutil.Process()
    prev_rss = 0

    while True:
        try:
            cpu_pct = psutil.cpu_percent(interval=1)
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()
            thread_count = threading.active_count()
            proc_cpu = proc.cpu_percent()
            proc_rss = proc.memory_info().rss / (1024 * 1024)

            # Jetson GPU memory — try multiple approaches
            gpu_mem_mb = -1
            try:
                # Method 1: /proc/meminfo NvMapMemUsed (Jetson-specific, no root needed)
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        if "NvMapMemUsed" in line:
                            gpu_mem_mb = int(line.split()[1]) / 1024  # kB to MB
                            break
            except Exception:
                pass
            if gpu_mem_mb < 0:
                try:
                    # Method 2: nvmap debug (needs root)
                    with open("/sys/kernel/debug/nvmap/iovmm/maps", "r") as f:
                        total = sum(int(line.split()[1]) for line in f if len(line.split()) > 1 and line.split()[1].isdigit())
                        gpu_mem_mb = total / (1024 * 1024)
                except Exception:
                    pass

            # Jetson thermal
            temp_c = -1
            try:
                with open("/sys/devices/virtual/thermal/thermal_zone0/temp", "r") as f:
                    temp_c = int(f.read().strip()) / 1000.0
            except Exception:
                pass

            # Track stats sizes
            track_count = 0
            defect_count = 0
            try:
                with TRACK_STATS_LOCK:
                    for s in TRACK_STATS.values():
                        track_count += len(s.get("tracks", {}))
                        defect_count += len(s.get("leaf_defects", {}))
            except Exception:
                pass

            # Server state: what is each camera doing right now
            cam_states = []
            for cid, cs in CAMERAS.items():
                parts = [f"cam{cid}"]
                if cs.shared.inferring:
                    parts.append("INFERRING")
                if cs.shared.recording:
                    parts.append("RECORDING")
                if cs.stream_running:
                    parts.append(f"STREAMING({cs.stream_clients})")
                if cs.pipeline is not None:
                    parts.append("PIPELINE_ON")
                else:
                    parts.append("PIPELINE_OFF")
                cam_states.append("+".join(parts))

            # Models loaded? (both pipelines need main + defect when two-stage is enabled)
            if MAIN_MODEL is not None and DEFECT_MODEL is not None:
                models_status = "LOADED"
            elif MAIN_MODEL is not None:
                models_status = "MAIN_ONLY"
            else:
                models_status = "NOT_LOADED"

            # Infer source
            with INFER_SOURCE_LOCK:
                isrc = INFER_SOURCE

            # RSS delta (catch sudden jumps)
            rss_delta = proc_rss - prev_rss if prev_rss > 0 else 0
            prev_rss = proc_rss

            ts = datetime.datetime.now().strftime("%H:%M:%S")

            # ── Flush queued events ──
            while _monitor_events:
                monitor_logger.info(f"  EVENT  {_monitor_events.popleft()}")

            # ── Main status line ──
            monitor_logger.info(
                f"{ts} | "
                f"cpu={cpu_pct:.0f}% proc={proc_cpu:.0f}% | "
                f"ram={mem.used//(1024*1024)}MB/{mem.total//(1024*1024)}MB ({mem.percent:.0f}%) | "
                f"swap={swap.used//(1024*1024)}MB | "
                f"rss={proc_rss:.0f}MB (Δ{rss_delta:+.0f}MB) | "
                f"gpu={gpu_mem_mb:.0f}MB | temp={temp_c:.0f}C"
            )
            monitor_logger.info(
                f"         "
                f"threads={thread_count} | tracks={track_count} defects={defect_count} | "
                f"models={models_status} | source={isrc} | "
                f"{' | '.join(cam_states)}"
            )

            # ── Warnings ──
            warnings = []
            if cpu_pct > 90:
                warnings.append(f"HIGH CPU {cpu_pct:.0f}%")
            if mem.percent > 85:
                warnings.append(f"HIGH RAM {mem.percent:.0f}%")
            if temp_c > 80:
                warnings.append(f"HIGH TEMP {temp_c:.0f}C")
            if swap.used > 500 * 1024 * 1024:
                warnings.append(f"SWAP {swap.used//(1024*1024)}MB")
            if rss_delta > 100:
                warnings.append(f"RSS JUMP +{rss_delta:.0f}MB")
            if warnings:
                monitor_logger.info(f"  !! WARNING: {' | '.join(warnings)}")

        except Exception as e:
            try:
                monitor_logger.info(f"  !! MONITOR ERROR: {e}")
            except Exception:
                pass
        time.sleep(4)  # ~5s total with the 1s cpu_percent interval


def _ram_log_loop():
    """Periodic RAM/RSS line in main app log (toggle via server_config)."""
    proc = psutil.Process()
    while True:
        try:
            vm = psutil.virtual_memory()
            sm = psutil.swap_memory()
            rss_mb = proc.memory_info().rss / (1024 * 1024)
            msg = (
                f"[RAM] used={vm.used // (1024 * 1024)}MB "
                f"total={vm.total // (1024 * 1024)}MB ({vm.percent:.0f}%) "
                f"rss={rss_mb:.0f}MB swap={sm.used // (1024 * 1024)}MB"
            )
            logging.info(
                "%s",
                msg,
            )
            # Also print to terminal/exced.log even when app logging is file-only.
            print(msg, flush=True)
        except Exception as exc:
            logging.warning("[RAM] logger error: %s", exc)
        time.sleep(max(1.0, float(RAM_LOG_INTERVAL_SEC)))


def _read_meminfo_map() -> dict:
    out = {}
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if ":" not in line:
                    continue
                k, v = line.split(":", 1)
                parts = v.strip().split()
                if not parts:
                    continue
                try:
                    out[k] = int(parts[0])  # kB for most entries
                except ValueError:
                    continue
    except Exception:
        pass
    return out


def _buddy_high_order_summary() -> str:
    """
    Return compact summary of high-order free blocks from /proc/buddyinfo.
    Focus on orders 8-10 (commonly indicative for large contiguous pressure).
    """
    try:
        lines = []
        with open("/proc/buddyinfo", "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if not lines:
            return "na"
        # Format:
        # Node 0, zone   Normal  123 45 6 ... (orders 0..10)
        items = []
        for ln in lines:
            parts = ln.split()
            if len(parts) < 15:
                continue
            zone = parts[3]
            nums = parts[-11:]
            try:
                o8, o9, o10 = int(nums[8]), int(nums[9]), int(nums[10])
            except Exception:
                continue
            items.append(f"{zone}:o8={o8},o9={o9},o10={o10}")
        return " | ".join(items) if items else "na"
    except Exception:
        return "na"


def _contig_diag_loop():
    """Periodic low-level allocator diagnostics for Jetson memory issues."""
    import datetime
    logger = logging.getLogger("contig_diag")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    open(CONTIG_DIAG_LOG_PATH, "w").close()
    fh = logging.handlers.RotatingFileHandler(
        CONTIG_DIAG_LOG_PATH, maxBytes=50 * 1024 * 1024, backupCount=1
    )
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.info("=" * 80)
    logger.info("CONTIG DIAG START — " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 80)

    proc = psutil.Process()
    while True:
        try:
            mi = _read_meminfo_map()
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            rss_mb = proc.memory_info().rss / (1024 * 1024)
            mem_avail_mb = mi.get("MemAvailable", 0) / 1024.0
            mem_free_mb = mi.get("MemFree", 0) / 1024.0
            nvmap_mb = mi.get("NvMapMemUsed", 0) / 1024.0
            cma_total_mb = mi.get("CmaTotal", 0) / 1024.0
            cma_free_mb = mi.get("CmaFree", 0) / 1024.0
            swap_free_mb = mi.get("SwapFree", 0) / 1024.0
            buddy = _buddy_high_order_summary()
            logger.info(
                f"{ts} | avail={mem_avail_mb:.0f}MB free={mem_free_mb:.0f}MB "
                f"nvmap={nvmap_mb:.0f}MB cma={cma_free_mb:.0f}/{cma_total_mb:.0f}MB "
                f"swap_free={swap_free_mb:.0f}MB rss={rss_mb:.0f}MB | buddy {buddy}"
            )
        except Exception as e:
            try:
                logger.info(f"{datetime.datetime.now().strftime('%H:%M:%S')} | diag_error={e}")
            except Exception:
                pass
        time.sleep(CONTIG_DIAG_INTERVAL_SEC)


def _log_startup_runtime_summary():
    """Print the key runtime config summary at server startup."""
    try:
        infer_cfg = config_store.load_infer_config()
    except Exception:
        infer_cfg = {}
    try:
        m1 = config_store._load_model1_config()
    except Exception:
        m1 = {}
    try:
        m2 = config_store._load_model2_config()
    except Exception:
        m2 = {}

    m1_model = m1.get("model", {}) if isinstance(m1.get("model"), dict) else {}
    m2_model = m2.get("model", {}) if isinstance(m2.get("model"), dict) else {}
    m1_cfg = m1_model.get("config", {}) if isinstance(m1_model.get("config"), dict) else {}
    m2_cfg = m2_model.get("config", {}) if isinstance(m2_model.get("config"), dict) else {}
    infer_models = infer_cfg.get("models", {}) if isinstance(infer_cfg, dict) else {}

    # Some deployments (uvicorn + custom logging) may not show logging.info on stdout.
    # Print the same summary so it always appears in the terminal on boot.
    def _p(msg: str):
        try:
            print(msg, flush=True)
        except Exception:
            pass

    _p("========== STARTUP CONFIG SUMMARY ==========")
    _p("API modes: /stream/mode mode={raw|infer} source={camera|video}")
    _p(
        "Modes: "
        f"exceed_enabled={EXCEED_ENABLED} "
        f"stable_runtime_mode={STABLE_RUNTIME_MODE} "
        f"processing_mode={PROCESSING_MODE} "
        f"frame_mode={FRAME_MODE}"
    )
    _p(
        "FPS: "
        f"stream={STREAM_FPS:.1f} "
        f"infer={INFER_FPS:.1f} "
        f"record_capture={RECORD_MODE_CAPTURE_FPS:.1f} "
        f"record_process={RECORD_MODE_PROCESS_FPS:.1f}"
    )
    _p(
        "Model1 (main): "
        f"backend={m1_model.get('backend','')} "
        f"path={m1_model.get('path','')} "
        f"imgsz={m1_cfg.get('imgsz','')} "
        f"type={m1_cfg.get('model_type','')} "
        f"roi={m1_cfg.get('roi_enabled','')}"
    )
    _p(
        "Model2 (defect): "
        f"backend={m2_model.get('backend','')} "
        f"path={m2_model.get('path','')} "
        f"imgsz={m2_cfg.get('imgsz','')} "
        f"rfdetr_num_classes={m2_cfg.get('rfdetr_num_classes','')}"
    )
    wm = str(infer_cfg.get("warmup_main_image") or "").strip()
    wd = str(infer_cfg.get("warmup_defect_image") or "").strip()
    if wm or wd:
        _p(
            "Warmup images (optional): "
            f"main={wm or '(zeros)'} "
            f"defect={wd or wm or '(zeros)'} — set in config/pipeline_config.json infer.*"
        )
    _p("Runtime: legacy_only (deepstream_removed)")
    _p("============================================")

    logging.info("========== STARTUP CONFIG SUMMARY ==========")
    logging.info("API modes: /stream/mode mode={raw|infer} source={camera|video}")
    logging.info(
        "Modes: exceed_enabled=%s stable_runtime_mode=%s processing_mode=%s frame_mode=%s",
        EXCEED_ENABLED,
        STABLE_RUNTIME_MODE,
        PROCESSING_MODE,
        FRAME_MODE,
    )
    logging.info(
        "FPS: stream=%.1f infer=%.1f record_capture=%.1f record_process=%.1f",
        STREAM_FPS,
        INFER_FPS,
        RECORD_MODE_CAPTURE_FPS,
        RECORD_MODE_PROCESS_FPS,
    )
    logging.info(
        "Model1 (main): backend=%s path=%s imgsz=%s type=%s roi=%s",
        m1_model.get("backend", ""),
        m1_model.get("path", ""),
        m1_cfg.get("imgsz", ""),
        m1_cfg.get("model_type", ""),
        m1_cfg.get("roi_enabled", ""),
    )
    logging.info(
        "Model2 (defect): backend=%s path=%s imgsz=%s rfdetr_num_classes=%s",
        m2_model.get("backend", ""),
        m2_model.get("path", ""),
        m2_cfg.get("imgsz", ""),
        m2_cfg.get("rfdetr_num_classes", ""),
    )
    logging.info(
        "Legacy models: main=%s (backend=%s) defect=%s (backend=%s)",
        infer_models.get("main", m1_model.get("path", "")),
        m1_model.get("backend", ""),
        infer_models.get("defect", m2_model.get("path", "")),
        m2_model.get("backend", ""),
    )
    logging.info("Runtime: legacy_only (deepstream_removed)")
    logging.info("============================================")


def _start_live_plot_gui_if_enabled():
    """Launch external realtime plot GUI process when enabled."""
    global LIVE_PLOT_GUI_PROCESS
    if not LIVE_PLOT_GUI_ENABLED:
        logging.info("Live plot GUI disabled (live_plot_gui_enabled=false)")
        return
    if LIVE_PLOT_GUI_PROCESS is not None and LIVE_PLOT_GUI_PROCESS.poll() is None:
        return
    script_path = os.path.join(os.path.dirname(__file__), "scripts", "realtime_monitor_gui.py")
    if not os.path.isfile(script_path):
        logging.warning("Live plot GUI script missing: %s", script_path)
        return
    try:
        LIVE_PLOT_GUI_PROCESS = subprocess.Popen(
            [sys.executable, script_path],
            cwd=os.path.dirname(__file__),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logging.info("Live plot GUI started (pid=%s)", LIVE_PLOT_GUI_PROCESS.pid)
    except Exception as exc:
        logging.warning("Failed to start live plot GUI: %s", exc)


def _stop_live_plot_gui():
    """Stop external realtime plot GUI process if running."""
    global LIVE_PLOT_GUI_PROCESS
    proc = LIVE_PLOT_GUI_PROCESS
    LIVE_PLOT_GUI_PROCESS = None
    if proc is None:
        return
    try:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=2.0)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


@app.on_event("startup")
def start_pipeline():
    if MONITOR_ENABLED:
        if RAM_KILL_PCT > 0:
            SMART_MONITOR.set_ram_kill_handler(_graceful_ram_kill_handler)
            logging.info("RAM kill enabled (RAM_KILL_PCT=%.1f%%)", RAM_KILL_PCT)
        else:
            logging.info("RAM kill disabled (RAM_KILL_PCT=0)")
        SMART_MONITOR.start()
        logging.info(
            "Smart monitor enabled -> monitor.log (interval=%ss)",
            MONITOR_INTERVAL_SEC,
        )
    else:
        logging.info("Smart monitor disabled (monitor_enabled=false)")

    if CONTIG_DIAG_ENABLED:
        threading.Thread(target=_contig_diag_loop, daemon=True).start()
        logging.info(
            "Contiguous memory diagnostics enabled -> contig_diag.log (interval=%ss)",
            CONTIG_DIAG_INTERVAL_SEC,
        )
    else:
        logging.info("Contiguous memory diagnostics disabled (contig_diag_enabled=false)")

    if RAM_LOG_ENABLED:
        threading.Thread(target=_ram_log_loop, daemon=True).start()
        logging.info(
            "RAM logger enabled in main log (interval=%ss)",
            RAM_LOG_INTERVAL_SEC,
        )
    else:
        logging.info("RAM logger disabled (ram_log_enabled=false)")

    # Log settings loaded from unified app config.
    try:
        file_settings = config_store.load_camera_settings()
        logging.info("Camera settings from config/camera_config.json: %s", file_settings)
    except Exception as exc:
        logging.warning("Failed to read camera settings from app config: %s", exc)
    logging.info("Camera settings at startup: %s", CAMERA_SETTINGS)
    _log_startup_runtime_summary()
    # Legacy monitor loop is disabled when smart monitor is active.
    if not MONITOR_ENABLED:
        threading.Thread(target=_monitor_loop, daemon=True).start()
        logging.info("Legacy monitor started -> monitor.log")

    # Warn if swap is already in use (leftover from previous crash)
    try:
        import psutil
        swap = psutil.swap_memory()
        if swap.used > 100 * 1024 * 1024:  # >100MB swap used
            logging.warning(
                "HIGH SWAP ON STARTUP: %dMB used — previous crash may have left stale allocations. "
                "Consider: sudo swapoff -a && sudo swapon -a",
                swap.used // (1024 * 1024),
            )
    except Exception:
        pass

    # Load models BEFORE cameras — TensorRT needs contiguous GPU memory and
    # must allocate before camera NVMM buffers fragment the shared Jetson memory.
    logging.info("Model config: preload_models=%s, unload_on_stop=%s, exceed_enabled=%s",
                 PRELOAD_MODELS, UNLOAD_ON_STOP, EXCEED_ENABLED)
    if not EXCEED_ENABLED:
        logging.info("Exceed disabled (image mode) — skipping model loading")
    elif PRELOAD_MODELS:
        logging.info("Loading YOLO models at startup (preload_models=true)...")
        try:
            import infer_main as _infer_paths
            _cfg = _infer_paths.get_config()
            _m = _cfg.get("models", {})
            print(
                "[startup] Preloading models:",
                _infer_paths._abs_model_path(_m.get("main", "")),
                "|",
                _infer_paths._abs_model_path(_m.get("defect", "")),
            )
            main_model, defect_model = get_models()
            main_cfg = _cfg.get("main") if isinstance(_cfg.get("main"), dict) else {}
            defect_cfg = _cfg.get("defect") if isinstance(_cfg.get("defect"), dict) else {}

            def _warmup_side(section: dict, default: int = 640) -> int:
                v = section.get("imgsz") if isinstance(section, dict) else None
                try:
                    s = int(v) if v is not None else default
                except (TypeError, ValueError):
                    s = default
                if s <= 0:
                    s = default
                if s % 32 != 0:
                    s = int(max(32, (s // 32) * 32))
                return s

            main_side = _warmup_side(main_cfg, 640)
            defect_side = _warmup_side(defect_cfg, 640)
            wm = str(_cfg.get("warmup_main_image") or "").strip()
            wd = str(_cfg.get("warmup_defect_image") or "").strip()
            dummy_main, main_wsrc = _warmup_bgr_square(wm, main_side, "main")
            defect_path = wd if wd else wm
            dummy_defect, defect_wsrc = _warmup_bgr_square(defect_path, defect_side, "defect")
            logging.info(
                "Warmup: %sx%s %s | defect %sx%s %s (config: infer.warmup_main_image / warmup_defect_image in pipeline_config.json)",
                main_side,
                main_side,
                main_wsrc,
                defect_side,
                defect_side,
                defect_wsrc,
            )
            def _warmup_yolo_main(model, dummy: np.ndarray, imgsz: int) -> None:
                """Match FP32 vs FP16 TensorRT engines: try FP32 predict first, then FP16, then fallbacks."""
                last_exc: Exception | None = None
                attempts: list[tuple[str, dict]] = [
                    ("half=False+imgsz (FP32 path)", {"half": False, "imgsz": imgsz}),
                    ("half=True+imgsz (FP16 engine)", {"half": True, "imgsz": imgsz}),
                    ("imgsz only", {"imgsz": imgsz}),
                    ("engine default", {}),
                ]
                for label, extra in attempts:
                    kwargs = {"source": dummy, "verbose": False, "device": 0, **extra}
                    try:
                        model.predict(**kwargs)
                        logging.info("Warmup main YOLO succeeded (%s)", label)
                        return
                    except TypeError as e:
                        last_exc = e
                        continue
                    except Exception as e:
                        last_exc = e
                        continue
                assert last_exc is not None
                raise last_exc

            _warmup_yolo_main(main_model, dummy_main, main_side)
            try:
                if defect_model.__class__.__name__.startswith("RFDETR"):
                    defect_model.predict(source=dummy_defect, conf=0.25, imgsz=defect_side)
                else:
                    defect_model.predict(source=dummy_defect, verbose=False, device=0, imgsz=defect_side)
            except Exception as de:
                logging.warning("Defect model warmup failed (non-fatal): %s", de)
            logging.info("YOLO models loaded and warmed up — ready for inference")
        except Exception as e:
            logging.error("Failed to load YOLO models at startup: %s", e)
            print(f"[ERROR] YOLO preload failed: {e} — fix config/model1_config.json and config/model2_config.json model.path values, then restart.")
    else:
        logging.info("Skipping model preload (preload_models=false) — will load on first inference")

    logging.info("Starting camera pipeline...")
    active, physical = init_cameras()
    logging.info("Camera slots: %s | V4L devices for mode: %s", active, physical or "none")
    if len(active) == 1:
        logging.info("Single camera slot mode")
    start_boot = bool(SERVER_CONFIG.get("start_camera_on_boot", True))
    if not start_boot:
        logging.warning(
            "start_camera_on_boot=false — skipping nvarguscamerasrc startup (avoids Argus segfault "
            "when camera busy or TensorRT/ISP conflict). Use source=video for file inference; "
            "set true in config/server_config.json (mode_related.start_camera_on_boot) when live camera works, then restart."
        )
    phys_set = set(physical) if physical else set()
    if start_boot:
        for cam_state in CAMERAS.values():
            if cam_state.cam_id not in phys_set:
                continue
            success = _start_camera(cam_state)
            if not success:
                logging.error("Failed to start camera %s", cam_state.cam_id)
    logging.info("Camera pipeline startup complete")
    _start_live_plot_gui_if_enabled()

@app.on_event("shutdown")
def stop_pipeline():
    with monitor_task("app_shutdown"):
        if DEEPSTREAM_SESSION.is_running():
            DEEPSTREAM_SESSION.stop()
        for cam_state in CAMERAS.values():
            # Disable all flags first
            cam_state.shared.inferring = False
            cam_state.shared.recording = False
            
            # Stop stream clients
            with cam_state.stream_lock:
                cam_state.stream_running = False
            
            # Stop recording
            stop_recording(cam_state)
            
            # Stop camera (which stops the pipeline thread)
            _stop_camera(cam_state)
        
        # Clean up models
        unload_all_models()
        
        # Clean up video capture
        global INFER_VIDEO_CAP
        with VIDEO_CAP_LOCK:
            if INFER_VIDEO_CAP:
                INFER_VIDEO_CAP.release()
                INFER_VIDEO_CAP = None
    if MONITOR_ENABLED:
        SMART_MONITOR.stop()
    _stop_live_plot_gui()

# Configure logging to write to frontend_logs.log
logging.basicConfig(
    filename="frontend_logs.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    filemode="w"
)

@app.post("/log")
async def log_message(request: Request):
    data = await request.json()
    message = data.get("message", "No message provided")
    logging.info(message)
    return {"status": "success"}

@app.post("/save-local")
async def save_local(file: UploadFile = File(...), path: str = Form(...)):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(await file.read())
        return {"status": "saved", "path": path}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/camera/settings")
def get_camera_settings():
    return CAMERA_SETTINGS

def _apply_size_nested_update(cfg, payload):
    leaf = payload.get("leaf")
    if isinstance(leaf, dict):
        for dim in ("length", "width"):
            dim_cfg = leaf.get(dim)
            if isinstance(dim_cfg, dict):
                for bound in ("min", "max"):
                    if bound in dim_cfg:
                        cfg["leaf"][dim][bound] = float(dim_cfg[bound])
    stem = payload.get("stem")
    if isinstance(stem, dict):
        dim_cfg = stem.get("length")
        if isinstance(dim_cfg, dict):
            for bound in ("min", "max"):
                if bound in dim_cfg:
                    cfg["stem"]["length"][bound] = float(dim_cfg[bound])

@app.get("/size/config")
def get_size_config():
    with SIZE_CONFIG_LOCK:
        doc = copy.deepcopy(SIZE_CONFIG_DOC) if isinstance(SIZE_CONFIG_DOC, dict) else _default_size_doc()
    return doc

@app.post("/size/config")
async def update_size_config(request: Request):
    global SIZE_CONFIG, SIZE_CM_PER_PX, SIZE_STEM_SIZE, SIZE_DEBUG, SIZE_DEBUG_COLOR, SIZE_CONFIG_DOC
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        payload = await request.json()
    else:
        form = await request.form()
        payload = dict(form)
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="invalid_payload")
    if "size_config" in payload and isinstance(payload.get("size_config"), str):
        try:
            payload["size_config"] = json.loads(payload["size_config"])
        except Exception:
            raise HTTPException(status_code=400, detail="invalid_size_config_json")
    normalized = _normalize_size_payload_to_v2(payload)
    if normalized is None:
        raise HTTPException(status_code=400, detail="invalid_size_config_format")

    with SIZE_CONFIG_LOCK:
        SIZE_CONFIG_DOC = normalized
        SIZE_CM_PER_PX = float(normalized["cm_per_px"])
        SIZE_CONFIG = _size_doc_to_legacy_thresholds(normalized)
        # Keep legacy runtime toggles unchanged for compatibility.
        save_size_config(SIZE_CONFIG, SIZE_CM_PER_PX, SIZE_STEM_SIZE, SIZE_DEBUG, SIZE_DEBUG_COLOR, SIZE_CONFIG_DOC)
        return copy.deepcopy(SIZE_CONFIG_DOC)


@app.get("/processing/mode")
def get_processing_mode():
    return {"processing_mode": PROCESSING_MODE}


@app.post("/processing/mode")
async def set_processing_mode(request: Request):
    global PROCESSING_MODE, SERVER_CONFIG
    # Legacy realtime-only runtime: record_mode removed.
    PROCESSING_MODE = "live_process"
    SERVER_CONFIG["processing_mode"] = "live_process"
    save_config(SERVER_CONFIG)
    return {"processing_mode": "live_process"}

@app.post("/camera/settings")
async def set_camera_settings(request: Request):
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        data = await request.json()
    else:
        form = await request.form()
        # Filter out empty strings from form data (missing fields)
        data = {k: v for k, v in dict(form).items() if v != ""}

    def _int(key):
        v = data.get(key)
        return int(v) if v is not None else None
    def _float(key):
        v = data.get(key)
        return float(v) if v is not None else None
    def _bool(key):
        v = data.get(key)
        if v is None:
            return None
        if isinstance(v, bool):
            return v
        return str(v).lower() in ("true", "1", "yes")
    def _str(key):
        v = data.get(key)
        return str(v) if v is not None else None

    cam = _int("cam") or 0
    wbmode = _int("wbmode")
    exposure_us = _int("exposure_us")
    gain = _float("gain")
    gain_min = _float("gain_min")
    gain_max = _float("gain_max")
    digital_gain = _float("digital_gain")
    digital_gain_min = _float("digital_gain_min")
    digital_gain_max = _float("digital_gain_max")
    aelock = _bool("aelock")
    camera_mode = _str("camera_mode")

    if camera_mode is None and cam not in CAMERAS:
        raise HTTPException(status_code=404, detail="camera_not_available")

    mode_changed = False
    prev_mode = CAMERA_SETTINGS.get("camera_mode", "single")
    if wbmode is not None:
        CAMERA_SETTINGS["wbmode"] = int(wbmode)
    if exposure_us is not None:
        CAMERA_SETTINGS["exposure_us"] = int(exposure_us)
    if gain is not None:
        gain_val = float(gain)
        CAMERA_SETTINGS["gain_min"] = gain_val
        CAMERA_SETTINGS["gain_max"] = gain_val
    if gain_min is not None:
        CAMERA_SETTINGS["gain_min"] = float(gain_min)
    if gain_max is not None:
        CAMERA_SETTINGS["gain_max"] = float(gain_max)
    if digital_gain is not None:
        digital_gain_val = float(digital_gain)
        CAMERA_SETTINGS["digital_gain_min"] = digital_gain_val
        CAMERA_SETTINGS["digital_gain_max"] = digital_gain_val
    if digital_gain_min is not None:
        CAMERA_SETTINGS["digital_gain_min"] = float(digital_gain_min)
    if digital_gain_max is not None:
        CAMERA_SETTINGS["digital_gain_max"] = float(digital_gain_max)
    if aelock is not None:
        CAMERA_SETTINGS["aelock"] = bool(aelock)
    if camera_mode is not None:
        if camera_mode not in ("single", "dual"):
            raise HTTPException(status_code=400, detail="camera_mode_must_be_single_or_dual")
        CAMERA_SETTINGS["camera_mode"] = camera_mode
        mode_changed = (camera_mode != prev_mode)
    # Resolution is fixed at 1080p — ignore resolution parameter if sent

    save_settings(CAMERA_SETTINGS)

    if mode_changed:
        rebuild_cameras()
    else:
        restart_cameras_in_place()

    return {"status": "updated", "settings": CAMERA_SETTINGS}

@app.get("/stream")
def stream(request: Request, cam: int = 0):
    if cam not in CAMERAS:
        raise HTTPException(status_code=404, detail="camera_not_available")
    cam_state = CAMERAS[cam]
    monitor_event(f"STREAM_CONNECT cam{cam} (prev clients={cam_state.stream_clients})")
    with cam_state.stream_lock:
        # NOTE: Do NOT bump stream_token on connect. Bumping the token invalidates
        # existing generators and effectively enforces "only one client".
        # We only bump the token on mode/source transitions (see /stream/mode),
        # so all clients are forced to reconnect when the stream state changes.
        if STREAM_MAX_CLIENTS > 0 and cam_state.stream_clients >= STREAM_MAX_CLIENTS:
            raise HTTPException(status_code=429, detail="too_many_stream_clients")
        token = cam_state.stream_token
        cam_state.stream_clients += 1
        cam_state.stream_running = True

    return StreamingResponse(
        mjpeg_async_generator(request, cam_state, token),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )

@app.get("/stream/mode")
def get_stream_mode():
    with STREAM_MODE_LOCK:
        mode = STREAM_MODE
    with INFER_SOURCE_LOCK:
        infer_source = INFER_SOURCE
    return {
        "mode": mode,
        "infer_source": infer_source,
        "runtime": "legacy",
    }


@app.get("/debug/runtime")
def get_debug_runtime():
    with STREAM_MODE_LOCK:
        stream_mode = STREAM_MODE
    with INFER_SOURCE_LOCK:
        infer_source = INFER_SOURCE
    with VIDEO_WORKER_LOCK:
        video_worker_running = VIDEO_WORKER_RUNNING
        video_worker_started_ts = VIDEO_WORKER_STARTED_TS
        worker = VIDEO_WORKER_THREAD
        video_worker_alive = bool(worker is not None and worker.is_alive())
    with VIDEO_CAP_LOCK:
        video_cap_open = bool(INFER_VIDEO_CAP is not None and INFER_VIDEO_CAP.isOpened())
    cameras = {}
    for cam_id, cam_state in CAMERAS.items():
        cameras[str(cam_id)] = {
            "running": bool(cam_state.running),
            "pipeline_on": bool(cam_state.pipeline is not None),
            "stream_running": bool(cam_state.stream_running),
            "stream_clients": int(cam_state.stream_clients),
            "inferring": bool(cam_state.shared.inferring),
            "recording": bool(cam_state.shared.recording),
            "processing_active": bool(getattr(cam_state.shared, "processing_active", False)),
            "processing_status": str(getattr(cam_state.shared, "processing_status", "idle")),
            "latest_seq": int(getattr(cam_state.shared, "latest_seq", 0)),
            "latest_ts": float(getattr(cam_state.shared, "latest_ts", 0.0)),
            "frame_generation": int(getattr(cam_state.shared, "frame_generation", 0)),
            "has_latest": bool(cam_state.shared.latest is not None),
            "has_latest_raw": bool(cam_state.shared.latest_raw is not None),
            "infer_queue_size": (
                int(cam_state.shared.infer_queue.qsize())
                if hasattr(cam_state.shared, "infer_queue")
                else -1
            ),
        }
    return {
        "status": "ok",
        "stream_mode": stream_mode,
        "infer_source": infer_source,
        "video_source_path": os.path.abspath(INFER_VIDEO_PATH),
        "video_ended": bool(VIDEO_ENDED),
        "video_read_fails": int(VIDEO_READ_FAILS),
        "video_cap_open": video_cap_open,
        "video_worker_running": video_worker_running,
        "video_worker_alive": video_worker_alive,
        "video_worker_started_ts": float(video_worker_started_ts),
        "deepstream_running": False,
        "camera_pipeline_active": bool(_has_active_camera_pipeline()),
        "cameras": cameras,
    }


@app.get("/deepstream/config")
def get_deepstream_config():
    raise HTTPException(status_code=410, detail="deepstream_removed_use_legacy_runtime")


@app.post("/deepstream/config")
async def set_deepstream_config(payload: dict):
    raise HTTPException(status_code=410, detail="deepstream_removed_use_legacy_runtime")


@app.post("/deepstream/preset")
async def apply_deepstream_preset(payload: dict):
    raise HTTPException(status_code=410, detail="deepstream_removed_use_legacy_runtime")


@app.get("/deepstream/launch")
def get_deepstream_launch(source: str = "camera", video_path: str | None = None):
    raise HTTPException(status_code=410, detail="deepstream_removed_use_legacy_runtime")


@app.get("/deepstream/status")
def get_deepstream_status():
    raise HTTPException(status_code=410, detail="deepstream_removed_use_legacy_runtime")

@app.post("/stream/mode")
async def set_stream_mode(
    mode: str = Form(...),
    source: str = Form(None),
    category: str = Form(None),
    size_config: str = Form(None),
    cm_per_px: float = Form(None),
):
    with monitor_task("set_stream_mode", extra=f"mode={mode} source={source}"):
        global STREAM_MODE, LAST_MODE_CHANGE_TS, LAST_MODE, LAST_SOURCE, INFER_SOURCE
        global INFER_VIDEO_CAP, VIDEO_ENDED, VIDEO_READ_FAILS, INFER_VIDEO_PATH
        global SIZE_CONFIG, SIZE_CM_PER_PX, SIZE_CONFIG_DOC
        if mode not in ("raw", "infer"):
            raise HTTPException(status_code=400, detail="mode_must_be_raw_or_infer")
        if source is not None and source not in ("camera", "video"):
            raise HTTPException(status_code=400, detail="source_must_be_camera_or_video")
        if STABLE_RUNTIME_MODE and source == "video":
            logging.warning("stable_runtime_mode=true: forcing source=camera (video source disabled)")
            source = "camera"
        if not EXCEED_ENABLED:
            if mode == "infer" or source == "video" or size_config is not None or cm_per_px is not None:
                raise HTTPException(status_code=403, detail="exceed_disabled")
        parsed_size_thresholds = None
        parsed_cm_per_px = None
        applied_category = None
        if mode == "infer":
            if source not in ("camera", "video"):
                raise HTTPException(status_code=400, detail="source_required_for_infer")
            if category is None or str(category).strip() == "":
                raise HTTPException(status_code=400, detail="category_required_for_infer")
            if cm_per_px is None:
                raise HTTPException(status_code=400, detail="cm_per_px_required_for_infer")
            if size_config is None:
                raise HTTPException(status_code=400, detail="size_config_required_for_infer")
            try:
                parsed_size = json.loads(size_config)
            except Exception:
                raise HTTPException(status_code=400, detail="invalid_size_config_json")
            if not isinstance(parsed_size, dict):
                raise HTTPException(status_code=400, detail="invalid_size_config_format")
            # Backward compatibility: leaf/stem -> default/other
            if "leaf" in parsed_size or "stem" in parsed_size:
                parsed_size = {
                    "default": parsed_size.get("leaf", {}),
                    "other": {
                        "length": (parsed_size.get("stem", {}) if isinstance(parsed_size.get("stem"), dict) else {}).get("length", {}),
                        "width": {"min": 0.0, "max": 2.0},
                    },
                }
            dflt = _sanitize_profile(parsed_size.get("default"))
            other = _sanitize_profile(parsed_size.get("other"))
            if dflt is None or other is None:
                raise HTTPException(status_code=400, detail="invalid_size_config_format")
            parsed_size_thresholds = {"default": dflt, "other": other}
            try:
                parsed_cm_per_px = float(cm_per_px)
            except Exception:
                raise HTTPException(status_code=400, detail="invalid_cm_per_px")
            if parsed_cm_per_px <= 0:
                raise HTTPException(status_code=400, detail="invalid_cm_per_px")
            applied_category = str(category).strip()
        with INFER_SOURCE_LOCK:
            current_source = INFER_SOURCE
        # UX guard: when UI returns to raw/home without explicitly sending source,
        # auto-switch from video back to camera so live preview resumes.
        if mode == "raw" and source is None and current_source == "video":
            source = "camera"
        desired_source = source if source is not None else current_source
        # New infer session must start from clean state/media so UI shows only fresh results.
        if mode == "infer":
            clear_infer_video_artifacts()
            reset_infer_state_file()
            reset_track_stats()
        # Legacy-only runtime: DeepStream path disabled.
        deepstream_requested = False
        # Fast-path: already raw/camera and no active infer/record work.
        # Avoid repeated teardown/restart churn from frequent UI raw-mode posts.
        with STREAM_MODE_LOCK:
            _curr_mode = STREAM_MODE
        if mode == "raw" and desired_source == "camera" and _curr_mode == "raw":
            any_busy = any(
                bool(getattr(cs.shared, "inferring", False))
                or bool(getattr(cs.shared, "recording", False))
                or bool(getattr(cs.shared, "processing_active", False))
                for cs in CAMERAS.values()
            )
            if not any_busy:
                return {"status": "ok", "mode": "raw", "source": "camera", "ignored": True}
        now = time.time()
        with MODE_CHANGE_LOCK:
            if (
                mode == LAST_MODE
                and desired_source == LAST_SOURCE
                and (now - LAST_MODE_CHANGE_TS) < MODE_CHANGE_COOLDOWN
            ):
                return {"status": "ok", "mode": STREAM_MODE, "ignored": True}
            LAST_MODE_CHANGE_TS = now
            LAST_MODE = mode
            LAST_SOURCE = desired_source
        print(f"[DEBUG] MODE_CHANGE mode={mode} source={desired_source}")
        monitor_event(f"MODE_CHANGE mode={mode} source={desired_source}")
        with STREAM_MODE_LOCK:
            STREAM_MODE = mode

        same_camera_source_transition = current_source == "camera" and desired_source == "camera"
        # Always bump generation on mode/source changes so stale infer workers stop.
        # For same-camera raw<->infer transitions, do not clear latest frame buffers to keep
        # stream continuity; just invalidate old workers.
        for _cs in CAMERAS.values():
            try:
                _cs.shared.frame_generation = int(getattr(_cs.shared, "frame_generation", 0)) + 1
                if not same_camera_source_transition:
                    _cs.shared.latest_raw = None
                    _cs.shared.latest_seq = int(getattr(_cs.shared, "latest_seq", 0)) + 1
                    _cs.shared.latest_ts = time.time()
                    # Clear buffered frames so any restart begins from fresh input.
                    if getattr(_cs.shared, "frame_mode", "") == "buffer" and hasattr(_cs.shared, "infer_queue"):
                        while True:
                            try:
                                _cs.shared.infer_queue.get_nowait()
                            except Exception:
                                break
                # When returning to raw/home, never keep publishing the last infer frame.
                # Clearing `latest` forces MJPEG to wait for a fresh live camera frame.
                if mode == "raw":
                    _cs.shared.latest = None
            except Exception:
                pass

        # Force any existing MJPEG client to reconnect on mode changes.
        # Without this, browsers may keep showing the last received frame (appears "frozen")
        # even after inference stops or UI returns to home.
        for _cs in CAMERAS.values():
            try:
                with _cs.stream_lock:
                    if bool(getattr(_cs, "stream_running", False)):
                        _cs.stream_token += 1
            except Exception:
                pass

        if source is not None:
            # Stop video worker if switching away from video
            if source != "video":
                stop_video_worker()
            with INFER_SOURCE_LOCK:
                if source == "video":
                    # Same resolution as module init (`INFER_VIDEO_PATH`): honor infer.video_source_path,
                    # then fall back to sample.mp4 / first readable local file.
                    candidate = _load_infer_video_path()
                    if not _is_readable_video_file(candidate):
                        raise HTTPException(
                            status_code=400,
                            detail=f"video_source_unreadable: {candidate}",
                        )
                    INFER_VIDEO_PATH = candidate
                INFER_SOURCE = source
                VIDEO_ENDED = False
                VIDEO_READ_FAILS = 0
                with VIDEO_CAP_LOCK:
                    if INFER_VIDEO_CAP:
                        INFER_VIDEO_CAP.release()
                        INFER_VIDEO_CAP = None
            if not STABLE_RUNTIME_MODE:
                if source == "video":
                    safe_stop_all_cameras()
                else:
                    skip_python_camera = False
                    if not skip_python_camera:
                        if not ensure_live_camera_resumed("explicit_source_camera", force_restart=False):
                            logging.warning("explicit_source_camera: no live camera output after source switch")
        else:
            # UI commonly sends only `mode` and omits `source`.
            # If effective source is camera but pipelines were previously stopped
            # (e.g., after video infer), ensure live camera resumes in raw mode.
            if (
                mode == "raw"
                and desired_source == "camera"
                and not STABLE_RUNTIME_MODE
                and bool(SERVER_CONFIG.get("start_camera_on_boot", True))
                and not _has_active_camera_pipeline()
            ):
                if not ensure_live_camera_resumed("implicit_raw_camera_resume", force_restart=False):
                    logging.warning("implicit_raw_camera_resume: no live camera output after raw switch")

        try:
            with INFER_SOURCE_LOCK:
                _src = INFER_SOURCE
            with VIDEO_WORKER_LOCK:
                _vw = VIDEO_WORKER_RUNNING
            _cam_on = _has_active_camera_pipeline()
            logging.info(
                "mode_switch snapshot: mode=%s desired_source=%s infer_source=%s video_worker=%s camera_pipeline=%s",
                mode,
                desired_source,
                _src,
                _vw,
                _cam_on,
            )
        except Exception:
            pass
        # DeepStream path (feature flag): execute gst-launch runtime and skip Python inference workers.
        if deepstream_requested:
            stop_video_worker()
            # Ensure Argus/OpenCV camera pipelines do not compete for the same camera.
            if not STABLE_RUNTIME_MODE:
                safe_stop_all_cameras()
            for cam_state in CAMERAS.values():
                if parsed_size_thresholds is not None:
                    setattr(cam_state.shared, "size_thresholds", copy.deepcopy(parsed_size_thresholds))
                if applied_category is not None:
                    setattr(cam_state.shared, "active_category", applied_category)
                if parsed_cm_per_px is not None:
                    setattr(cam_state.shared, "cm_per_px", float(parsed_cm_per_px))
                cam_state.shared.session_mode = PROCESSING_MODE
                cam_state.shared.processing_active = False
                cam_state.shared.processing_error = None
                cam_state.shared.processing_status = "running"
                cam_state.shared.processing_total_frames = 0
                cam_state.shared.processing_done_frames = 0
                cam_state.shared.processing_started_ts = 0.0
                cam_state.shared.inferring = False
                cam_state.shared.recording = bool(PROCESSING_MODE == "live_process")
            with INFER_SOURCE_LOCK:
                ds_video = INFER_VIDEO_PATH if desired_source == "video" else None
            ds_record_output = None
            if PROCESSING_MODE == "record_mode":
                ts = int(time.time() * 1000)
                ds_record_output = os.path.join(video_dir, f"top_infer_ds_{ts}.mp4")
            cam0 = CAMERAS.get(0)

            def _publish_deepstream_frame(frame: np.ndarray):
                """Publish DeepStream camera frames to MJPEG + optional recorder."""
                if desired_source != "camera":
                    return
                if cam0 is None or frame is None:
                    return
                cam0.shared.latest_raw = frame
                cam0.shared.latest = frame
                try:
                    cam0.shared.latest_seq = int(getattr(cam0.shared, "latest_seq", 0)) + 1
                    cam0.shared.latest_ts = time.time()
                except Exception:
                    pass
                if bool(getattr(cam0.shared, "recording", False)) and _record_frame_due(cam0, RECORD_MODE_CAPTURE_FPS):
                    if not recorder_write(cam0.ffmpeg_process, frame):
                        cam0.shared.recording = False
                        logging.warning("cam %s: recorder write failed, stopping record", cam0.cam_id)

            # For live camera DeepStream, record from bridged frames.
            if PROCESSING_MODE == "live_process" and desired_source == "camera":
                for _cid, _cs in CAMERAS.items():
                    _cs.shared.recording = True
                    start_recording(_cs, "infer")
            try:
                started = DEEPSTREAM_SESSION.start(
                    source=desired_source,
                    video_path=ds_video,
                    record_output_path=ds_record_output,
                    publish_frame_fn=_publish_deepstream_frame,
                )
            except Exception as exc:
                logging.error("DeepStream start failed, falling back to legacy runtime: %s", exc)
                monitor_event(f"DEEPSTREAM_START_FAILED_FALLBACK source={desired_source} err={exc}")
                try:
                    DEEPSTREAM_SESSION.stop()
                except Exception:
                    pass
                # If DeepStream camera failed after we stopped legacy camera, recover it now
                # so the legacy infer path can continue in the same request.
                if desired_source == "camera" and not STABLE_RUNTIME_MODE:
                    ensure_live_camera_resumed("deepstream_start_failed_fallback", force_restart=True)
                # Reset any pre-start recording flags before legacy path takes over.
                for _cs in CAMERAS.values():
                    _cs.shared.recording = False
                stop_all_recordings()
                started = None
            if started is None:
                # Continue into legacy infer path below.
                pass
            else:
                # Live MJPEG: parallel OpenCV reader for file source (DeepStream runs its own decode).
                video_preview = False
                if PROCESSING_MODE == "live_process" and desired_source == "video":
                    cam0 = CAMERAS.get(0)
                    if cam0 is None:
                        raise HTTPException(status_code=500, detail="video_worker_camera_state_missing")
                    if not start_video_worker():
                        raise HTTPException(status_code=500, detail="video_worker_start_failed")
                    if not _wait_for_video_output(cam0, timeout_sec=15.0):
                        with VIDEO_WORKER_LOCK:
                            worker = VIDEO_WORKER_THREAD
                            running = VIDEO_WORKER_RUNNING
                        raise HTTPException(
                            status_code=500,
                            detail=(
                                "video_worker_no_output:"
                                f" running={running}"
                                f" alive={bool(worker is not None and worker.is_alive())}"
                                f" path={os.path.abspath(INFER_VIDEO_PATH)}"
                            ),
                        )
                    video_preview = True
                elif PROCESSING_MODE == "live_process" and desired_source == "camera":
                    # Wait for first bridged frame so /stream doesn't freeze on stale output.
                    baseline_seq = int(getattr(cam0.shared, "latest_seq", 0)) if cam0 is not None else 0
                    deadline = time.time() + 2.0
                    while cam0 is not None and time.time() < deadline:
                        if (
                            cam0.shared.latest is not None
                            and int(getattr(cam0.shared, "latest_seq", 0)) > baseline_seq
                        ):
                            break
                        time.sleep(0.03)
                return {
                    "mode": "infer",
                    "source": desired_source,
                    "processing_mode": PROCESSING_MODE,
                    "session_state": ("recording" if PROCESSING_MODE == "record_mode" else "running"),
                    "runtime": "deepstream",
                    "category": applied_category,
                    "cm_per_px": float(parsed_cm_per_px),
                    "applied_size_config": copy.deepcopy(parsed_size_thresholds),
                    "deepstream": started,
                    "mjpeg_parallel_video_preview": video_preview,
                }
        # Legacy-only runtime: no DeepStream teardown path.
        if mode == "infer":
            for cam_state in CAMERAS.values():
                if parsed_size_thresholds is not None:
                    setattr(cam_state.shared, "size_thresholds", copy.deepcopy(parsed_size_thresholds))
                if applied_category is not None:
                    setattr(cam_state.shared, "active_category", applied_category)
                if parsed_cm_per_px is not None:
                    setattr(cam_state.shared, "cm_per_px", float(parsed_cm_per_px))
                cam_state.shared.session_mode = PROCESSING_MODE
                cam_state.shared.processing_active = False
                cam_state.shared.processing_error = None
                cam_state.shared.processing_status = "running"
                cam_state.shared.processing_total_frames = 0
                cam_state.shared.processing_done_frames = 0
                cam_state.shared.processing_started_ts = 0.0
                start_recording(cam_state, "infer")
                cam_state.shared.inferring = True
            # Persist selected infer run metadata globally for recovery endpoints.
            if parsed_size_thresholds is not None and applied_category is not None and parsed_cm_per_px is not None:
                with SIZE_CONFIG_LOCK:
                    cats = SIZE_CONFIG_DOC.get("categories", {}) if isinstance(SIZE_CONFIG_DOC, dict) else {}
                    if not isinstance(cats, dict):
                        cats = {}
                    cats = copy.deepcopy(cats)
                    cats[applied_category] = copy.deepcopy(parsed_size_thresholds)
                    SIZE_CONFIG_DOC = {
                        "cm_per_px": float(parsed_cm_per_px),
                        "selected_category": applied_category,
                        "categories": cats,
                    }
                    SIZE_CM_PER_PX = float(parsed_cm_per_px)
                    SIZE_CONFIG = _size_doc_to_legacy_thresholds(SIZE_CONFIG_DOC)
                    save_size_config(
                        SIZE_CONFIG,
                        SIZE_CM_PER_PX,
                        SIZE_STEM_SIZE,
                        SIZE_DEBUG,
                        SIZE_DEBUG_COLOR,
                        SIZE_CONFIG_DOC,
                    )
            # Start video worker if source is video
            with INFER_SOURCE_LOCK:
                source_now = INFER_SOURCE
            if source_now == "video":
                cam0 = CAMERAS.get(0)
                if cam0 is None:
                    raise HTTPException(status_code=500, detail="video_worker_camera_state_missing")
                if not start_video_worker():
                    raise HTTPException(status_code=500, detail="video_worker_start_failed")
                if not _wait_for_video_output(cam0, timeout_sec=15.0):
                    with VIDEO_WORKER_LOCK:
                        worker = VIDEO_WORKER_THREAD
                        running = VIDEO_WORKER_RUNNING
                    raise HTTPException(
                        status_code=500,
                        detail=(
                            "video_worker_no_output:"
                            f" running={running}"
                            f" alive={bool(worker is not None and worker.is_alive())}"
                            f" path={os.path.abspath(INFER_VIDEO_PATH)}"
                        ),
                    )
        if mode == "raw":
            stop_video_worker()
            if not STABLE_RUNTIME_MODE:
                resume_live_camera_after_video_infer_source()
                # Hard guarantee for "Home/raw" UX: always try to restore a producing
                # camera pipeline so MJPEG doesn't sit on a frozen last frame.
                if desired_source == "camera":
                    if not ensure_live_camera_resumed("raw_mode_resume_check", force_restart=False):
                        ensure_live_camera_resumed("raw_mode_resume_force", force_restart=True)
            was_inferring = any(cs.shared.inferring for cs in CAMERAS.values())
            was_recording = any(cs.shared.recording for cs in CAMERAS.values())
            for cam_state in CAMERAS.values():
                persist_stats_now(cam_state.cam_id)
                cam_state.shared.inferring = False
                cam_state.shared.recording = False
                cam_state.shared.processing_status = "idle"
            stop_all_recordings()
            # Drop in-memory tracking dicts (tracks, leaf_defects, sizes, …). Pruning only
            # runs during inference; after stop those dicts never shrink and RSS creeps each session.
            if was_inferring:
                reset_track_stats()
                # Best-effort: encourage returning freed memory to OS. Python/NumPy often
                # keep arenas/caches, so RSS may not drop without this.
                try:
                    import gc

                    gc.collect()
                except Exception:
                    pass
                try:
                    import ctypes

                    ctypes.CDLL("libc.so.6").malloc_trim(0)
                except Exception:
                    pass
            if was_inferring and UNLOAD_ON_STOP and (MAIN_MODEL is not None or DEFECT_MODEL is not None):
                # Wait for infer worker to finish current frame before unloading
                time.sleep(0.3)
                monitor_event("MODELS_UNLOAD — unload_on_stop=true, freeing GPU+RAM")
                unload_all_models()
        if mode == "infer":
            return {
                "mode": "infer",
                "source": desired_source,
                "processing_mode": PROCESSING_MODE,
                "session_state": "running",
                "category": applied_category,
                "cm_per_px": float(parsed_cm_per_px),
                "applied_size_config": copy.deepcopy(parsed_size_thresholds),
            }
        # raw mode response (+ optional recording info)
        INFER_STATE.setdefault("last_video", {})
        for _cid, _cs in CAMERAS.items():
            _path = getattr(_cs, "current_video_path", None)
            if _path:
                _fn = os.path.basename(_path)
                INFER_STATE["last_video"][str(_cid)] = {
                    "file": os.path.abspath(_path),
                    "url": f"/media/videos/{_fn}",
                    "ts": time.time(),
                }
        save_infer_state(INFER_STATE)

        recording = None
        last_video_cam0 = (
            INFER_STATE.get("last_video", {}).get("0")
            if isinstance(INFER_STATE.get("last_video", {}), dict)
            else None
        )
        if isinstance(last_video_cam0, dict):
            lv_url = last_video_cam0.get("url")
            lv_file = os.path.basename(str(last_video_cam0.get("file", ""))) if last_video_cam0.get("file") else None
            if lv_url or lv_file:
                recording = {"url": lv_url, "file": lv_file}
        if recording is None:
            cam0 = CAMERAS.get(0)
            if cam0 is not None and cam0.current_video_path:
                fn = os.path.basename(cam0.current_video_path)
                recording = {"url": f"/media/videos/{fn}", "file": fn}

        resp = {
            "mode": "raw",
            "source": desired_source,
            "processing_mode": PROCESSING_MODE,
            "session_state": "idle",
        }
        if recording is not None:
            resp["recording"] = recording
        return resp

@app.post("/infer/start")
def infer_start(cam: int = 0):
    """Start inference on a specific camera."""
    require_exceed()
    if cam not in CAMERAS:
        raise HTTPException(status_code=404, detail="camera_not_available")
    
    with monitor_task("infer_start", extra=f"cam={cam}"):
        cam_state = CAMERAS[cam]
        reset_infer_state_file()
        reset_track_stats()

        # Inference always records
        monitor_event(f"INFER_START cam{cam} — loading models + recording")
        cam_state.shared.inferring = True
        cam_state.shared.recording = True
        start_recording(cam_state, "infer")
        
        return {
            "status": "inference_started",
            "file": os.path.abspath(cam_state.current_video_path),
            "url": f"/media/videos/{os.path.basename(cam_state.current_video_path)}",
        }

def _infer_assessment_payload(cam: int = 0) -> dict:
    """Same JSON shape as GET /infer/stats; no exceed guard. Used for RAM-shutdown snapshot."""
    state = load_infer_state()
    summary = _normalize_infer_summary(state.get("stats", {}).get(str(cam), {}))
    if summary == _normalize_infer_summary({}):
        with TRACK_STATS_LOCK:
            live = TRACK_STATS.get(cam)
            if isinstance(live, dict):
                fallback = _normalize_infer_summary(_build_fallback_summary_from_stats(live))
                if fallback != _normalize_infer_summary({}):
                    summary = fallback
                    INFER_STATE.setdefault("stats", {})
                    INFER_STATE["stats"][str(cam)] = summary
                    save_infer_state(INFER_STATE)
    last_video = state.get("last_video", {}).get(str(cam))
    with SIZE_CONFIG_LOCK:
        fallback_category = str(SIZE_CONFIG_DOC.get("selected_category", DEFAULT_SELECTED_CATEGORY))
        fallback_cm = float(SIZE_CONFIG_DOC.get("cm_per_px", SIZE_CM_PER_PX))
    cam_state = CAMERAS.get(cam)
    active_category = (
        str(getattr(cam_state.shared, "active_category", fallback_category))
        if cam_state is not None
        else fallback_category
    )
    try:
        active_cm = (
            float(getattr(cam_state.shared, "cm_per_px", fallback_cm))
            if cam_state is not None
            else float(fallback_cm)
        )
    except Exception:
        active_cm = float(fallback_cm)
    processing_progress = None
    if cam_state is not None:
        total = int(getattr(cam_state.shared, "processing_total_frames", 0) or 0)
        done = int(getattr(cam_state.shared, "processing_done_frames", 0) or 0)
        if done < 0:
            done = 0
        if total < 0:
            total = 0
        if total > 0:
            pct = round(min(100.0, max(0.0, (done * 100.0) / total)), 2)
        else:
            pct = 0.0
        processing_progress = {
            "done_frames": done,
            "total_frames": total,
            "percent": pct,
            "started_ts": float(getattr(cam_state.shared, "processing_started_ts", 0.0) or 0.0),
        }
    return {
        "status": "ok",
        "category": active_category,
        "cm_per_px": active_cm,
        "processing_mode": PROCESSING_MODE,
        "session_state": (
            "processing"
            if (cam_state is not None and bool(getattr(cam_state.shared, "processing_active", False)))
            else ("running" if (cam_state is not None and bool(getattr(cam_state.shared, "inferring", False))) else "idle")
        ),
        "processing_error": (
            getattr(cam_state.shared, "processing_error", None) if cam_state is not None else None
        ),
        "processing_progress": processing_progress,
        "summary": summary,
        "last_video": last_video,
        "video_ended": False,
    }


def _graceful_ram_kill_handler(snap: dict) -> None:
    """
    Stop inference/recording, persist stats, write assessment JSON (per-camera /infer/stats shape),
    tear down cameras/models, then exit immediately (avoids Uvicorn waiting on long-lived streams).
    """
    global _RAM_KILL_SHUTDOWN_DONE, INFER_VIDEO_CAP
    with _RAM_KILL_SHUTDOWN_LOCK:
        if _RAM_KILL_SHUTDOWN_DONE:
            return
        _RAM_KILL_SHUTDOWN_DONE = True

    monitor_event("RAM_KILL — graceful shutdown: stopping inference and saving assessment")
    logging.error(
        "RAM critical %.1f%% — stopping inference, persisting assessment, exiting",
        float(snap.get("ram_pct", 0.0)),
    )

    try:
        stop_video_worker()
    except Exception:
        logging.exception("stop_video_worker during RAM kill")

    try:
        if DEEPSTREAM_SESSION.is_running():
            DEEPSTREAM_SESSION.stop()
    except Exception:
        logging.exception("DeepStream stop during RAM kill")

    try:
        for cam_state in CAMERAS.values():
            with cam_state.stream_lock:
                cam_state.stream_running = False
            cam_state.shared.inferring = False
            cam_state.shared.recording = False
    except Exception:
        logging.exception("Stopping infer/recording flags during RAM kill")

    try:
        stop_all_recordings()
    except Exception:
        logging.exception("stop_all_recordings during RAM kill")

    time.sleep(0.45)

    try:
        for cam_id in list(CAMERAS.keys()):
            persist_stats_now(cam_id)
    except Exception:
        logging.exception("persist_stats_now during RAM kill")

    assessments = {}
    try:
        for cam_id in sorted(CAMERAS.keys()):
            assessments[str(cam_id)] = _infer_assessment_payload(cam_id)
    except Exception as exc:
        assessments["error"] = str(exc)

    payload = {
        "status": "ram_shutdown",
        "ram_snapshot": snap,
        "assessments": assessments,
        "assessment_file": RAM_SHUTDOWN_ASSESSMENT_FILE,
        "ts_unix": time.time(),
    }

    try:
        with open(RAM_SHUTDOWN_ASSESSMENT_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        logging.exception("Failed to write %s", RAM_SHUTDOWN_ASSESSMENT_FILE)

    try:
        INFER_STATE["last_ram_shutdown"] = {
            "ts_unix": payload["ts_unix"],
            "ram_pct": snap.get("ram_pct"),
            "assessment_file": RAM_SHUTDOWN_ASSESSMENT_FILE,
        }
        save_infer_state(INFER_STATE)
    except Exception:
        logging.exception("Failed to persist last_ram_shutdown to infer state")

    try:
        reset_track_stats()
    except Exception:
        logging.exception("reset_track_stats during RAM kill")

    try:
        unload_all_models()
    except Exception:
        logging.exception("unload_all_models during RAM kill")

    try:
        for cam_state in CAMERAS.values():
            _stop_camera(cam_state)
    except Exception:
        logging.exception("_stop_camera during RAM kill")

    try:
        with VIDEO_CAP_LOCK:
            if INFER_VIDEO_CAP:
                INFER_VIDEO_CAP.release()
                INFER_VIDEO_CAP = None
    except Exception:
        logging.exception("INFER_VIDEO_CAP release during RAM kill")

    done_msg = (
        f"RAM shutdown complete — assessment saved to {RAM_SHUTDOWN_ASSESSMENT_FILE} "
        "(per-camera object matches GET /infer/stats)"
    )
    logging.error(done_msg)
    print(done_msg, flush=True)
    try:
        SMART_MONITOR.log(done_msg)
    except Exception:
        pass

    try:
        for h in logging.root.handlers:
            try:
                h.flush()
            except Exception:
                pass
    except Exception:
        pass

    os._exit(2)


@app.post("/infer/stop")
def infer_stop(cam: int = 0):
    """Stop inference on a specific camera."""
    require_exceed()
    if cam not in CAMERAS:
        raise HTTPException(status_code=404, detail="camera_not_available")
    
    with monitor_task("infer_stop", extra=f"cam={cam}"):
        if DEEPSTREAM_SESSION.is_running():
            DEEPSTREAM_SESSION.stop()
        cam_state = CAMERAS[cam]
        monitor_event(f"INFER_STOP cam{cam}")
        cam_state.shared.inferring = False
        cam_state.shared.recording = False
        stop_recording(cam_state)
        # Wait for infer worker to finish current frame before unloading
        time.sleep(0.3)

        if UNLOAD_ON_STOP:
            monitor_event("MODELS_UNLOAD — unload_on_stop=true, freeing GPU+RAM after infer/stop")
            unload_all_models()

        # Persist final stats
        for cam_id in CAMERAS.keys():
            persist_stats_now(cam_id)
        reset_track_stats()

        if cam_state.current_video_path:
            filename = os.path.basename(cam_state.current_video_path)
            file_url = os.path.abspath(cam_state.current_video_path)
            url = f"/media/videos/{filename}"
        else:
            file_url = None
            url = None
        
        return {
            "status": "inference_stopped",
            "file": file_url,
            "url": url,
        }

@app.get("/infer/stats")
def infer_stats(cam: int = 0):
    require_exceed()
    return _infer_assessment_payload(cam)

@app.get("/capture")
def capture(request: Request, cam: int = 0):
    if cam not in CAMERAS:
        raise HTTPException(status_code=404, detail="camera_not_available")
    frame = get_latest_frame(CAMERAS[cam])
    if frame is None:
        return {"error": "no frame"}

    filename = "top.jpg" if cam == 0 else "bottom.jpg"
    path = f"{capture_dir}/{filename}"
    with open(path, "wb") as f:
        f.write(frame)

    return {
        "status": "captured",
        "file": os.path.abspath(path),
        "url": f"/media/images/{filename}",
    }

def _record_start(cam: int, mode: str):
    """Start recording - sets recording flag in shared state."""
    if cam not in CAMERAS:
        raise HTTPException(status_code=404, detail="camera_not_available")
    if mode not in ("raw", "infer"):
        raise HTTPException(status_code=400, detail="mode_must_be_raw_or_infer")
    if mode == "infer" and not EXCEED_ENABLED:
        raise HTTPException(status_code=403, detail="exceed_disabled")
    
    cam_state = CAMERAS[cam]
    
    if mode == "infer":
        # Inference recording also enables inference
        cam_state.shared.inferring = True
    
    start_recording(cam_state, mode)
    
    return {
        "status": "recording_started",
        "mode": mode,
        "file": os.path.abspath(cam_state.current_video_path),
        "url": f"/media/videos/{os.path.basename(cam_state.current_video_path)}",
    }

@app.post("/record/start")
def record_start(request: Request, cam: int = 0, mode: str = Form("raw")):
    return _record_start(cam=cam, mode=mode)

@app.get("/record/start")
def record_start_get(cam: int = 0, mode: str = "raw"):
    # GET alias for Imager compatibility
    return _record_start(cam=cam, mode=mode)

def _record_stop(cam: int):
    """Stop recording - clears recording flag and returns to idle state.

    When STREAM_MODE is infer (Exceed assessment), this ends the full session: stop inference
    workers, wait for the last frame to land in stats, stop all cameras' recordings, persist,
    set stream mode to raw, and optionally unload models — same outcome as /stream/mode raw.
    """
    global STREAM_MODE, LAST_MODE_CHANGE_TS, LAST_MODE
    if cam not in CAMERAS:
        raise HTTPException(status_code=404, detail="camera_not_available")

    cam_state = CAMERAS[cam]

    with STREAM_MODE_LOCK:
        ending_infer_session = STREAM_MODE == "infer"

    if ending_infer_session:
        # Frontend may accidentally call /record/stop while infer is running.
        # Safety behavior: stop recording only; keep infer session active.
        monitor_event("RECORD_STOP — infer mode active; stopping recording only")
        stop_recording(cam_state)
        if cam_state.current_video_path:
            filename = os.path.basename(cam_state.current_video_path)
            file_url = os.path.abspath(cam_state.current_video_path)
            url = f"/media/videos/{filename}"
        else:
            file_url = None
            url = None

        INFER_STATE.setdefault("last_video", {})
        INFER_STATE["last_video"][str(cam)] = {
            "file": file_url,
            "url": url,
            "ts": time.time(),
        }
        save_infer_state(INFER_STATE)
        persist_stats_now(cam)
    else:
        stop_recording(cam_state)
        if cam_state.current_video_path:
            filename = os.path.basename(cam_state.current_video_path)
            file_url = os.path.abspath(cam_state.current_video_path)
            url = f"/media/videos/{filename}"
        else:
            file_url = None
            url = None

        INFER_STATE.setdefault("last_video", {})
        INFER_STATE["last_video"][str(cam)] = {
            "file": file_url,
            "url": url,
            "ts": time.time(),
        }
        save_infer_state(INFER_STATE)

        for cam_id in CAMERAS.keys():
            persist_stats_now(cam_id)

    # Response uses the requested camera's output path (same file as before for cam=0).
    if cam_state.current_video_path:
        filename = os.path.basename(cam_state.current_video_path)
        file_url = os.path.abspath(cam_state.current_video_path)
        url = f"/media/videos/{filename}"
    else:
        file_url = None
        url = None

    return {"status": "recording_stopped", "file": file_url, "url": url}

@app.post("/record/stop")
def record_stop(request: Request, cam: int = 0):
    return _record_stop(cam=cam)

@app.get("/record/stop")
def record_stop_get(cam: int = 0):
    # GET alias for Imager compatibility
    return _record_stop(cam=cam)

@app.get("/uisetup")
def get_uisetup():
    with UISETUP_LOCK:
        settings = _read_uisetup_settings()
    return {"success": True, "settings": settings}

@app.post("/uisetup")
async def post_uisetup(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    settings = body.get("settings") if isinstance(body, dict) else None
    if not isinstance(settings, dict):
        raise HTTPException(status_code=400, detail="Missing 'settings' object")

    # Keep loose validation so frontend can evolve.
    for key in ("farm", "server", "developer"):
        if key not in settings or not isinstance(settings.get(key), dict):
            raise HTTPException(status_code=400, detail=f"Missing '{key}' section")

    with UISETUP_LOCK:
        updated_at = _write_uisetup_settings(settings)
    return {"success": True, "updated_at": updated_at}
