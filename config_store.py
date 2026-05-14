import os

from utils import deep_merge_dict, read_json_file, write_json_file


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, "config")
LEGACY_APP_CONFIG_PATH = os.path.join(CONFIG_DIR, "app_config.json")
SERVER_CONFIG_PATH = os.path.join(CONFIG_DIR, "server_config.json")
CAMERA_CONFIG_PATH = os.path.join(CONFIG_DIR, "camera_config.json")
INFER_CONFIG_PATH = os.path.join(CONFIG_DIR, "infer_config.json")
MODEL1_CONFIG_PATH = os.path.join(CONFIG_DIR, "model1_config.json")
MODEL2_CONFIG_PATH = os.path.join(CONFIG_DIR, "model2_config.json")
PIPELINE_CONFIG_PATH = os.path.join(CONFIG_DIR, "pipeline_config.json")
DEEPSTREAM_CONFIG_PATH = os.path.join(CONFIG_DIR, "deepstream_config.json")
INFER_STATE_PATH = os.path.join(CONFIG_DIR, "infer_state.json")
TRACKER_CONFIG_PATH = os.path.join(CONFIG_DIR, "bytetrack.yaml")

DEFAULT_SERVER_CONFIG = {
    "model_related": {
        "exceed_enabled": False,
    },
    "mode_related": {
        "start_camera_on_boot": True,
        "auto_resume_camera_after_video_infer": False,
        "stable_runtime_mode": False,
        "stream_fps": 30,
        "realtime_mode": True,
        "buffer_sizes": {
            "gst_queue": 30,
            "appsink": 30,
            "infer_queue": 30,
        },
    },
    "others": {
        "monitor_enabled": True,
        "monitor_interval_sec": 5,
        "contig_diag_enabled": True,
        "contig_diag_interval_sec": 2,
    },
}

DEFAULT_CAMERA_CONFIG = {
    "wbmode": 3,
    "exposure_us": 10000000,
    "gain_min": 3.0,
    "gain_max": 3.0,
    "digital_gain_min": 1.0,
    "digital_gain_max": 1.0,
    "aelock": True,
    "camera_mode": "single",
}

DEFAULT_MODEL1_CONFIG = {
    "model": {
        "path": "models/best.engine",
        "backend": "yolo",
        "config": {
            "model2_pass_classes": ["all"],
            "object_classes": ["all"],
            "main_defect_classes": [],
            "size_measure_classes": ["all"],
        },
    },
    "classes": {},
}

DEFAULT_MODEL2_CONFIG = {
    "model": {
        "path": "models/defect.engine",
        "backend": "rfdetr",
        "config": {
            "size_measure_enabled": False,
            "size_measure_classes": [],
            "size_measure_thresholds_cm": {
                "length": {"min": 0.0, "max": 0.0},
                "width": {"min": 0.0, "max": 0.0},
            },
            "size_final_logic": "auto",
        },
    },
    "classes": {},
}

DEFAULT_PIPELINE_CONFIG = {
    "infer": {
        "preload_models": True,
        "unload_on_stop": True,
        "video_source_path": "media/videos/local/sample.mp4",
        "tracking_enabled": False,
        # Optional BGR image paths for startup warmup (relative to Server2/ or absolute).
        # Empty = black placeholder. defect falls back to main if defect path is empty.
        "warmup_main_image": "",
        "warmup_defect_image": "",
        # GC + CUDA cache flush interval (frames). Lower = more frequent but more GIL stalls.
        "gc_every_n_frames": 10,
    },
    "size": {},
}

DEFAULT_DEEPSTREAM_CONFIG = {
    "enabled": False,
    "pipeline": {
        "batch_size": 1,
        "width": 1920,
        "height": 1080,
        "batched_push_timeout": 40000,
        "live_source": True,
    },
    "source": {
        "camera": {
            "sensor_id": 0,
            "framerate": "60/1",
            "width": 1920,
            "height": 1080,
            "format": "NV12",
        },
        "video": {
            "loop": False,
            "drop_frame_interval": 0,
        },
    },
    "elements": {
        "streammux": {"name": "mux"},
        "primary_infer": {"name": "pgie", "unique_id": 1},
        "secondary_infer": {"name": "sgie", "unique_id": 2},
        "tracker": {"name": "tracker"},
        "osd": {"name": "osd"},
        "sink": {"name": "sink", "type": "fakesink", "sync": False},
    },
    "models": {
        "main": {
            "enabled": True,
            "backend": "yolo",
            "config_file": "",
            "engine_file": "",
            "labels_file": "",
            "gie_type": "primary",
            "network_type": 0,
            "num_detected_classes": 1,
            "batch_size": 1,
            "interval": 0,
            "custom_lib_path": "",
            "parse_bbox_func_name": "",
            "parse_classifier_func_name": "",
        },
        "defect": {
            "enabled": False,
            "backend": "rfdetr",
            "config_file": "",
            "engine_file": "",
            "labels_file": "",
            "gie_type": "secondary",
            "operate_on_gie_id": 1,
            "network_type": 0,
            "num_detected_classes": 1,
            "batch_size": 1,
            "interval": 0,
            "custom_lib_path": "",
            "parse_bbox_func_name": "",
            "parse_classifier_func_name": "",
        },
    },
}


def _load_json(path: str, default: dict) -> dict:
    data = read_json_file(path, {})
    if not isinstance(data, dict):
        return default.copy()
    return deep_merge_dict(default, data)


def _load_legacy_app_config() -> dict:
    data = read_json_file(LEGACY_APP_CONFIG_PATH, {})
    if not isinstance(data, dict):
        return {}
    return data


def _load_server_from_legacy() -> dict:
    legacy = _load_legacy_app_config()
    section = legacy.get("server", {})
    if not isinstance(section, dict):
        return DEFAULT_SERVER_CONFIG.copy()
    return deep_merge_dict(DEFAULT_SERVER_CONFIG, section)


def _load_camera_from_legacy() -> dict:
    legacy = _load_legacy_app_config()
    section = legacy.get("camera", {})
    if not isinstance(section, dict):
        return DEFAULT_CAMERA_CONFIG.copy()
    return deep_merge_dict(DEFAULT_CAMERA_CONFIG, section)


def _load_infer_file_from_legacy() -> dict:
    legacy = _load_legacy_app_config()
    merged = {"infer": {}, "class_map": {}, "size": {}}
    for key in ("infer", "class_map", "size"):
        v = legacy.get(key, {})
        if isinstance(v, dict):
            merged[key] = v
    return merged


def load_server_config() -> dict:
    if os.path.exists(SERVER_CONFIG_PATH):
        return _load_json(SERVER_CONFIG_PATH, DEFAULT_SERVER_CONFIG)
    return _load_server_from_legacy()


def save_server_config(data: dict):
    payload = data if isinstance(data, dict) else {}
    write_json_file(SERVER_CONFIG_PATH, payload, indent=2)


def load_camera_settings() -> dict:
    if os.path.exists(CAMERA_CONFIG_PATH):
        return _load_json(CAMERA_CONFIG_PATH, DEFAULT_CAMERA_CONFIG)
    return _load_camera_from_legacy()


def save_camera_settings(data: dict):
    payload = data if isinstance(data, dict) else {}
    write_json_file(CAMERA_CONFIG_PATH, payload, indent=2)


def _load_legacy_infer_bundle() -> dict:
    if os.path.exists(INFER_CONFIG_PATH):
        return _load_json(INFER_CONFIG_PATH, {"infer": {}, "class_map": {}, "size": {}})
    return _load_infer_file_from_legacy()


def _ensure_split_infer_files():
    if os.path.exists(MODEL1_CONFIG_PATH) and os.path.exists(MODEL2_CONFIG_PATH) and os.path.exists(PIPELINE_CONFIG_PATH):
        return

    legacy = _load_legacy_infer_bundle()
    infer = legacy.get("infer", {}) if isinstance(legacy.get("infer"), dict) else {}
    class_map = legacy.get("class_map", {}) if isinstance(legacy.get("class_map"), dict) else {}
    size = legacy.get("size", {}) if isinstance(legacy.get("size"), dict) else {}
    models = infer.get("models", {}) if isinstance(infer.get("models"), dict) else {}

    m1 = deep_merge_dict(
        DEFAULT_MODEL1_CONFIG,
        {
            "model": {
                "path": models.get("main", DEFAULT_MODEL1_CONFIG["model"]["path"]),
                "config": infer.get("main", {}),
            },
            "classes": class_map.get("main_model", {}),
        },
    )
    m2 = deep_merge_dict(
        DEFAULT_MODEL2_CONFIG,
        {
            "model": {
                "path": models.get("defect", DEFAULT_MODEL2_CONFIG["model"]["path"]),
                "config": infer.get("defect", {}),
            },
            "classes": class_map.get("defect_model", {}),
        },
    )
    pipe = deep_merge_dict(
        DEFAULT_PIPELINE_CONFIG,
        {
            "infer": {
                "preload_models": infer.get("preload_models", True),
                "unload_on_stop": infer.get("unload_on_stop", True),
                "video_source_path": infer.get("video_source_path", DEFAULT_PIPELINE_CONFIG["infer"]["video_source_path"]),
            },
            "size": size,
        },
    )

    write_json_file(MODEL1_CONFIG_PATH, m1, indent=2)
    write_json_file(MODEL2_CONFIG_PATH, m2, indent=2)
    write_json_file(PIPELINE_CONFIG_PATH, pipe, indent=2)


def _load_model1_config() -> dict:
    _ensure_split_infer_files()
    return _load_json(MODEL1_CONFIG_PATH, DEFAULT_MODEL1_CONFIG)


def _load_model2_config() -> dict:
    _ensure_split_infer_files()
    return _load_json(MODEL2_CONFIG_PATH, DEFAULT_MODEL2_CONFIG)


def _load_pipeline_config() -> dict:
    _ensure_split_infer_files()
    return _load_json(PIPELINE_CONFIG_PATH, DEFAULT_PIPELINE_CONFIG)


def _load_deepstream_config() -> dict:
    return _load_json(DEEPSTREAM_CONFIG_PATH, DEFAULT_DEEPSTREAM_CONFIG)


def _save_model1_config(data: dict):
    write_json_file(MODEL1_CONFIG_PATH, data if isinstance(data, dict) else DEFAULT_MODEL1_CONFIG, indent=2)


def _save_model2_config(data: dict):
    write_json_file(MODEL2_CONFIG_PATH, data if isinstance(data, dict) else DEFAULT_MODEL2_CONFIG, indent=2)


def _save_pipeline_config(data: dict):
    write_json_file(PIPELINE_CONFIG_PATH, data if isinstance(data, dict) else DEFAULT_PIPELINE_CONFIG, indent=2)


def _save_deepstream_config(data: dict):
    write_json_file(
        DEEPSTREAM_CONFIG_PATH,
        data if isinstance(data, dict) else DEFAULT_DEEPSTREAM_CONFIG,
        indent=2,
    )


def load_infer_config() -> dict:
    m1 = _load_model1_config()
    m2 = _load_model2_config()
    pipe = _load_pipeline_config()
    pipe_infer = pipe.get("infer", {}) if isinstance(pipe.get("infer"), dict) else {}
    m1_model = m1.get("model", {}) if isinstance(m1.get("model"), dict) else {}
    m2_model = m2.get("model", {}) if isinstance(m2.get("model"), dict) else {}
    m1_cfg = m1_model.get("config", {}) if isinstance(m1_model.get("config"), dict) else {}
    m2_cfg = m2_model.get("config", {}) if isinstance(m2_model.get("config"), dict) else {}
    m1_backend = str(
        m1_model.get("backend", DEFAULT_MODEL1_CONFIG["model"].get("backend", "yolo"))
    ).strip().lower()
    m2_backend = str(
        m2_model.get("backend", DEFAULT_MODEL2_CONFIG["model"].get("backend", "rfdetr"))
    ).strip().lower()
    main_cfg = m1_cfg.copy()
    main_cfg["backend"] = m1_backend
    defect_cfg = m2_cfg.copy()
    defect_cfg["backend"] = m2_backend
    return {
        "models": {
            "main": m1_model.get("path", DEFAULT_MODEL1_CONFIG["model"]["path"]),
            "defect": m2_model.get("path", DEFAULT_MODEL2_CONFIG["model"]["path"]),
        },
        "main": main_cfg,
        "defect": defect_cfg,
        "preload_models": pipe_infer.get("preload_models", True),
        "unload_on_stop": pipe_infer.get("unload_on_stop", True),
        "video_source_path": pipe_infer.get("video_source_path", DEFAULT_PIPELINE_CONFIG["infer"]["video_source_path"]),
        "tracking_enabled": bool(pipe_infer.get("tracking_enabled", False)),
        "warmup_main_image": str(pipe_infer.get("warmup_main_image", "") or "").strip(),
        "warmup_defect_image": str(pipe_infer.get("warmup_defect_image", "") or "").strip(),
        "gc_every_n_frames": int(pipe_infer.get("gc_every_n_frames", DEFAULT_PIPELINE_CONFIG["infer"]["gc_every_n_frames"])),
    }


def save_infer_config(data: dict):
    d = data if isinstance(data, dict) else {}
    models = d.get("models", {}) if isinstance(d.get("models"), dict) else {}

    m1 = _load_model1_config()
    m2 = _load_model2_config()
    pipe = _load_pipeline_config()

    m1.setdefault("model", {})
    m2.setdefault("model", {})
    m1["model"]["path"] = models.get("main", m1["model"].get("path", DEFAULT_MODEL1_CONFIG["model"]["path"]))
    m2["model"]["path"] = models.get("defect", m2["model"].get("path", DEFAULT_MODEL2_CONFIG["model"]["path"]))
    if isinstance(d.get("main"), dict):
        main = dict(d.get("main", {}))
        backend = main.pop("backend", None)
        if backend is not None:
            m1["model"]["backend"] = str(backend).strip().lower()
        m1["model"]["config"] = main
    if isinstance(d.get("defect"), dict):
        defect = dict(d.get("defect", {}))
        backend = defect.pop("backend", None)
        if backend is not None:
            m2["model"]["backend"] = str(backend).strip().lower()
        m2["model"]["config"] = defect

    pipe.setdefault("infer", {})
    pipe["infer"]["preload_models"] = d.get("preload_models", pipe["infer"].get("preload_models", True))
    pipe["infer"]["unload_on_stop"] = d.get("unload_on_stop", pipe["infer"].get("unload_on_stop", True))
    pipe["infer"]["video_source_path"] = d.get(
        "video_source_path",
        pipe["infer"].get("video_source_path", DEFAULT_PIPELINE_CONFIG["infer"]["video_source_path"]),
    )
    pipe["infer"]["warmup_main_image"] = str(
        d.get("warmup_main_image", pipe["infer"].get("warmup_main_image", "")) or ""
    ).strip()
    pipe["infer"]["warmup_defect_image"] = str(
        d.get("warmup_defect_image", pipe["infer"].get("warmup_defect_image", "")) or ""
    ).strip()
    pipe["infer"]["gc_every_n_frames"] = int(
        d.get("gc_every_n_frames", pipe["infer"].get("gc_every_n_frames", DEFAULT_PIPELINE_CONFIG["infer"]["gc_every_n_frames"]))
    )

    _save_model1_config(m1)
    _save_model2_config(m2)
    _save_pipeline_config(pipe)


def infer_config_mtime() -> float:
    try:
        _ensure_split_infer_files()
        times = []
        for p in (MODEL1_CONFIG_PATH, MODEL2_CONFIG_PATH, PIPELINE_CONFIG_PATH):
            if os.path.exists(p):
                times.append(os.path.getmtime(p))
        if times:
            return max(times)
        if os.path.exists(INFER_CONFIG_PATH):
            return os.path.getmtime(INFER_CONFIG_PATH)
        return os.path.getmtime(LEGACY_APP_CONFIG_PATH)
    except OSError:
        return 0.0


def class_config_mtime() -> float:
    try:
        _ensure_split_infer_files()
        times = []
        for p in (MODEL1_CONFIG_PATH, MODEL2_CONFIG_PATH):
            if os.path.exists(p):
                times.append(os.path.getmtime(p))
        if times:
            return max(times)
        if os.path.exists(INFER_CONFIG_PATH):
            return os.path.getmtime(INFER_CONFIG_PATH)
        return os.path.getmtime(LEGACY_APP_CONFIG_PATH)
    except OSError:
        return 0.0


def load_class_config() -> dict:
    m1 = _load_model1_config()
    m2 = _load_model2_config()
    c1 = m1.get("classes", {}) if isinstance(m1.get("classes"), dict) else {}
    c2 = m2.get("classes", {}) if isinstance(m2.get("classes"), dict) else {}
    return {"main_model": c1.copy(), "defect_model": c2.copy()}


def save_class_config(data: dict):
    d = data if isinstance(data, dict) else {}
    m1 = _load_model1_config()
    m2 = _load_model2_config()
    main_map = d.get("main_model", {})
    defect_map = d.get("defect_model", {})
    m1["classes"] = main_map if isinstance(main_map, dict) else {}
    m2["classes"] = defect_map if isinstance(defect_map, dict) else {}
    _save_model1_config(m1)
    _save_model2_config(m2)


def load_size_config_raw() -> dict:
    pipe = _load_pipeline_config()
    size = pipe.get("size", {})
    return size.copy() if isinstance(size, dict) else {}


def save_size_config_raw(data: dict):
    pipe = _load_pipeline_config()
    pipe["size"] = data if isinstance(data, dict) else {}
    _save_pipeline_config(pipe)


def load_deepstream_config() -> dict:
    return _load_deepstream_config()


def save_deepstream_config(data: dict):
    _save_deepstream_config(data if isinstance(data, dict) else DEFAULT_DEEPSTREAM_CONFIG)


def load_infer_state() -> dict:
    data = read_json_file(INFER_STATE_PATH, {"stats": {}, "last_video": {}})
    if not isinstance(data, dict):
        return {"stats": {}, "last_video": {}}
    data.setdefault("stats", {})
    data.setdefault("last_video", {})
    return data


def save_infer_state(data: dict):
    write_json_file(INFER_STATE_PATH, data, indent=2)
