"""Inference configuration loader."""

import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "infer_config.json")

DEFAULT_CONFIG = {
    "models": {
        "main": "models/best.engine",
        "defect": "models/defect.engine",
    },
    "main": {
        "conf_leaf": 0.7,
        "box_thickness": 4,
        "colors": {
            "leaf": [0, 255, 0]
        },
    },
    "defect": {
        "conf_stem": 0.45,
        "conf_other": 0.30,
        "box_thickness": 2,
        "min_area_px": {
            "yellow": 0,
            "white": 0,
            "ipd": 0,
            "stem": 0,
            "other": 0
        },
        "vote_thresholds": {
            "min_count": 5
        },
        "colors": {
            "stem": [144, 238, 144],
            "other": [0, 0, 255],
            "yellow": [0, 255, 255],
            "white": [255, 255, 255],
            "ipd": [0, 0, 255],
        },
    },
}


def load_config():
    if not os.path.exists(CONFIG_PATH):
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()
    try:
        with open(CONFIG_PATH, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return DEFAULT_CONFIG.copy()
        merged = DEFAULT_CONFIG.copy()
        merged.update(data)
        return merged
    except Exception:
        return DEFAULT_CONFIG.copy()


def save_config(config):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
