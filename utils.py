import json
import os
from typing import Any


def read_json_file(path: str, default: Any):
    """Read JSON file and return default on any error."""
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data
    except Exception:
        return default


def write_json_file(path: str, payload: Any, indent: int = 2):
    """Write JSON file, creating parent directory if needed."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=indent)


def deep_merge_dict(base: dict, override: dict) -> dict:
    """Recursively merge override into base and return merged dict."""
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge_dict(out[key], value)
        else:
            out[key] = value
    return out
