"""Model loading utilities."""

from ultralytics import YOLO
from .config import load_config

_CFG = load_config()
ENGINE_PATH = _CFG.get("models", {}).get("main", "models/best.engine")
DEFECT_ENGINE_PATH = _CFG.get("models", {}).get("defect", "models/defect.engine")


def load_model(engine_path: str = ENGINE_PATH):
    """Load a single YOLO model from a TensorRT engine file."""
    return YOLO(engine_path)


def load_models(main_path: str = ENGINE_PATH, defect_path: str = DEFECT_ENGINE_PATH):
    """Load both the main and defect YOLO models."""
    return YOLO(main_path), YOLO(defect_path)
