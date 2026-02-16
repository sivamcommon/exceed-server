"""Model loading utilities."""

from ultralytics import YOLO

ENGINE_PATH = "models/best.engine"
DEFECT_ENGINE_PATH = "models/defect.engine"


def load_model(engine_path: str = ENGINE_PATH):
    """Load a single YOLO model from a TensorRT engine file."""
    return YOLO(engine_path)


def load_models(main_path: str = ENGINE_PATH, defect_path: str = DEFECT_ENGINE_PATH):
    """Load both the main and defect YOLO models."""
    return YOLO(main_path), YOLO(defect_path)
