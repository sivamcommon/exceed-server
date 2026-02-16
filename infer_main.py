"""
Inference entrypoint.

This file exposes a minimal public API while the implementation lives in
`infer/` modules for clarity and easier maintenance.
"""

from infer.models import load_model, load_models, ENGINE_PATH, DEFECT_ENGINE_PATH
from infer.core import (
    LEAF_CLASS_ALIASES,
    STEM_CLASS_ALIASES,
    extract_main_detections,
    select_leaf_indices,
    run_defect_on_leaf,
    class_name,
)
from infer.draw import draw_main_boxes, draw_defects
from infer.stats import update_tracking_stats, compute_tracking_summary, compute_size_summary

# Default calibration and size thresholds (kept here for external imports)
MM_PER_PX = 0.1  # default placeholder; replace with your calibration
SIZE_THRESHOLDS = {
    "leaf": {
        "height": {"min": 40.0, "max": 70.0},
        "width": {"min": 25.0, "max": 50.0},
    },
    "stem": {
        "height": {"min": 5.0, "max": 30.0},
        "width": {"min": 2.0, "max": 10.0},
    },
}


def infer_frame_leaf_grouped_tracked(
    main_model,
    defect_model,
    frame_bgr,
    stats,
    imgsz: int = 640,
    conf: float = 0.45,
    device: int = 0,
    size_cfg=SIZE_THRESHOLDS,
    mm_per_px: float = MM_PER_PX,
):
    """
    Run two-stage inference on a single frame.

    Input
    - main_model: YOLO model for leaf detection (tracks objects).
    - defect_model: YOLO model for defect detection (runs on leaf crops).
    - frame_bgr: numpy array (H, W, 3) in BGR format.
    - stats: dict for tracking state (mutated in-place). Use {} initially.
    - imgsz: inference size for both models.
    - conf: confidence threshold for both models.
    - device: GPU device index (0 by default).
    - size_cfg: dict of size thresholds used for size summary.
    - mm_per_px: calibration scalar to convert pixels to mm.

    Output
    - annotated: BGR frame with boxes for leaves and defects.
    - summary: dict with counts and size summary.

    Use case
    - Call this per frame in the streaming loop to get a visual overlay and
      aggregate statistics (leaf count, defect count, size spec summary).
    """
    results = main_model.track(
        source=frame_bgr,
        persist=True,
        verbose=False,
        imgsz=imgsz,
        conf=conf,
        iou=0.8,
        device=device,
    )
    result = results[0]

    xyxy, confs, cls_ids, track_ids, names = extract_main_detections(result)
    keep_indices = [] if cls_ids is None else select_leaf_indices(cls_ids, names)

    annotated = draw_main_boxes(frame_bgr, xyxy, confs, cls_ids, names, keep_indices)

    if keep_indices and xyxy is not None:
        for i in keep_indices:
            x1, y1, x2, y2 = xyxy[i]
            defects = run_defect_on_leaf(
                defect_model,
                frame_bgr,
                (x1, y1, x2, y2),
                imgsz,
                conf,
                device,
            )
            defect_found = draw_defects(annotated, defects)

            if defect_found:
                leaf_label = f"{class_name(names, int(cls_ids[i]))} {confs[i]:.2f}"
                import cv2
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(
                    annotated,
                    leaf_label,
                    (int(x1), max(0, int(y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

            if stats is not None and track_ids is not None:
                leaf_tid = int(track_ids[i])
                for d in defects:
                    stats.setdefault("leaf_defects", {}).setdefault(leaf_tid, set()).add(d["name"])

    if stats is not None and xyxy is not None and cls_ids is not None:
        update_tracking_stats(stats, xyxy, cls_ids, track_ids, names, keep_indices)

    summary = compute_tracking_summary(stats) if stats is not None else {}
    if stats is not None:
        summary["size_summary"] = compute_size_summary(stats, size_cfg=size_cfg, mm_per_px=mm_per_px)

    return annotated, summary


__all__ = [
    "ENGINE_PATH",
    "DEFECT_ENGINE_PATH",
    "MM_PER_PX",
    "SIZE_THRESHOLDS",
    "LEAF_CLASS_ALIASES",
    "STEM_CLASS_ALIASES",
    "load_model",
    "load_models",
    "infer_frame_leaf_grouped_tracked",
    "compute_tracking_summary",
    "compute_size_summary",
]
