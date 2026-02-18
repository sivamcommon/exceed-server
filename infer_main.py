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
from infer.config import load_config

# Default calibration and size thresholds (kept here for external imports)
MM_PER_PX = 0.1  # default placeholder; replace with your calibration
SIZE_THRESHOLDS = {
    "leaf": {
        "height": {"min": 40.0, "max": 70.0},
        "width": {"min": 25.0, "max": 50.0},
    },
    "stem": {
        "height": {"min": 5.0, "max": 30.0},
    },
}


def infer_frame_leaf_grouped_tracked(
    main_model,
    defect_model,
    frame_bgr,
    stats,
    imgsz: int | None = None,
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
    cfg = load_config()
    main_cfg = cfg.get("main", {})
    defect_cfg = cfg.get("defect", {})
    main_colors = main_cfg.get("colors", {})
    defect_colors = defect_cfg.get("colors", {})
    main_thickness = int(main_cfg.get("box_thickness", 2))
    defect_thickness = int(defect_cfg.get("box_thickness", 2))
    conf_leaf = float(main_cfg.get("conf_leaf", conf))
    conf_stem = float(defect_cfg.get("conf_stem", conf))
    conf_other = float(defect_cfg.get("conf_other", conf))
    min_area_px = defect_cfg.get("min_area_px", {})
    vote_cfg = defect_cfg.get("vote_thresholds", {})
    vote_min_count = int(vote_cfg.get("min_count", 5))

    track_kwargs = {
        "source": frame_bgr,
        "persist": True,
        "verbose": False,
        "conf": conf_leaf,
        "iou": 0.8,
        "tracker": "trackers/bytetrack.yaml",
        "device": device,
    }
    if imgsz is not None:
        track_kwargs["imgsz"] = imgsz
    results = main_model.track(**track_kwargs)
    result = results[0]

    xyxy, confs, cls_ids, track_ids, names = extract_main_detections(result)
    keep_indices = [] if cls_ids is None else select_leaf_indices(cls_ids, names)

    # De-duplicate overlapping leaf boxes (keeps highest confidence)
    if keep_indices and confs is not None and xyxy is not None:
        def iou(a, b):
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            inter_x1 = max(ax1, bx1)
            inter_y1 = max(ay1, by1)
            inter_x2 = min(ax2, bx2)
            inter_y2 = min(ay2, by2)
            inter_w = max(0.0, inter_x2 - inter_x1)
            inter_h = max(0.0, inter_y2 - inter_y1)
            inter = inter_w * inter_h
            area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
            area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
            return inter / (area_a + area_b - inter)

        iou_thresh = 0.5
        sorted_idx = sorted(keep_indices, key=lambda i: confs[i], reverse=True)
        dedup = []
        for idx in sorted_idx:
            box = xyxy[idx]
            if all(iou(box, xyxy[j]) < iou_thresh for j in dedup):
                dedup.append(idx)
        # Remove leaf boxes inside a larger leaf (keep only the bigger one)
        final = []
        for idx in dedup:
            x1, y1, x2, y2 = xyxy[idx]
            area_i = max(1.0, (x2 - x1) * (y2 - y1))
            contained = False
            for j in dedup:
                if j == idx:
                    continue
                X1, Y1, X2, Y2 = xyxy[j]
                area_j = max(1.0, (X2 - X1) * (Y2 - Y1))
                if area_j <= area_i:
                    continue
                # Full containment
                if x1 >= X1 and y1 >= Y1 and x2 <= X2 and y2 <= Y2:
                    contained = True
                    break
                # High overlap of smaller inside larger (IoA)
                inter_x1 = max(x1, X1)
                inter_y1 = max(y1, Y1)
                inter_x2 = min(x2, X2)
                inter_y2 = min(y2, Y2)
                inter_w = max(0.0, inter_x2 - inter_x1)
                inter_h = max(0.0, inter_y2 - inter_y1)
                inter_area = inter_w * inter_h
                ioa = inter_area / area_i
                if ioa >= 0.6:
                    contained = True
                    break
            if not contained:
                final.append(idx)
        keep_indices = final

    annotated = draw_main_boxes(frame_bgr, xyxy, confs, cls_ids, names, keep_indices, main_colors, main_thickness)

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
                conf_stem=conf_stem,
                conf_other=conf_other,
                min_area_px=min_area_px,
            )
            # Keep only the largest stem per leaf (by box area)
            stem_defs = []
            other_defs = []
            for d in defects:
                cname_l = str(d["name"]).strip().lower()
                if cname_l in ("stem", "leaf stem"):
                    stem_defs.append(d)
                else:
                    other_defs.append(d)
            if stem_defs:
                def _area(defect):
                    x1d, y1d, x2d, y2d = defect["box"]
                    return max(1, (x2d - x1d) * (y2d - y1d))
                stem_defs = [max(stem_defs, key=_area)]
            defects = other_defs + stem_defs

            # Update defect vote counts before deciding to draw
            if stats is not None and track_ids is not None:
                leaf_tid = int(track_ids[i])
                stats.setdefault("leaf_defect_total", {}).setdefault(leaf_tid, 0)
                stats["leaf_defect_total"][leaf_tid] += 1

                for d in defects:
                    cname_l = str(d["name"]).strip().lower()
                    if cname_l in ("stem", "leaf stem"):
                        x1d, y1d, x2d, y2d = d["box"]
                        w = max(1.0, (x2d - x1d))
                        h = max(1.0, (y2d - y1d))
                        height_mm = max(w, h) * float(mm_per_px)
                        stems = stats.setdefault("leaf_stems", {})
                        prev = stems.get(leaf_tid)
                        stems[leaf_tid] = height_mm if prev is None else max(prev, height_mm)
                    else:
                        stats.setdefault("leaf_defect_counts", {}).setdefault(leaf_tid, {})
                        curr = stats["leaf_defect_counts"][leaf_tid].get(d["name"], 0)
                        stats["leaf_defect_counts"][leaf_tid][d["name"]] = curr + 1

            # Filter defects for drawing using vote min_count
            filtered = []
            if stats is not None and track_ids is not None:
                leaf_tid = int(track_ids[i])
                counts = stats.get("leaf_defect_counts", {}).get(leaf_tid, {})
                for d in defects:
                    cname_l = str(d["name"]).strip().lower()
                    if cname_l in ("stem", "leaf stem"):
                        filtered.append(d)
                        continue
                    cnt = counts.get(d["name"], 0)
                    if cnt >= vote_min_count:
                        filtered.append(d)
            else:
                # If we don't have tracking IDs yet, do not draw defect boxes
                filtered = [d for d in defects if str(d["name"]).strip().lower() in ("stem", "leaf stem")]

            defect_found = any(
                str(d["name"]).strip().lower() not in ("stem", "leaf stem") for d in filtered
            )
            draw_defects(annotated, filtered, defect_colors, defect_thickness)

            if defect_found:
                import cv2
                cv2.rectangle(
                    annotated,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 0, 255),
                    main_thickness,
                )
            else:
                # Mark healthy leaf with green 'G' at center
                cx = int((x1 + x2) * 0.5)
                cy = int((y1 + y2) * 0.5)
                import cv2
                cv2.putText(
                    annotated,
                    "G",
                    (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            # stats updates already handled above

    if stats is not None and xyxy is not None and cls_ids is not None:
        update_tracking_stats(stats, xyxy, cls_ids, track_ids, names, keep_indices)

    summary = compute_tracking_summary(stats) if stats is not None else {}
    summary["frame_leaf_count"] = len(keep_indices) if keep_indices is not None else 0
    if stats is not None:
        summary["size_summary"] = compute_size_summary(stats, size_cfg=size_cfg, mm_per_px=mm_per_px)

        # Apply per-leaf defect voting (ratio OR min_count)
        defect_counts = {}
        total_defects = 0
        total_defected_leaf = 0
        leaf_defect_counts = stats.get("leaf_defect_counts", {})
        leaf_defect_total = stats.get("leaf_defect_total", {})
        for leaf_tid, counts in leaf_defect_counts.items():
            total = max(1, int(leaf_defect_total.get(leaf_tid, 0)))
            kept = []
            for name, cnt in counts.items():
                if cnt >= vote_min_count:
                    kept.append(name)
                    defect_counts[name] = defect_counts.get(name, 0) + 1
            if kept:
                total_defected_leaf += 1
                total_defects += len(kept)
        summary["defect_counts"] = defect_counts
        summary["total_defects"] = total_defects
        summary["total_defected_leaf"] = total_defected_leaf

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
