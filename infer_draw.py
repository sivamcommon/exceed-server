"""
Drawing and statistics for the leaf inspection pipeline.

This file handles two things:
  1. DRAWING: Rendering bounding boxes, masks, and defect annotations on frames.
  2. STATS: Tracking leaf objects across frames and computing quality summaries.

All class names, colors, and labels are loaded from class_config.json via
infer_detect.get_class_config() — nothing is hardcoded here.

Used by: infer_main.py calls these functions inside infer_frame_leaf_grouped_tracked().
"""

import logging
import math

import cv2
import numpy as np

from infer_detect import (
    LEAF_STEMS_OPENCV_MARKER,
    class_name,
    get_class_config,
    get_leaf_aliases,
    get_main_defect_aliases,
    get_stem_aliases,
    _build_alias_map,
    resolve_class,
)

# Set True to log per-leaf size measurements to the server log on every summary call.
SIZE_MEASURE_LOG = True


def _stem_length_px_cm_from_stats(stem_data, cm_per_px):
    """Decode leaf_stems tuple: model uses sqrt(area*ar); OpenCV uses mean px in first slot."""
    if stem_data is None:
        return 0.0, 0.0
    s_sum_area, s_sum_ar, s_count = stem_data
    if s_count <= 0:
        return 0.0, 0.0
    s_avg_area = s_sum_area / s_count
    s_avg_ar = s_sum_ar / s_count
    if abs(s_avg_ar - LEAF_STEMS_OPENCV_MARKER) < 1e-3:
        stem_len_px = s_avg_area
    else:
        stem_len_px = math.sqrt(s_avg_area * s_avg_ar)
    stem_len_cm = stem_len_px * float(cm_per_px)
    return stem_len_px, stem_len_cm


# ===========================================================================
# DRAWING FUNCTIONS
# ===========================================================================

def _get_main_class_visual(cname_l: str, main_section: dict, alias_map: dict):
    """Resolve a main-model class name to (color, mask_color) using class_config."""
    cls_key, entry = resolve_class(cname_l, main_section, alias_map)
    color = tuple(entry.get("color", (0, 0, 255)))
    mask_color = tuple(entry.get("mask_color", color))
    return color, mask_color


def draw_main_boxes(frame_bgr, xyxy, conf, cls_ids, names, keep_indices, colors_override, thickness):
    """
    Draw bounding boxes around detected leaves on the frame.

    Colors come from class_config.json (main_model section).
    colors_override: optional infer_config main.colors (per-class BGR); defaults come from class_config.

    Args:
        frame_bgr: Original frame (H, W, 3) BGR numpy array.
        xyxy: (N, 4) bounding boxes from the model.
        conf: (N,) confidence scores.
        cls_ids: (N,) integer class IDs.
        names: dict mapping class ID -> class name string.
        keep_indices: list of indices to draw (leaf-only, de-duplicated).
        colors_override: optional infer_config main.colors overrides.
        thickness: line thickness in pixels.

    Returns:
        New frame with boxes drawn on it.
    """
    if xyxy is None or len(keep_indices) == 0:
        return frame_bgr.copy()

    cc = get_class_config()
    main_section = cc.get("main_model", {})
    alias_map = _build_alias_map(main_section)

    img = frame_bgr.copy()
    for i in keep_indices:
        if i < 0 or i >= len(xyxy):
            continue
        x1, y1, x2, y2 = xyxy[i]
        cname = class_name(names, int(cls_ids[i]))
        cname_l = str(cname).strip().lower()

        # Resolve color: infer_config override > class_config > default
        cls_key, entry = resolve_class(cname_l, main_section, alias_map)
        box_color = tuple(entry.get("color", (0, 0, 255)))
        # Allow infer_config.json colors to override
        if cls_key in colors_override:
            box_color = tuple(colors_override[cls_key])

        label = entry.get("label", cname)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), box_color, thickness)
        cv2.putText(
            img,
            label,
            (int(x1), max(0, int(y1) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            box_color,
            1,
            cv2.LINE_AA,
        )
    return img


def draw_main_masks(frame_bgr, xyxy, cls_ids, names, keep_indices, colors_override, masks, alpha=0.35, masks_xy=None):
    """
    Draw semi-transparent segmentation masks over detected leaves.

    Colors come from class_config.json (main_model section).

    Args:
        frame_bgr: Original frame (H, W, 3) BGR numpy array.
        xyxy: (N, 4) bounding boxes.
        cls_ids: (N,) integer class IDs.
        names: dict mapping class ID -> class name string.
        keep_indices: list of leaf indices to draw.
        colors_override: optional infer_config main.colors overrides.
        masks: segmentation mask tensor from model (N, H, W).
        alpha: transparency for the mask overlay (0=invisible, 1=opaque).
        masks_xy: list of (K, 2) polygon contour arrays.

    Returns:
        New frame with masks blended on it.
    """
    if xyxy is None or len(keep_indices) == 0 or (masks is None and masks_xy is None):
        return frame_bgr.copy()

    cc = get_class_config()
    main_section = cc.get("main_model", {})
    alias_map = _build_alias_map(main_section)

    img = frame_bgr.copy()
    overlay = img.copy()
    h, w = img.shape[:2]

    for i in keep_indices:
        if i < 0 or i >= len(xyxy) or i >= len(masks):
            continue
        cname = class_name(names, int(cls_ids[i]))
        cname_l = str(cname).strip().lower()

        cls_key, entry = resolve_class(cname_l, main_section, alias_map)
        color = tuple(entry.get("mask_color", entry.get("color", (0, 0, 255))))
        if cls_key in colors_override:
            color = tuple(colors_override[cls_key])

        if masks_xy is not None and i < len(masks_xy) and masks_xy[i] is not None:
            poly = masks_xy[i].astype(np.int32)
            if poly.size == 0:
                continue
            cv2.fillPoly(overlay, [poly], color)
        else:
            mask = masks[i]
            if hasattr(mask, "cpu"):
                mask = mask.cpu().numpy()
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h))
            mask_bin = (mask > 0.5).astype(np.uint8)
            if mask_bin.max() == 0:
                continue
            overlay[mask_bin == 1] = color

    cv2.addWeighted(overlay, float(alpha), img, 1.0 - float(alpha), 0, img)
    return img


def draw_defects(annotated, defects, colors_override, thickness, draw_stem=True):
    """
    Draw defect bounding boxes with short labels on the annotated frame.

    Labels and colors come from class_config.json (defect_model section).

    Args:
        annotated: Frame to draw on (modified in-place).
        defects: list of defect dicts from run_defect_on_leaf, each with
                 "box" (x1, y1, x2, y2), "name" (label from class_config), "conf".
        colors_override: optional infer_config defect.colors overrides.
        thickness: line thickness in pixels.
        draw_stem: If False, skip drawing detections with role="ignore" (stems).

    Returns:
        True if any non-stem defect was found, False otherwise.
    """
    cc = get_class_config()
    defect_section = cc.get("defect_model", {})
    alias_map = _build_alias_map(defect_section)
    default_entry = defect_section.get("_default", {})

    defect_found = False
    for d in defects:
        gx1, gy1, gx2, gy2 = d["box"]
        label = d.get("label", d["name"])  # Short label e.g. "YL" for drawing on frame

        cls_key, entry = resolve_class(d["name"], defect_section, alias_map)
        box_color = tuple(entry.get("color", default_entry.get("color", (0, 0, 255))))

        # Allow infer_config.json colors to override
        if cls_key in colors_override:
            box_color = tuple(colors_override[cls_key])

        if entry.get("role") != "ignore":
            defect_found = True
        elif not draw_stem:
            continue  # Skip drawing stems when draw_stem is False

        # White (or near-white) boxes: use dark text for readability
        bc = box_color
        text_color = bc
        if len(bc) >= 3 and sum(int(bc[k]) for k in range(3)) >= 720:
            text_color = (0, 0, 0)

        cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), box_color, thickness)
        cv2.putText(
            annotated,
            label,
            (gx1, max(0, gy1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            text_color,
            1,
            cv2.LINE_AA,
        )
    return defect_found


# ===========================================================================
# TRACKING STATISTICS
# ===========================================================================

def update_tracking_stats(stats, xyxy, cls_ids, track_ids, names, keep_indices):
    """
    Update per-track statistics for the current frame's detections.

    Args:
        stats: Mutable dict that accumulates tracking state across frames.
        xyxy: (N, 4) bounding boxes.
        cls_ids: (N,) integer class IDs.
        track_ids: (N,) ByteTrack IDs, or None.
        names: dict mapping class ID -> class name string.
        keep_indices: list of leaf-only detection indices.
    """
    if stats is None or track_ids is None:
        return

    tracks = stats.setdefault("tracks", {})
    stats.setdefault("leaf_defects", {})
    track_sizes = stats.setdefault("track_sizes", {})
    frame_idx = stats.get("frame_idx", 0) + 1
    stats["frame_idx"] = frame_idx

    for i in keep_indices:
        if i < 0 or i >= len(cls_ids):
            continue
        tid = int(track_ids[i])
        cname = class_name(names, int(cls_ids[i]))
        tinfo = tracks.setdefault(tid, {"class_counts": {}, "last_seen": 0})
        tinfo["class_counts"][cname] = tinfo["class_counts"].get(cname, 0) + 1
        tinfo["last_seen"] = frame_idx
        x1, y1, x2, y2 = xyxy[i]
        w = max(1.0, (x2 - x1))
        h = max(1.0, (y2 - y1))
        # Max-area method: keep (length_px, width_px) from the frame with the
        # largest bbox so partial/entering frames don't pull the measurement down.
        length = max(w, h)
        width = min(w, h)
        area = length * width
        prev = track_sizes.get(tid)
        if prev is None:
            track_sizes[tid] = (length, width)
        else:
            prev_length, prev_width = prev
            if area > prev_length * prev_width:
                track_sizes[tid] = (length, width)

    # Prune tracks not seen for 20+ frames to keep tracking memory bounded.
    PRUNE_AFTER = 20
    if frame_idx % 10 == 0:  # Check frequently enough for low PRUNE_AFTER values.
        stale = [t for t, info in tracks.items() if frame_idx - info["last_seen"] > PRUNE_AFTER]
        for t in stale:
            del tracks[t]
            track_sizes.pop(t, None)
            stats.get("leaf_defect_counts", {}).pop(t, None)
            stats.get("leaf_defect_total", {}).pop(t, None)
            stats.get("leaf_defects", {}).pop(t, None)
            # Keep all per-track accumulators bounded to active tracks.
            stats.get("leaf_stems", {}).pop(t, None)
            stats.get("model2_track_sizes", {}).pop(t, None)


def compute_tracking_summary(stats, leaf_class: str = "leaf", defect_classes=None):
    """
    Aggregate per-track data into a summary of total leaves and defects.

    Args:
        stats: The tracking state dict accumulated by update_tracking_stats.
        leaf_class: The canonical leaf class name to count (default "leaf").
        defect_classes: Optional set of defect names to count.

    Returns:
        Dict with keys: total_leaf, total_defects, total_defected_leaf,
        defect_counts, tracks_seen.
    """
    tracks = stats.get("tracks", {})
    leaf_defects = stats.get("leaf_defects", {})

    # Majority vote: pick the most frequent class for each track
    final_class = {}
    for tid, info in tracks.items():
        counts = info.get("class_counts", {})
        if not counts:
            continue
        final = max(counts.items(), key=lambda kv: kv[1])[0]
        final_class[tid] = final

    # Count leaves using aliases from class_config.json
    # Both "good" leaves (role=leaf) and main-model defects (role=defect)
    # are physical leaves and count toward total_leaf
    leaf_aliases = get_leaf_aliases()
    main_defect_aliases = get_main_defect_aliases()
    all_leaf_aliases = leaf_aliases | main_defect_aliases
    total_leaf = sum(1 for c in final_class.values() if str(c).strip().lower() in all_leaf_aliases)

    # Main-model defect leaves are automatically defected
    main_defected_tids = set()
    for tid, cname in final_class.items():
        if str(cname).strip().lower() in main_defect_aliases:
            main_defected_tids.add(tid)

    if defect_classes is None:
        observed = set()
        for defect_set in leaf_defects.values():
            observed.update(defect_set)
        defect_classes = observed

    defect_counts = {d: 0 for d in defect_classes}
    for defect_set in leaf_defects.values():
        for d in defect_set:
            if d in defect_counts:
                defect_counts[d] += 1

    total_defects = sum(defect_counts.values())

    total_defected_leaf = 0
    for leaf_tid, defect_set in leaf_defects.items():
        fc = str(final_class.get(leaf_tid, "")).strip().lower()
        if fc in all_leaf_aliases and len(defect_set) > 0:
            total_defected_leaf += 1
    # Also count main-model defect leaves that may not be in leaf_defects
    for tid in main_defected_tids:
        if tid not in leaf_defects or len(leaf_defects.get(tid, set())) == 0:
            total_defected_leaf += 1

    return {
        "total_leaf": total_leaf,
        "total_defects": total_defects,
        "total_defected_leaf": total_defected_leaf,
        "defect_counts": defect_counts,
        "tracks_seen": len(final_class),
    }


def compute_size_summary(
    stats,
    size_cfg,
    cm_per_px,
    stem_size: bool = False,
    leaf_aliases=None,
    stem_aliases=None,
    max_full_leaf: int = 20,
):
    """
    Compute in-spec / out-spec size quality summary for tracked leaves.

    Args:
        stats: Tracking state dict with track_sizes, leaf_stems, etc.
        size_cfg: Dict with leaf/stem size thresholds.
        cm_per_px: Calibration scalar (centimeters per pixel).
        stem_size: If True, also check stem length for in-spec decision.
        leaf_aliases: Override leaf aliases (default: from class_config.json).
        stem_aliases: Override stem aliases (default: from class_config.json).
        max_full_leaf: Maximum number of leaves to include in size summary.

    Returns:
        Dict: {"leaf": {"count": N, "in_spec": X, "out_spec": Y,
                        "in_spec_pct": P, "out_spec_pct": Q}}
    """
    if stats is None:
        return {"leaf": {}}

    if leaf_aliases is None:
        leaf_aliases = get_leaf_aliases()
    if stem_aliases is None:
        stem_aliases = get_stem_aliases()

    # Include main-model defect aliases only when caller did not explicitly pass
    # leaf_aliases (backward-compatible behavior).
    main_defect_als = get_main_defect_aliases() if leaf_aliases is None else set()

    tracks = stats.get("tracks", {})
    track_sizes = stats.get("track_sizes", {})
    leaf_stems = stats.get("leaf_stems", {})

    # Majority vote for final class per track
    final_class = {}
    for tid, info in tracks.items():
        counts = info.get("class_counts", {})
        if not counts:
            continue
        final = max(counts.items(), key=lambda kv: kv[1])[0]
        final_class[tid] = final

    def init_bucket():
        return {"in_spec": 0, "out_spec": 0, "count": 0, "in_spec_pct": 0, "out_spec_pct": 0}

    summary = {"leaf": init_bucket()}

    leaf_set = {str(v).strip().lower() for v in (leaf_aliases | main_defect_als)}

    def get_range(cfg, key):
        rng = cfg.get(key, {})
        return float(rng.get("min", 0.0)), float(rng.get("max", 0.0))

    # Build ordered list of "complete" leaves
    full_leaf_ids = stats.setdefault("size_full_leaf_ids", [])
    if not isinstance(full_leaf_ids, list):
        full_leaf_ids = []
        stats["size_full_leaf_ids"] = full_leaf_ids

    for tid, cls_nm in final_class.items():
        cname = str(cls_nm).strip().lower()
        if cname not in leaf_set:
            continue
        if tid not in track_sizes:
            continue
        # Include defected leaves (torn/cut/stage-2 defects): size cohort is first
        # N tracks with stable boxes, not only "clean" leaves. Stem thresholds apply
        # only when stem measurements exist (see below).
        if tid not in full_leaf_ids and len(full_leaf_ids) < int(max_full_leaf):
            full_leaf_ids.append(tid)

    # Get thresholds
    h_min, h_max = get_range(size_cfg.get("leaf", {}), "length")
    w_min, w_max = get_range(size_cfg.get("leaf", {}), "width")
    stem_min, stem_max = get_range(size_cfg.get("stem", {}), "length")

    # Check each leaf against thresholds
    for tid in full_leaf_ids[:int(max_full_leaf)]:
        cls_nm = str(final_class.get(tid, "")).strip().lower()
        if cls_nm not in leaf_set:
            continue
        if tid not in track_sizes:
            continue
        # Max-area method: use the best (largest bbox) frame directly.
        length_px, width_px = track_sizes[tid]
        length_cm = length_px * float(cm_per_px)
        width_cm = width_px * float(cm_per_px)

        stem_len_cm = 0.0
        stem_data = leaf_stems.get(tid)
        if stem_data is not None:
            _, stem_len_cm = _stem_length_px_cm_from_stats(stem_data, cm_per_px)

        summary["leaf"]["count"] += 1
        leaf_ok = (h_min <= length_cm <= h_max) and (w_min <= width_cm <= w_max)
        stem_ok = True
        if stem_size and stem_data is not None:
            stem_ok = stem_min <= stem_len_cm <= stem_max
        if leaf_ok and stem_ok:
            summary["leaf"]["in_spec"] += 1
        else:
            summary["leaf"]["out_spec"] += 1

        if SIZE_MEASURE_LOG:
            status = "IN_SPEC" if (leaf_ok and stem_ok) else "OUT_SPEC"
            logging.info(
                "[SIZE] track=%s class=%s  length=%.2fcm(%.1fpx)  width=%.2fcm(%.1fpx)"
                "  thresholds=L[%.1f-%.1f] W[%.1f-%.1f]  cm_per_px=%.5f  %s",
                tid, cls_nm,
                length_cm, length_px,
                width_cm, width_px,
                h_min, h_max, w_min, w_max,
                float(cm_per_px),
                status,
            )

    # Compute percentages
    def add_percentages(group):
        total = group["count"]
        if total > 0:
            in_pct = round(100.0 * group["in_spec"] / total)
            out_pct = round(100.0 * group["out_spec"] / total)
        else:
            in_pct = 0
            out_pct = 0
        group["in_spec_pct"] = in_pct
        group["out_spec_pct"] = out_pct

    add_percentages(summary["leaf"])
    return summary


def draw_size_debug(
    annotated,
    xyxy,
    track_ids,
    stats,
    size_cfg,
    cm_per_px,
    stem_size,
    color=(255, 150, 0),
):
    """
    Draw size debug overlay on leaves that are counted in the size summary.

    For each leaf in size_full_leaf_ids that is visible in this frame,
    draws a colored bounding box and annotates blade length, width,
    and stem length (in cm) so the user can verify what is being measured.

    Args:
        annotated: Frame to draw on (modified in-place).
        xyxy: (N, 4) bounding boxes from the model.
        track_ids: (N,) tracker IDs corresponding to xyxy.
        stats: Tracking state dict with track_sizes, leaf_stems, size_full_leaf_ids.
        size_cfg: Dict with leaf/stem size thresholds (for min/max display).
        cm_per_px: Calibration scalar.
        stem_size: Whether stem is included in size check.
        color: Unused; kept for API compatibility (leaf outline is not recolored).
    """
    if stats is None or xyxy is None or track_ids is None:
        return

    full_leaf_ids = stats.get("size_full_leaf_ids", [])
    if not full_leaf_ids:
        return

    track_sizes = stats.get("track_sizes", {})
    leaf_stems = stats.get("leaf_stems", {})
    full_set = set(full_leaf_ids)
    _ = color

    # Build a map: tid -> latest bbox in this frame (last occurrence wins)
    tid_to_box = {}
    for idx in range(len(xyxy)):
        if idx >= len(track_ids):
            break
        tid = int(track_ids[idx])
        if tid in full_set:
            tid_to_box[tid] = xyxy[idx]

    def get_range(cfg, key):
        rng = cfg.get(key, {})
        return float(rng.get("min", 0.0)), float(rng.get("max", 0.0))

    h_min, h_max = get_range(size_cfg.get("leaf", {}), "length")
    w_min, w_max = get_range(size_cfg.get("leaf", {}), "width")
    stem_min, stem_max = get_range(size_cfg.get("stem", {}), "length")

    for tid in full_leaf_ids:
        if tid not in tid_to_box:
            continue
        if tid not in track_sizes:
            continue

        x1, y1, x2, y2 = tid_to_box[tid]
        ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)

        length_px, width_px = track_sizes[tid]
        length_cm = length_px * float(cm_per_px)
        width_cm = width_px * float(cm_per_px)

        stem_len_cm = 0.0
        stem_data = leaf_stems.get(tid)
        if stem_data is not None:
            _, stem_len_cm = _stem_length_px_cm_from_stats(stem_data, cm_per_px)

        # Determine in-spec
        leaf_ok = (h_min <= length_cm <= h_max) and (w_min <= width_cm <= w_max)
        stem_ok = True
        if stem_size and stem_data is not None:
            stem_ok = stem_min <= stem_len_cm <= stem_max

        # Text only — do not redraw leaf border (keeps infer red/green verdict visible)

        # Build text lines
        spec_label = "OK" if (leaf_ok and stem_ok) else "OUT"
        lines = [
            f"L:{length_cm:.1f}cm W:{width_cm:.1f}cm [{spec_label}]",
        ]
        if stem_size:
            lines.append(f"S:{stem_len_cm:.1f}cm")

        # Draw text below the box
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        ty = iy2 + 15
        text_bgr = (255, 255, 255)
        for line in lines:
            (tw, th), _ = cv2.getTextSize(line, font, font_scale, thickness)
            cv2.rectangle(annotated, (ix1, ty - th - 2), (ix1 + tw + 4, ty + 3), (0, 0, 0), -1)
            cv2.putText(annotated, line, (ix1 + 2, ty), font, font_scale, text_bgr, thickness, cv2.LINE_AA)
            ty += th + 6
