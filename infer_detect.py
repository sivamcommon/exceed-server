"""
Detection logic for the two-stage leaf inspection pipeline.

This file handles all detection-related operations:
  - Extracting raw detections from YOLO model output
  - Filtering to keep only leaf-class objects
  - De-duplicating overlapping leaf boxes (IoU + containment)
  - Converting segmentation masks/polygons to tight bounding boxes
  - Running the defect model on individual leaf crops
  - Optional inner-ROI overlap filter (infer_config main.roi_*) to ignore edge detections

Class names, aliases, colors, and confidence thresholds are loaded from
class_config.json — nothing is hardcoded here.

Used by: infer_main.py calls these functions inside infer_frame_leaf_grouped_tracked().
"""

import json
import logging
import os
from typing import List, Tuple

import cv2
import numpy as np
import torch

import config_store

# Optional monitor hook — set via set_monitor() so detection debug prints
# also appear in monitor.log.
_monitor_log_fn = None


def set_monitor(fn):
    """Pass SmartMonitor.log (or any callable) to route detect logs to monitor.log."""
    global _monitor_log_fn
    _monitor_log_fn = fn


def _detect_print(msg: str):
    # Keep stdout quiet during runtime; forward debug lines to monitor instead.
    if _monitor_log_fn is not None:
        try:
            _monitor_log_fn(msg)
        except Exception:
            pass


# When leaf_stems[tid] averages to this "aspect" value, stem length (px) is
# stored directly in the first slot (mean stem euclidean length in crop pixels).
# See infer_main OpenCV stem path and infer_draw.compute_size_summary.
LEAF_STEMS_OPENCV_MARKER = -1.0

# ---------------------------------------------------------------------------
# Class config loading
# ---------------------------------------------------------------------------

_class_cfg_cache = None
_class_cfg_mtime: float = 0.0


def get_class_config():
    """Load class_config.json with mtime-based caching (same pattern as infer_config)."""
    global _class_cfg_cache, _class_cfg_mtime
    mtime = config_store.class_config_mtime()
    if _class_cfg_cache is None or mtime != _class_cfg_mtime:
        data = config_store.load_class_config()
        if not isinstance(data, dict):
            data = {}
        _class_cfg_cache = data if data else {"main_model": {}, "defect_model": {}}
        _class_cfg_mtime = mtime
    return _class_cfg_cache


def _build_alias_map(model_section: dict) -> dict:
    """Build a flat map: raw_alias_name -> class_key for fast lookup."""
    alias_map = {}
    for key, entry in model_section.items():
        if key.startswith("_"):
            continue
        for alias in entry.get("aliases", []):
            alias_map[alias.strip().lower()] = key
    return alias_map


def resolve_class(raw_name: str, model_section: dict, alias_map: dict) -> tuple:
    """Resolve a raw model output name to (class_key, class_entry).
    Checks: 1) direct config key match, 2) alias map, 3) _default fallback."""
    name_l = str(raw_name).strip().lower()
    # Direct config key match (e.g. "white spots" is already the key)
    if name_l in model_section:
        return name_l, model_section[name_l]
    # Alias lookup (e.g. "white area" → "white spots")
    key = alias_map.get(name_l)
    if key is not None and key in model_section:
        return key, model_section[key]
    default = model_section.get("_default", {})
    return name_l, default


def _resolve_raw_class_name(raw_name: str, model_section: dict) -> str:
    """
    Normalize raw model class names before alias/config resolution.

    Some backends (notably RF-DETR wrappers) return numeric class names ("0", "1", ...).
    When that happens, map the numeric index to the configured class key by order so
    downstream alias mapping can still resolve to semantic class names.
    """
    name = str(raw_name).strip().lower()
    if name.isdigit():
        idx = int(name)
        ordered_keys = [k for k in model_section.keys() if not str(k).startswith("_")]
        if 0 <= idx < len(ordered_keys):
            return str(ordered_keys[idx]).strip().lower()
    return name


def get_leaf_aliases() -> set:
    """Return aliases treated as leaf classes in the main model.

    Behavior:
    - If any class has role="leaf", use only those aliases (legacy behavior).
    - If no role fields exist, treat all configured classes as object classes.
    """
    cc = get_class_config()
    main = cc.get("main_model", {})
    has_leaf_role = any(
        isinstance(entry, dict) and entry.get("role") == "leaf"
        for key, entry in main.items()
        if not str(key).startswith("_")
    )
    aliases = set()
    if not has_leaf_role:
        for key, entry in main.items():
            if str(key).startswith("_"):
                continue
            aliases.add(str(key).strip().lower())
            for a in entry.get("aliases", []):
                aliases.add(str(a).strip().lower())
        return aliases

    for key, entry in main.items():
        if key.startswith("_"):
            continue
        if has_leaf_role and entry.get("role") != "leaf":
            continue
        aliases.add(str(key).strip().lower())
        for a in entry.get("aliases", []):
            aliases.add(a.strip().lower())
    return aliases


def get_main_defect_aliases() -> set:
    """Return aliases for classes explicitly marked role='defect' in main model.

    If no classes are marked role='defect', returns empty set.
    """
    cc = get_class_config()
    main = cc.get("main_model", {})
    aliases = set()
    for key, entry in main.items():
        if key.startswith("_"):
            continue
        if entry.get("role") != "defect":
            continue
        aliases.add(str(key).strip().lower())
        for a in entry.get("aliases", []):
            aliases.add(str(a).strip().lower())
    return aliases


def select_main_defect_indices(cls_ids, names) -> List[int]:
    """
    Return indices of detections that are defect-class objects from the main model.

    Uses aliases from class_config.json (role="defect" in main_model section).
    These are items like yellow, torn, cut that the main model identifies directly
    as defects without needing to go through the second defect model.
    """
    defect_aliases = get_main_defect_aliases()
    keep = []
    for i, c in enumerate(cls_ids):
        cname = str(class_name(names, int(c))).lower()
        if cname in defect_aliases:
            keep.append(i)
    return keep


def get_stem_aliases() -> set:
    """Return defect-model aliases for classes with role 'ignore' (e.g. stem), for size/stem logic."""
    cc = get_class_config()
    defect = cc.get("defect_model", {})
    aliases = set()
    for key, entry in defect.items():
        if key.startswith("_"):
            continue
        if entry.get("role") != "ignore":
            continue
        for a in entry.get("aliases", []):
            aliases.add(a.strip().lower())
    return aliases


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def class_name(names, cls_id: int) -> str:
    """
    Look up a human-readable class name from the model's names dict/list.

    Why: YOLO models store class names as either a dict {0: "leaf", 1: "stem"}
    or a list ["leaf", "stem"]. This function handles both formats so the rest
    of the code doesn't have to care.

    Args:
        names: dict or list of class names from the YOLO result object.
        cls_id: integer class index to look up.

    Returns:
        The class name string, or the string version of cls_id if not found.
    """
    if isinstance(names, dict):
        return names.get(cls_id, str(cls_id))
    if isinstance(names, list) and 0 <= cls_id < len(names):
        return names[cls_id]
    return str(cls_id)


def _get_min_hw(entry: dict, default_entry: dict) -> tuple[float, float]:
    """Read per-class minimum width/height thresholds (pixels)."""
    try:
        min_w = float(entry.get("min_w", default_entry.get("min_w", 0)))
    except Exception:
        min_w = float(default_entry.get("min_w", 0) or 0)
    try:
        min_h = float(entry.get("min_h", default_entry.get("min_h", 0)))
    except Exception:
        min_h = float(default_entry.get("min_h", 0) or 0)
    if min_w < 0:
        min_w = 0.0
    if min_h < 0:
        min_h = 0.0
    return min_w, min_h


def filter_indices_by_min_hw(indices, xyxy, cls_ids, names, model_section: dict) -> List[int]:
    """
    Filter detection indices using per-class min_w/min_h (pixels) from class_config.json.

    Any detection whose bbox width or height is smaller than the configured threshold
    for that class is dropped.
    """
    if not indices or xyxy is None or cls_ids is None:
        return indices
    alias_map = _build_alias_map(model_section)
    default_entry = model_section.get("_default", {})
    kept = []
    for i in indices:
        if i < 0 or i >= len(xyxy) or i >= len(cls_ids):
            continue
        x1, y1, x2, y2 = xyxy[i]
        w = float(x2) - float(x1)
        h = float(y2) - float(y1)
        raw_cname = str(class_name(names, int(cls_ids[i]))).strip().lower()
        _, entry = resolve_class(raw_cname, model_section, alias_map)
        min_w, min_h = _get_min_hw(entry, default_entry)
        if w < min_w or h < min_h:
            continue
        kept.append(i)
    return kept


def roi_bounds_from_frame(frame_shape, margin_pct: float) -> Tuple[float, float, float, float]:
    """
    Inner axis-aligned ROI inset equally from all four edges.

    Horizontal inset uses margin_pct of frame width (left and right); vertical uses
    margin_pct of frame height (top and bottom).

    Returns:
        (x1, y1, x2, y2) pixel coordinates of the inner rectangle.
    """
    if not frame_shape or len(frame_shape) < 2:
        return 0.0, 0.0, 0.0, 0.0
    h, w = float(frame_shape[0]), float(frame_shape[1])
    try:
        m = float(margin_pct)
    except (TypeError, ValueError):
        m = 0.0
    if m < 0.0:
        m = 0.0
    if m > 49.0:
        m = 49.0
    mx = w * (m / 100.0)
    my = h * (m / 100.0)
    x1, y1 = mx, my
    x2, y2 = w - mx, h - my
    if x2 <= x1 or y2 <= y1:
        return 0.0, 0.0, w, h
    return x1, y1, x2, y2


def bbox_roi_overlap_fraction(box, roi_xyxy: Tuple[float, float, float, float]) -> float:
    """Fraction of bbox area that lies inside the ROI rectangle [0, 1]."""
    x1, y1, x2, y2 = map(float, box)
    rx1, ry1, rx2, ry2 = map(float, roi_xyxy)
    iw = max(0.0, min(x2, rx2) - max(x1, rx1))
    ih = max(0.0, min(y2, ry2) - max(y1, ry1))
    inter = iw * ih
    area = max(0.0, (x2 - x1) * (y2 - y1))
    if area <= 0.0:
        return 0.0
    return inter / area


def bbox_center_inside_roi(box, roi_xyxy: Tuple[float, float, float, float]) -> bool:
    """True if bbox center (cx, cy) lies inside the ROI rectangle (inclusive)."""
    x1, y1, x2, y2 = map(float, box)
    rx1, ry1, rx2, ry2 = map(float, roi_xyxy)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    return rx1 <= cx <= rx2 and ry1 <= cy <= ry2


def filter_indices_by_roi_overlap(
    indices: List[int],
    xyxy,
    roi_xyxy: Tuple[float, float, float, float],
    min_overlap: float,
    mode: str = "overlap",
) -> List[int]:
    """
    Keep indices depending on roi_filter_mode:

    - overlap: bbox area fraction inside ROI >= min_overlap (default; edge boxes
      that are mostly inside the inner rect can still pass).
    - center: bbox center must lie inside ROI (stricter for conveyor edges).
    - both: overlap rule AND center rule.
    """
    if not indices or xyxy is None:
        return indices
    m = str(mode or "overlap").strip().lower()
    if m not in ("overlap", "center", "both"):
        m = "overlap"
    try:
        mo = float(min_overlap)
    except (TypeError, ValueError):
        mo = 0.5
    if mo < 0.0:
        mo = 0.0
    if mo > 1.0:
        mo = 1.0
    kept = []
    for i in indices:
        if i < 0 or i >= len(xyxy):
            continue
        box = xyxy[i]
        ok_overlap = bbox_roi_overlap_fraction(box, roi_xyxy) >= mo
        ok_center = bbox_center_inside_roi(box, roi_xyxy)
        if m == "center":
            if not ok_center:
                continue
        elif m == "both":
            if not (ok_overlap and ok_center):
                continue
        else:
            if not ok_overlap:
                continue
        kept.append(i)
    return kept


# ---------------------------------------------------------------------------
# Stage 1: Extract and filter main model detections
# ---------------------------------------------------------------------------

def extract_main_detections(result):
    """
    Parse a single YOLO result object into numpy arrays.

    Why: The YOLO result object has nested attributes (result.boxes.xyxy, etc.)
    stored as PyTorch tensors on GPU. This function moves everything to CPU numpy
    arrays in one place so downstream code works with plain numpy.

    Args:
        result: A single YOLO result object (from model.track()[0]).

    Returns:
        Tuple of (xyxy, conf, cls_ids, track_ids, names, masks, masks_xy):
        - xyxy: (N, 4) array of bounding boxes in [x1, y1, x2, y2] format.
        - conf: (N,) array of confidence scores.
        - cls_ids: (N,) array of integer class IDs.
        - track_ids: (N,) array of ByteTrack tracking IDs, or None if not tracked.
        - names: dict mapping class ID -> class name string.
        - masks: tensor of binary segmentation masks (if seg model), or None.
        - masks_xy: list of polygon contours per detection (if seg model), or None.
        All arrays are None if no detections were found.
    """
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None, None, None, None, result.names if hasattr(result, "names") else {}, None, None
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    track_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None
    names = result.names if hasattr(result, "names") else {}
    masks = None
    masks_xy = None
    if hasattr(result, "masks") and result.masks is not None:
        masks = result.masks.data
        masks_xy = getattr(result.masks, "xy", None)
    return xyxy, conf, cls_ids, track_ids, names, masks, masks_xy


def select_leaf_indices(cls_ids, names) -> List[int]:
    """
    Return indices of detections that are leaf-class objects.

    Uses aliases from class_config.json (role="leaf" in main_model section).

    Args:
        cls_ids: (N,) array of integer class IDs from extract_main_detections.
        names: dict mapping class ID -> class name string.

    Returns:
        List of integer indices into the cls_ids array where the class is a leaf.
    """
    leaf_aliases = get_leaf_aliases()
    keep = []
    for i, c in enumerate(cls_ids):
        cname = str(class_name(names, int(c))).lower()
        if cname in leaf_aliases:
            keep.append(i)
    return keep


def dedup_leaf_boxes(keep_indices, xyxy, confs, iou_thresh=0.5, ioa_thresh=0.6):
    """
    Remove duplicate and contained leaf bounding boxes.

    Why: The model sometimes detects the same leaf twice with slightly different
    boxes, or detects a small leaf that is actually inside a larger leaf's box.
    This creates false counts. We solve it in two passes:

    Pass 1 (IoU dedup): Sort by confidence, greedily keep boxes that don't
    overlap too much (IoU < 0.5) with any already-kept box.

    Pass 2 (Containment removal): If a smaller box is mostly inside a bigger box
    (IoA >= 0.6), remove the smaller one. IoA = intersection / area_of_smaller_box.

    Args:
        keep_indices: list of detection indices that passed the leaf class filter.
        xyxy: (N, 4) bounding box array.
        confs: (N,) confidence array.
        iou_thresh: IoU threshold for deduplication (default 0.5).
        ioa_thresh: IoA threshold for containment removal (default 0.6).

    Returns:
        Filtered list of detection indices with duplicates removed.
    """
    if not keep_indices or confs is None or xyxy is None:
        return keep_indices

    def iou(a, b):
        """Compute Intersection over Union between two boxes."""
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

    # Pass 1: IoU-based deduplication (keep highest confidence first)
    sorted_idx = sorted(keep_indices, key=lambda i: confs[i], reverse=True)
    dedup = []
    for idx in sorted_idx:
        box = xyxy[idx]
        if all(iou(box, xyxy[j]) < iou_thresh for j in dedup):
            dedup.append(idx)

    # Pass 2: Remove smaller boxes that are mostly contained inside a larger box
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
                continue  # Only check if j is bigger
            # Full containment check
            if x1 >= X1 and y1 >= Y1 and x2 <= X2 and y2 <= Y2:
                contained = True
                break
            # Partial containment: compute IoA (intersection / area_of_smaller)
            inter_x1 = max(x1, X1)
            inter_y1 = max(y1, Y1)
            inter_x2 = min(x2, X2)
            inter_y2 = min(y2, Y2)
            inter_w = max(0.0, inter_x2 - inter_x1)
            inter_h = max(0.0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h
            ioa = inter_area / area_i
            if ioa >= ioa_thresh:
                contained = True
                break
        if not contained:
            final.append(idx)
    return final


def dedup_indices_keep_biggest_overlap(indices, xyxy, overlap_thresh: float = 0.8) -> List[int]:
    """
    Deduplicate boxes by overlap ratio and keep the biggest area box.

    Overlap ratio = intersection / min(area_a, area_b). If ratio >= threshold,
    the smaller box is dropped. This matches "same area overlap" behavior better
    than IoU for nested/near-duplicate detections.
    """
    if not indices or xyxy is None:
        return indices
    try:
        thr = float(overlap_thresh)
    except (TypeError, ValueError):
        thr = 0.8
    thr = max(0.0, min(1.0, thr))

    def _area(i: int) -> float:
        x1, y1, x2, y2 = xyxy[i]
        return max(1.0, (float(x2) - float(x1)) * (float(y2) - float(y1)))

    def _overlap_min_area(i: int, j: int) -> float:
        ax1, ay1, ax2, ay2 = xyxy[i]
        bx1, by1, bx2, by2 = xyxy[j]
        iw = max(0.0, min(float(ax2), float(bx2)) - max(float(ax1), float(bx1)))
        ih = max(0.0, min(float(ay2), float(by2)) - max(float(ay1), float(by1)))
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        return inter / min(_area(i), _area(j))

    # Keep bigger areas first.
    ordered = sorted(indices, key=_area, reverse=True)
    kept: List[int] = []
    for i in ordered:
        if all(_overlap_min_area(i, j) < thr for j in kept):
            kept.append(i)
    # Keep stable ascending index order for downstream logic.
    kept.sort()
    return kept


def dedup_defects_keep_biggest_overlap(defects: list, overlap_thresh: float = 0.8) -> list:
    """Deduplicate defect dicts by overlap and keep biggest boxes."""
    if not defects:
        return defects
    boxes = [d.get("box") for d in defects]
    indices = list(range(len(defects)))
    keep = dedup_indices_keep_biggest_overlap(indices, boxes, overlap_thresh=overlap_thresh)
    return [defects[i] for i in keep]


# ---------------------------------------------------------------------------
# Mask/polygon to bounding box converters
# ---------------------------------------------------------------------------

def mask_to_box(mask, frame_shape):
    """
    Convert a binary segmentation mask to a tight bounding box.

    Args:
        mask: 2D array or tensor (H, W) with values 0-1.
        frame_shape: (H, W, C) shape of the original frame.

    Returns:
        (x1, y1, x2, y2) tuple, or None if the mask is empty.
    """
    if mask is None:
        return None
    m = mask
    if hasattr(m, "cpu"):
        m = m.cpu().numpy()
    h, w = frame_shape[:2]
    if m.shape[:2] != (h, w):
        m = cv2.resize(m, (w, h))
    m = (m > 0.5).astype(np.uint8)
    if m.max() == 0:
        return None
    ys, xs = np.where(m == 1)
    if ys.size == 0 or xs.size == 0:
        return None
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max())
    y2 = int(ys.max())
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def poly_to_box(poly):
    """
    Convert a polygon contour (Nx2 array of xy points) to a tight bounding box.

    Args:
        poly: (N, 2) numpy array of [x, y] polygon vertices.

    Returns:
        (x1, y1, x2, y2) tuple, or None if the polygon is empty/invalid.
    """
    if poly is None or len(poly) == 0:
        return None
    xs = poly[:, 0]
    ys = poly[:, 1]
    if xs.size == 0 or ys.size == 0:
        return None
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max())
    y2 = int(ys.max())
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def tighten_boxes_with_masks(keep_indices, xyxy, masks, masks_xy, frame_shape):
    """
    Replace YOLO bounding boxes with tighter boxes derived from segmentation masks.

    Args:
        keep_indices: list of detection indices to process.
        xyxy: (N, 4) original bounding boxes (will be copied, not modified).
        masks: segmentation mask tensor from the model.
        masks_xy: polygon contour list from the model.
        frame_shape: (H, W, C) shape of the frame.

    Returns:
        New (N, 4) bounding box array with tightened boxes for leaf indices.
    """
    xyxy_tight = xyxy.copy()
    for i in keep_indices:
        tight = None
        if masks_xy is not None and i < len(masks_xy):
            tight = poly_to_box(masks_xy[i])
        if tight is None and masks is not None and i < len(masks):
            tight = mask_to_box(masks[i], frame_shape)
        if tight is not None:
            x1, y1, x2, y2 = tight
            xyxy_tight[i] = (x1, y1, x2, y2)
    return xyxy_tight


# ---------------------------------------------------------------------------
# Stage 2: Leaf crop (shared by defect model and OpenCV stem measurement)
# ---------------------------------------------------------------------------

def extract_masked_leaf_crop(frame_bgr, leaf_box: Tuple[float, float, float, float], leaf_mask):
    """
    Crop the leaf bbox, then keep only pixels inside the stage-1 seg mask (polygon);
    pixels outside the mask in the crop are set to white (BGR) for stage-2 input.

    Returns:
        BGR numpy crop, or None if the region is empty or mask excludes all pixels.
    """
    margin = 3
    x1, y1, x2, y2 = leaf_box
    ix1 = max(0, int(x1) - margin)
    iy1 = max(0, int(y1) - margin)
    ix2 = min(frame_bgr.shape[1], int(x2) + margin)
    iy2 = min(frame_bgr.shape[0], int(y2) + margin)
    if ix2 <= ix1 or iy2 <= iy1:
        return None

    crop = frame_bgr[iy1:iy2, ix1:ix2]
    if crop.size == 0:
        return None

    if leaf_mask is not None:
        mask = leaf_mask
        if hasattr(mask, "cpu"):
            mask = mask.cpu().numpy()
        if mask.shape[:2] != frame_bgr.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.float32),
                (frame_bgr.shape[1], frame_bgr.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            mask = mask.astype(np.float32)
        mask_crop = mask[iy1:iy2, ix1:ix2]
        mask_bin = (mask_crop > 0.5).astype(np.uint8)
        kern_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kern_close)
        kern_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kern_open)
        # Shrink the edge slightly so stage-2 sees less dark halo/background.
        kern_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask_bin = cv2.erode(mask_bin, kern_erode, iterations=2)
        if mask_bin.size == 0 or mask_bin.max() == 0:
            return None
        crop = crop.copy()
        # White background outside leaf polygon (defect model + OpenCV stem expect uniform bg)
        crop[mask_bin == 0] = (255, 255, 255)

    return crop


# ---------------------------------------------------------------------------
# Stage 2: Defect detection on individual leaf crops
# ---------------------------------------------------------------------------

def run_defect_batch(
    defect_model,
    frame_bgr,
    leaf_boxes: List[Tuple[float, float, float, float]],
    leaf_masks: list,
    imgsz: int | None,
    conf: float,
    device: int,
    min_area_px: dict | None = None,
) -> List[list]:
    """
    Run defect model on all leaves in one batched GPU call (RF-DETR dynamic-batch engine).
    Returns list of defect-dict lists, one per leaf, in the same order as leaf_boxes.
    Falls back to sequential run_defect_on_leaf if batch not supported.
    """
    if defect_model is None or bool(getattr(defect_model, "_gomicro_disabled", False)):
        return [[] for _ in leaf_boxes]

    _cls = getattr(type(defect_model), "__name__", "")
    if _cls != "RFDETRDefectWrapper" or not defect_model.supports_batch:
        return [
            run_defect_on_leaf(defect_model, frame_bgr, box, mask, imgsz, conf, device, min_area_px=min_area_px)
            for box, mask in zip(leaf_boxes, leaf_masks)
        ]

    # Compute min_conf across all defect classes (same as run_defect_on_leaf)
    cc = get_class_config()
    defect_section = cc.get("defect_model", {})
    default_entry = defect_section.get("_default", {})
    alias_map = _build_alias_map(defect_section)
    min_conf = conf
    for key, entry in defect_section.items():
        if key.startswith("_"):
            continue
        c = float(entry.get("confidence", default_entry.get("confidence", conf)))
        if c < min_conf:
            min_conf = c

    # Extract and pre-scale all crops
    crops = []
    offsets = []   # (ix1, iy1, scale_x, scale_y) per leaf
    valid_idx = []  # indices of leaves that produced a non-empty crop
    for i, (leaf_box, leaf_mask) in enumerate(zip(leaf_boxes, leaf_masks)):
        crop = extract_masked_leaf_crop(frame_bgr, leaf_box, leaf_mask)
        if crop is None:
            offsets.append(None)
            continue
        _MAX_CROP = 640
        h_c, w_c = crop.shape[:2]
        sx = sy = 1.0
        if h_c > _MAX_CROP or w_c > _MAX_CROP:
            scale = _MAX_CROP / max(h_c, w_c)
            nw = max(1, int(w_c * scale))
            nh = max(1, int(h_c * scale))
            crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)
            sx = nw / w_c
            sy = nh / h_c
        x1, y1 = leaf_box[0], leaf_box[1]
        offsets.append((max(0, int(x1)), max(0, int(y1)), sx, sy))
        crops.append(crop)
        valid_idx.append(i)

    all_defects: List[list] = [[] for _ in leaf_boxes]
    if not crops:
        return all_defects

    try:
        batch_results = defect_model.predict_batch(crops, conf=min_conf, imgsz=imgsz)
    except Exception as exc:
        setattr(defect_model, "_gomicro_disabled", True)
        if not getattr(defect_model, "_gomicro_disable_warned", False):
            logging.exception("Defect model batch call failed — stage-2 disabled: %s", exc)
            setattr(defect_model, "_gomicro_disable_warned", True)
        return all_defects

    for batch_pos, leaf_i in enumerate(valid_idx):
        d_results = batch_results[batch_pos]
        if not d_results or d_results[0].boxes is None or len(d_results[0].boxes) == 0:
            continue
        d_res = d_results[0]
        ix1, iy1, scale_x, scale_y = offsets[leaf_i]
        d_xyxy = d_res.boxes.xyxy.cpu().numpy()
        d_conf = d_res.boxes.conf.cpu().numpy()
        d_cls_ids = d_res.boxes.cls.cpu().numpy().astype(int)
        d_names = d_res.names if hasattr(d_res, "names") else {}

        defects = []
        for j in range(len(d_xyxy)):
            dx1, dy1, dx2, dy2 = d_xyxy[j]
            raw_cname = class_name(d_names, int(d_cls_ids[j]))
            raw_cname = _resolve_raw_class_name(raw_cname, defect_section)
            cconf = float(d_conf[j])
            w = float(dx2) - float(dx1)
            h = float(dy2) - float(dy1)
            area = max(0.0, w * h)
            cls_key, cls_entry = resolve_class(raw_cname, defect_section, alias_map)
            mapped_label = cls_entry.get("label", cls_key)
            conf_threshold = float(cls_entry.get("confidence", default_entry.get("confidence", conf)))
            if cconf < conf_threshold:
                continue
            min_w, min_h = _get_min_hw(cls_entry, default_entry)
            if w < min_w or h < min_h:
                continue
            if min_area_px is not None:
                min_area = float(min_area_px.get(cls_key, min_area_px.get("other", 0)))
            else:
                min_area = float(cls_entry.get("min_area_px", default_entry.get("min_area_px", 0)))
            if area < min_area:
                continue
            defects.append({
                "box": (ix1 + int(dx1 / scale_x), iy1 + int(dy1 / scale_y),
                        ix1 + int(dx2 / scale_x), iy1 + int(dy2 / scale_y)),
                "name": cls_key,
                "label": mapped_label,
                "conf": cconf,
            })
        all_defects[leaf_i] = dedup_defects_keep_biggest_overlap(defects, overlap_thresh=0.8)

    return all_defects


def run_defect_on_leaf(
    defect_model,
    frame_bgr,
    leaf_box: Tuple[float, float, float, float],
    leaf_mask,
    imgsz: int | None,
    conf: float,
    device: int,
    min_area_px: dict | None = None,
    debug_log_raw: bool = False,
    debug_log_filter_reasons: bool = False,
    debug_leaf_tid: int | None = None,
):
    """
    Run the defect model on a single cropped leaf region.

    All class names, confidence thresholds, and filtering are driven by
    class_config.json (defect_model section).

    Args:
        defect_model: YOLO model for defect detection.
        frame_bgr: Full frame as numpy array (H, W, 3) BGR.
        leaf_box: (x1, y1, x2, y2) bounding box of the leaf in frame coordinates.
        leaf_mask: Segmentation mask for this leaf (optional, from seg model).
        imgsz: Inference input size override (None = model default).
        conf: Base confidence threshold (used if class has no specific one).
        device: GPU device index.
        min_area_px: Override dict mapping defect type -> minimum pixel area.
                     If None, uses values from class_config.json.
        debug_log_raw: If True, print all raw candidates before filtering.
        debug_log_filter_reasons: If True, print why each detection was dropped.
        debug_leaf_tid: Track ID of this leaf (for debug log context).

    Returns:
        List of defect dicts, each with:
        - "box": (x1, y1, x2, y2) in full-frame coordinates
        - "name": mapped_label from class_config.json
        - "conf": float confidence score
    """
    if defect_model is None or bool(getattr(defect_model, "_gomicro_disabled", False)):
        return []

    crop = extract_masked_leaf_crop(frame_bgr, leaf_box, leaf_mask)
    if crop is None:
        return []

    # Cap crop to fixed max size so TRT workspace stays bounded (never grows past one size).
    _MAX_CROP = 640
    scale_x = scale_y = 1.0
    _h_crop, _w_crop = crop.shape[:2]
    if _h_crop > _MAX_CROP or _w_crop > _MAX_CROP:
        _scale = _MAX_CROP / max(_h_crop, _w_crop)
        _new_w = max(1, int(_w_crop * _scale))
        _new_h = max(1, int(_h_crop * _scale))
        crop = cv2.resize(crop, (_new_w, _new_h), interpolation=cv2.INTER_LINEAR)
        scale_x = _new_w / _w_crop
        scale_y = _new_h / _h_crop

    x1, y1, x2, y2 = leaf_box
    ix1 = max(0, int(x1))
    iy1 = max(0, int(y1))

    # Run defect model on the crop
    # Use the lowest per-class confidence from class_config.json so YOLO
    # doesn't pre-filter detections before our per-class thresholds apply.
    cc = get_class_config()
    defect_section = cc.get("defect_model", {})
    default_entry = defect_section.get("_default", {})
    min_conf = conf
    for key, entry in defect_section.items():
        if key.startswith("_"):
            continue
        c = float(entry.get("confidence", default_entry.get("confidence", conf)))
        if c < min_conf:
            min_conf = c
    predict_kwargs = {
        "source": crop,
        "verbose": False,
        "conf": min_conf,
        "device": device,
    }
    if imgsz is not None:
        predict_kwargs["imgsz"] = imgsz

    _cls = getattr(type(defect_model), "__name__", "")
    try:
        if _cls == "RFDETRDefectWrapper":
            # RF-DETR wrapper only accepts source / conf / imgsz; avoid YOLO-only kwargs.
            rk = {"source": crop, "conf": min_conf}
            if imgsz is not None:
                rk["imgsz"] = imgsz
            d_results = defect_model.predict(**rk)
        else:
            with torch.inference_mode():
                d_results = defect_model.predict(**predict_kwargs)
    except Exception as exc:
        setattr(defect_model, "_gomicro_disabled", True)
        warned = bool(getattr(defect_model, "_gomicro_disable_warned", False))
        if not warned:
            logging.exception(
                "Defect model (stage-2) failed — stage-2 disabled until restart: %s",
                exc,
            )
            _detect_print(f"WARNING: defect model disabled after runtime failure: {exc}")
            setattr(defect_model, "_gomicro_disable_warned", True)
        return []

    d_res = d_results[0]
    if d_res.boxes is None or len(d_res.boxes) == 0:
        try:
            del d_results
            del d_res
            del crop
        except Exception:
            pass
        return []

    # Parse defect model output
    d_xyxy = d_res.boxes.xyxy.cpu().numpy()
    d_conf = d_res.boxes.conf.cpu().numpy()
    d_cls_ids = d_res.boxes.cls.cpu().numpy().astype(int)
    d_names = d_res.names if hasattr(d_res, "names") else {}

    # Load class config for defect model
    cc = get_class_config()
    defect_section = cc.get("defect_model", {})
    alias_map = _build_alias_map(defect_section)
    default_entry = defect_section.get("_default", {})

    # Filter and map each detection
    defects = []
    raw_items = []
    dropped_items = []
    for j in range(len(d_xyxy)):
        dx1, dy1, dx2, dy2 = d_xyxy[j]
        raw_cname = class_name(d_names, int(d_cls_ids[j]))
        raw_cname = _resolve_raw_class_name(raw_cname, defect_section)
        cconf = float(d_conf[j])
        area = max(0.0, (dx2 - dx1) * (dy2 - dy1))
        w = float(dx2) - float(dx1)
        h = float(dy2) - float(dy1)

        # Resolve via class_config.json
        cls_key, cls_entry = resolve_class(raw_cname, defect_section, alias_map)
        mapped_label = cls_entry.get("label", cls_key)

        raw_items.append(f"{raw_cname}({mapped_label})@{cconf:.2f}[a={area:.1f}]")

        # Per-class confidence threshold from config
        conf_threshold = float(cls_entry.get("confidence", default_entry.get("confidence", conf)))
        if cconf < conf_threshold:
            if debug_log_filter_reasons:
                dropped_items.append(
                    f"{mapped_label}@{cconf:.2f}[a={area:.1f}] drop=conf<{conf_threshold:.2f}"
                )
            continue

        # Per-class minimum width/height (pixels) from config
        min_w, min_h = _get_min_hw(cls_entry, default_entry)
        if w < min_w or h < min_h:
            if debug_log_filter_reasons:
                dropped_items.append(
                    f"{mapped_label}@{cconf:.2f}[w={w:.1f},h={h:.1f}] drop=wh<{min_w:.1f},{min_h:.1f}"
                )
            continue

        # Per-class minimum area from config (override from min_area_px param if provided)
        if min_area_px is not None:
            min_area = float(min_area_px.get(cls_key, min_area_px.get("other", 0)))
        else:
            min_area = float(cls_entry.get("min_area_px", default_entry.get("min_area_px", 0)))
        if area < min_area:
            if debug_log_filter_reasons:
                dropped_items.append(
                    f"{mapped_label}@{cconf:.2f}[a={area:.1f}] drop=area<{min_area:.1f}"
                )
            continue

        # Map crop coordinates back to full-frame coordinates (undo scale_x/scale_y)
        defects.append({
            "box": (ix1 + int(dx1 / scale_x), iy1 + int(dy1 / scale_y), ix1 + int(dx2 / scale_x), iy1 + int(dy2 / scale_y)),
            "name": cls_key,           # Config key e.g. "yellow spots", "white spots"
            "label": mapped_label,     # Short label e.g. "YL", "WS" (drawn on frame)
            "conf": cconf,
        })

    # Debug logging
    if debug_log_raw:
        leaf_txt = f"{debug_leaf_tid}" if debug_leaf_tid is not None else "-1"
        raw_txt = ", ".join(raw_items) if raw_items else "none"
        _detect_print(f"[debug][defect_model][raw] leaf_tid={leaf_txt} candidates={raw_txt}")
    if debug_log_filter_reasons:
        leaf_txt = f"{debug_leaf_tid}" if debug_leaf_tid is not None else "-1"
        drop_txt = ", ".join(dropped_items) if dropped_items else "none"
        _detect_print(f"[debug][defect_model][drop] leaf_tid={leaf_txt} reasons={drop_txt}")
    out = dedup_defects_keep_biggest_overlap(defects, overlap_thresh=0.8)
    try:
        del d_results
    except Exception:
        pass
    try:
        del d_res
    except Exception:
        pass
    try:
        del crop
    except Exception:
        pass
    return out
