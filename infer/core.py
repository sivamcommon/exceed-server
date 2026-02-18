"""Core detection helpers: extract detections, select leaves, run defect model."""

from typing import List, Tuple

import cv2

LEAF_CLASS_ALIASES = ("leaf", "spinach")
STEM_CLASS_ALIASES = ("stem", "leaf stem")


def class_name(names, cls_id: int) -> str:
    if isinstance(names, dict):
        return names.get(cls_id, str(cls_id))
    if isinstance(names, list) and 0 <= cls_id < len(names):
        return names[cls_id]
    return str(cls_id)


def extract_main_detections(result):
    """Return (xyxy, conf, cls_ids, track_ids, names) from main model result."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None, None, None, None, result.names if hasattr(result, "names") else {}
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    track_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None
    names = result.names if hasattr(result, "names") else {}
    return xyxy, conf, cls_ids, track_ids, names


def select_leaf_indices(cls_ids, names) -> List[int]:
    leaf_aliases = set(LEAF_CLASS_ALIASES)
    keep = []
    for i, c in enumerate(cls_ids):
        cname = str(class_name(names, int(c))).lower()
        if cname in leaf_aliases:
            keep.append(i)
    return keep


def run_defect_on_leaf(
    defect_model,
    frame_bgr,
    leaf_box: Tuple[float, float, float, float],
    imgsz: int | None,
    conf: float,
    device: int,
    conf_stem: float = 0.45,
    conf_other: float = 0.45,
    min_area_px: dict | None = None,
):
    """Run defect model on a single leaf crop and return list of defect dicts."""
    x1, y1, x2, y2 = leaf_box
    ix1 = max(0, int(x1))
    iy1 = max(0, int(y1))
    ix2 = min(frame_bgr.shape[1], int(x2))
    iy2 = min(frame_bgr.shape[0], int(y2))
    if ix2 <= ix1 or iy2 <= iy1:
        return []
    crop = frame_bgr[iy1:iy2, ix1:ix2]
    if crop.size == 0:
        return []

    predict_kwargs = {
        "source": crop,
        "verbose": False,
        "conf": conf,
        "device": device,
    }
    if imgsz is not None:
        predict_kwargs["imgsz"] = imgsz
    d_results = defect_model.predict(**predict_kwargs)
    d_res = d_results[0]
    if d_res.boxes is None or len(d_res.boxes) == 0:
        return []

    d_xyxy = d_res.boxes.xyxy.cpu().numpy()
    d_conf = d_res.boxes.conf.cpu().numpy()
    d_cls_ids = d_res.boxes.cls.cpu().numpy().astype(int)
    d_names = d_res.names if hasattr(d_res, "names") else {}

    defects = []
    for j in range(len(d_xyxy)):
        dx1, dy1, dx2, dy2 = d_xyxy[j]
        cname = class_name(d_names, int(d_cls_ids[j]))
        cconf = float(d_conf[j])
        cname_l = str(cname).strip().lower()
        if cname_l in ("stem", "leaf stem"):
            if cconf < conf_stem:
                continue
        else:
            if cconf < conf_other:
                continue
        # Minimum area filter per defect type
        key = "other"
        if "yellow" in cname_l:
            key = "yellow"
        elif "white" in cname_l:
            key = "white"
        elif "ipd" in cname_l:
            key = "ipd"
        elif cname_l in ("stem", "leaf stem"):
            key = "stem"
        min_area = 0 if min_area_px is None else float(min_area_px.get(key, 0))
        area = max(0.0, (dx2 - dx1) * (dy2 - dy1))
        if area < min_area:
            continue
        defects.append({
            "box": (ix1 + int(dx1), iy1 + int(dy1), ix1 + int(dx2), iy1 + int(dy2)),
            "name": cname,
            "conf": cconf,
        })
    return defects
