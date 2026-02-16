import cv2
from ultralytics import YOLO
#nano trackers/bytetrack.yaml
ENGINE_PATH = "best_11f.engine"
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
LEAF_CLASS_ALIASES = ("leaf", "spinach")
STEM_CLASS_ALIASES = ("stem", "leaf stem")

def load_model(engine_path: str = ENGINE_PATH):
    return YOLO(engine_path)

def draw_filtered_boxes(frame_bgr, result, keep_indices, color=(0, 255, 0)):
    if result.boxes is None or len(keep_indices) == 0:
        return frame_bgr.copy()

    img = frame_bgr.copy()
    xyxy = result.boxes.xyxy.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    names = result.names if hasattr(result, "names") else {}

    def cls_name(cls_id):
        if isinstance(names, dict):
            return names.get(cls_id, str(cls_id))
        if isinstance(names, list) and 0 <= cls_id < len(names):
            return names[cls_id]
        return str(cls_id)

    for i in keep_indices:
        if i < 0 or i >= len(xyxy):
            continue
        x1, y1, x2, y2 = xyxy[i]
        cname = cls_name(cls_ids[i])
        if cname == "leaf":
            box_color = (0, 255, 0)
        elif cname == "stem":
            box_color = (0, 0, 0)
        else:
            box_color = (0, 0, 255)
        label = f"{cname} {conf[i]:.2f}"
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, box_color, 2)
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

def group_leaf_detections(
    result,
    leaf_class: str = "leaf",
    stem_class: str = "leaf stem",
    defect_classes=None,
    min_ioa: float = 0.2,
):
    if defect_classes is None:
        defect_classes = {"IPD", "IPD2", "WS", "WS2", "YL"}

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return {"groups": [], "keep_indices": []}

    names = result.names if hasattr(result, "names") else {}
    xyxy = boxes.xyxy.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy().astype(int)

    def cls_name(cls_id):
        if isinstance(names, dict):
            return names.get(cls_id, str(cls_id))
        if isinstance(names, list) and 0 <= cls_id < len(names):
            return names[cls_id]
        return str(cls_id)

    leaf_indices = [i for i, c in enumerate(cls_ids) if cls_name(c) == leaf_class]
    if not leaf_indices:
        return {"groups": [], "keep_indices": []}

    groups = []
    keep = set(leaf_indices)

    for leaf_idx in leaf_indices:
        lx1, ly1, lx2, ly2 = xyxy[leaf_idx]
        leaf_group = {"leaf_index": leaf_idx, "leaf_box": [lx1, ly1, lx2, ly2], "members": []}

        for i, c in enumerate(cls_ids):
            name = cls_name(c)
            if i == leaf_idx or (name != stem_class and name not in defect_classes):
                continue

            x1, y1, x2, y2 = xyxy[i]
            inter_x1 = max(lx1, x1)
            inter_y1 = max(ly1, y1)
            inter_x2 = min(lx2, x2)
            inter_y2 = min(ly2, y2)
            inter_w = max(0.0, inter_x2 - inter_x1)
            inter_h = max(0.0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h
            area = max(1.0, (x2 - x1) * (y2 - y1))
            ioa = inter_area / area

            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            center_inside = (lx1 <= cx <= lx2) and (ly1 <= cy <= ly2)

            if ioa >= min_ioa or center_inside:
                leaf_group["members"].append({"index": i, "class": name, "box": [x1, y1, x2, y2]})
                keep.add(i)

        groups.append(leaf_group)

    return {"groups": groups, "keep_indices": sorted(keep)}

def filter_grouped_detections(result, grouping, leaf_class: str = "leaf", stem_class: str = "leaf stem"):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return grouping

    xyxy = boxes.xyxy.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()
    names = result.names if hasattr(result, "names") else {}

    def cls_name(cls_id):
        if isinstance(names, dict):
            return names.get(cls_id, str(cls_id))
        if isinstance(names, list) and 0 <= cls_id < len(names):
            return names[cls_id]
        return str(cls_id)


    leaf_indices = [i for i, c in enumerate(cls_ids) if cls_name(c) == leaf_class]
    keep = set(grouping.get("keep_indices", []))

    # Filter out low-confidence leaves (< 0.6)
    high_conf_leaves = set([i for i in leaf_indices if conf[i] >= 0.6])
    leaf_indices = [i for i in leaf_indices if i in high_conf_leaves]

    # Remove duplicate leaves via containment (leaf inside leaf)
    leaves_to_remove = set()
    min_ioa = 0.8  # how much of the smaller leaf is inside the bigger one
    for i in leaf_indices:
        x1, y1, x2, y2 = xyxy[i]
        area_i = max(1.0, (x2 - x1) * (y2 - y1))
        for j in leaf_indices:
            if i == j:
                continue
            X1, Y1, X2, Y2 = xyxy[j]
            area_j = max(1.0, (X2 - X1) * (Y2 - Y1))
            inter_x1 = max(x1, X1)
            inter_y1 = max(y1, Y1)
            inter_x2 = min(x2, X2)
            inter_y2 = min(y2, Y2)
            inter_w = max(0.0, inter_x2 - inter_x1)
            inter_h = max(0.0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h
            if inter_area <= 0:
                continue
            # IoA of smaller box inside larger box
            if area_i <= area_j:
                ioa = inter_area / area_i
                if ioa >= min_ioa:
                    leaves_to_remove.add(i)
            else:
                ioa = inter_area / area_j
                if ioa >= min_ioa:
                    leaves_to_remove.add(j)
    for leaf_idx in leaves_to_remove:
        if leaf_idx in keep:
            keep.remove(leaf_idx)

    # For each leaf group, keep only the biggest stem.
    for group in grouping.get("groups", []):
        stem_members = [m for m in group.get("members", []) if m.get("class") == stem_class]
        if len(stem_members) <= 1:
            continue
        max_area = -1.0
        keep_idx = None
        for m in stem_members:
            idx = m.get("index")
            if idx is None:
                continue
            x1, y1, x2, y2 = xyxy[idx]
            area = max(1.0, (x2 - x1) * (y2 - y1))
            if area > max_area:
                max_area = area
                keep_idx = idx
        for m in stem_members:
            idx = m.get("index")
            if idx is not None and idx != keep_idx and idx in keep:
                keep.remove(idx)

    grouping["keep_indices"] = sorted(keep)
    return grouping

def update_tracking_stats(
    stats,
    result,
    grouping,
    leaf_class: str = "leaf",
    defect_classes=None,
):
    if defect_classes is None:
        defect_classes = {"IPD", "IPD2", "WS", "WS2", "YL"}

    if stats is None:
        return

    tracks = stats.setdefault("tracks", {})
    leaf_defects = stats.setdefault("leaf_defects", {})
    track_sizes = stats.setdefault("track_sizes", {})
    stats["frame_idx"] = stats.get("frame_idx", 0) + 1

    boxes = result.boxes
    if boxes is None or boxes.id is None:
        return

    xyxy = boxes.xyxy.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    track_ids = boxes.id.cpu().numpy().astype(int)
    names = result.names if hasattr(result, "names") else {}

    def cls_name(cls_id):
        if isinstance(names, dict):
            return names.get(cls_id, str(cls_id))
        if isinstance(names, list) and 0 <= cls_id < len(names):
            return names[cls_id]
        return str(cls_id)

    keep_indices = grouping.get("keep_indices", [])
    for i in keep_indices:
        if i < 0 or i >= len(cls_ids):
            continue
        tid = int(track_ids[i])
        cname = cls_name(cls_ids[i])
        tinfo = tracks.setdefault(tid, {"class_counts": {}, "last_seen": 0})
        tinfo["class_counts"][cname] = tinfo["class_counts"].get(cname, 0) + 1
        tinfo["last_seen"] = stats["frame_idx"]
        x1, y1, x2, y2 = xyxy[i]
        track_sizes[tid] = (max(1.0, (x2 - x1)), max(1.0, (y2 - y1)))

    for group in grouping.get("groups", []):
        leaf_idx = group.get("leaf_index")
        if leaf_idx is None or leaf_idx < 0 or leaf_idx >= len(track_ids):
            continue
        leaf_tid = int(track_ids[leaf_idx])
        for member in group.get("members", []):
            midx = member.get("index")
            mclass = member.get("class")
            if midx is None or mclass not in defect_classes:
                continue
            if midx < 0 or midx >= len(track_ids):
                continue
            defect_tid = int(track_ids[midx])
            leaf_defects.setdefault(leaf_tid, set()).add(defect_tid)

def compute_tracking_summary(
    stats,
    leaf_class: str = "leaf",
    defect_classes=None,
):
    if defect_classes is None:
        defect_classes = {"IPD", "IPD2", "WS", "WS2", "YL"}

    tracks = stats.get("tracks", {})
    leaf_defects = stats.get("leaf_defects", {})

    final_class = {}
    for tid, info in tracks.items():
        counts = info.get("class_counts", {})
        if not counts:
            continue
        final = max(counts.items(), key=lambda kv: kv[1])[0]
        final_class[tid] = final

    total_leaf = sum(1 for c in final_class.values() if c == leaf_class)
    total_defects = sum(1 for c in final_class.values() if c in defect_classes)
    defect_counts = {d: 0 for d in defect_classes}
    for c in final_class.values():
        if c in defect_counts:
            defect_counts[c] += 1

    total_defected_leaf = 0
    for leaf_tid, defect_tids in leaf_defects.items():
        if final_class.get(leaf_tid) == leaf_class and len(defect_tids) > 0:
            total_defected_leaf += 1

    summary = {
        "total_leaf": total_leaf,
        "total_defects": total_defects,
        "total_defected_leaf": total_defected_leaf,
        "defect_counts": defect_counts,
        "tracks_seen": len(final_class),
    }
    return summary

def compute_size_summary(
    stats,
    size_cfg=SIZE_THRESHOLDS,
    mm_per_px: float = MM_PER_PX,
    leaf_aliases=LEAF_CLASS_ALIASES,
    stem_aliases=STEM_CLASS_ALIASES,
):
    if stats is None:
        return {"leaf": {}, "stem": {}}

    tracks = stats.get("tracks", {})
    track_sizes = stats.get("track_sizes", {})

    final_class = {}
    for tid, info in tracks.items():
        counts = info.get("class_counts", {})
        if not counts:
            continue
        final = max(counts.items(), key=lambda kv: kv[1])[0]
        final_class[tid] = final

    def init_bucket():
        return {"in_spec": 0, "out_spec": 0, "count": 0, "in_spec_pct": 0, "out_spec_pct": 0}

    summary = {
        "leaf": init_bucket(),
        "stem": init_bucket(),
    }

    leaf_set = set(leaf_aliases)
    stem_set = set(stem_aliases)

    def get_range(cfg, key):
        rng = cfg.get(key, {})
        return float(rng.get("min", 0.0)), float(rng.get("max", 0.0))

    for tid, cls_name in final_class.items():
        if tid not in track_sizes:
            continue
        w, h = track_sizes[tid]
        height_mm = max(w, h) * float(mm_per_px)
        width_mm = min(w, h) * float(mm_per_px)

        if cls_name in leaf_set:
            h_min, h_max = get_range(size_cfg.get("leaf", {}), "height")
            w_min, w_max = get_range(size_cfg.get("leaf", {}), "width")
            summary["leaf"]["count"] += 1
            if (h_min <= height_mm <= h_max) and (w_min <= width_mm <= w_max):
                summary["leaf"]["in_spec"] += 1
            else:
                summary["leaf"]["out_spec"] += 1
        elif cls_name in stem_set:
            h_min, h_max = get_range(size_cfg.get("stem", {}), "height")
            w_min, w_max = get_range(size_cfg.get("stem", {}), "width")
            summary["stem"]["count"] += 1
            if (h_min <= height_mm <= h_max) and (w_min <= width_mm <= w_max):
                summary["stem"]["in_spec"] += 1
            else:
                summary["stem"]["out_spec"] += 1

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
    add_percentages(summary["stem"])
    return summary

def infer_frame_leaf_grouped_tracked(
    model,
    frame_bgr,
    stats,
    imgsz: int = 640,
    conf: float = 0.45,
    device: int = 0,
    min_ioa: float = 0.2,
    size_cfg=SIZE_THRESHOLDS,
    mm_per_px: float = MM_PER_PX,
):
    results = model.track(
        source=frame_bgr,
        persist=True,
        verbose=False,
        imgsz=imgsz,
        conf=conf,
        iou=0.8,
        device=device,
    )
    result = results[0]
    grouping = group_leaf_detections(result, min_ioa=min_ioa)
    grouping = filter_grouped_detections(result, grouping)
    keep_indices = grouping["keep_indices"]
    annotated = draw_filtered_boxes(frame_bgr, result, keep_indices)
    update_tracking_stats(stats, result, grouping)
    summary = compute_tracking_summary(stats) if stats is not None else {}
    if stats is not None:
        summary["size_summary"] = compute_size_summary(stats, size_cfg=size_cfg, mm_per_px=mm_per_px)
    return annotated, summary
