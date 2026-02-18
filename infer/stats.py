"""Tracking stats and summaries."""

LEAF_CLASS_ALIASES = ("leaf", "spinach")
STEM_CLASS_ALIASES = ("stem", "leaf stem")


def update_tracking_stats(stats, xyxy, cls_ids, track_ids, names, keep_indices):
    """Update per-track stats for kept detections."""
    if stats is None or track_ids is None:
        return

    tracks = stats.setdefault("tracks", {})
    stats.setdefault("leaf_defects", {})
    track_sizes = stats.setdefault("track_sizes", {})
    stats["frame_idx"] = stats.get("frame_idx", 0) + 1

    def class_name(names_map, cls_id):
        if isinstance(names_map, dict):
            return names_map.get(cls_id, str(cls_id))
        if isinstance(names_map, list) and 0 <= cls_id < len(names_map):
            return names_map[cls_id]
        return str(cls_id)

    for i in keep_indices:
        if i < 0 or i >= len(cls_ids):
            continue
        tid = int(track_ids[i])
        cname = class_name(names, int(cls_ids[i]))
        tinfo = tracks.setdefault(tid, {"class_counts": {}, "last_seen": 0})
        tinfo["class_counts"][cname] = tinfo["class_counts"].get(cname, 0) + 1
        tinfo["last_seen"] = stats["frame_idx"]
        x1, y1, x2, y2 = xyxy[i]
        w = max(1.0, (x2 - x1))
        h = max(1.0, (y2 - y1))
        prev = track_sizes.get(tid)
        if prev is None:
            track_sizes[tid] = (w, h)
        else:
            track_sizes[tid] = (max(prev[0], w), max(prev[1], h))


def compute_tracking_summary(stats, leaf_class: str = "leaf", defect_classes=None):
    """Summarize tracked objects and defect counts."""
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
        if final_class.get(leaf_tid) == leaf_class and len(defect_set) > 0:
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
    size_cfg,
    mm_per_px,
    leaf_aliases=LEAF_CLASS_ALIASES,
    stem_aliases=STEM_CLASS_ALIASES,
):
    """Compute in-spec/out-of-spec size summary for tracked objects."""
    if stats is None:
        return {"leaf": {}, "stem": {}}

    tracks = stats.get("tracks", {})
    track_sizes = stats.get("track_sizes", {})
    leaf_stems = stats.get("leaf_stems", {})

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

    # Stem sizes come from second-model detections (per-leaf lists).
    if leaf_stems:
        h_min, h_max = get_range(size_cfg.get("stem", {}), "height")
        for height_mm in leaf_stems.values():
            if height_mm is None:
                continue
            summary["stem"]["count"] += 1
            if h_min <= float(height_mm) <= h_max:
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
