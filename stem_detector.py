"""
Robust Leaf Stem (Tail) Detection using OpenCV
================================================

Design goals:
- NOT dependent on color, lighting, or exposure
- Works with any leaf color on any roughly uniform background
- Detects the "tail" — a narrow elongated protrusion from the main leaf body

Algorithm:
1. ROBUST SEGMENTATION: Adaptive method that works regardless of color/lighting
   - Otsu's thresholding on grayscale (auto-picks threshold)
   - Combined with edge-based segmentation as fallback
   - Flood-fill from corners to isolate background
   
2. CONTOUR ANALYSIS: Extract the leaf boundary
   - Find largest contour
   - Compute convex hull and convexity defects
   
3. TAIL DETECTION via geometry:
   - Compute distance transform → local width at every point
   - Skeleton + endpoint tracing with width profiling
   - The tail/stem is a narrow protrusion: width << leaf body width
   - Classify: no stem, short stem, medium stem, long stem

4. VALIDATION:
   - Stem must be elongated (length/width ratio > 2)
   - Stem must be narrow relative to leaf body
   - Stem must originate from a contour endpoint region
"""

import cv2
import numpy as np
import os
from typing import Tuple, Dict, List


# =============================================================================
# STEP 1: ROBUST SEGMENTATION (lighting/color independent)
# =============================================================================

def segment_leaf_robust(image: np.ndarray) -> np.ndarray:
    """
    Segment leaf from background using multiple strategies.
    Robust to varying color, lighting, exposure.
    
    Strategy:
    1. Otsu's auto-threshold (primary — works for bimodal histograms)
    2. Flood-fill from corners to identify background (secondary)
    3. If flood-fill removes >80% of Otsu's leaf, ignore flood-fill
       (it leaked through thin/bright regions)
    4. Morphological cleanup + hole filling
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # --- Method 1: Otsu's auto-threshold ---
    _, otsu_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # --- Method 2: Flood-fill from borders ---
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    flood_img = blur.copy()
    
    border_points = []
    step = max(1, min(h, w) // 20)
    for x in range(0, w, step):
        border_points.extend([(x, 0), (x, h - 1)])
    for y in range(0, h, step):
        border_points.extend([(0, y), (w - 1, y)])
    border_points.extend([(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)])
    
    for (px, py) in border_points:
        if flood_mask[py + 1, px + 1] == 0:
            cv2.floodFill(flood_img, flood_mask, (px, py), 255,
                         loDiff=12, upDiff=12,
                         flags=cv2.FLOODFILL_MASK_ONLY | (255 << 8))
    
    bg_mask = flood_mask[1:-1, 1:-1]
    flood_leaf = (bg_mask == 0).astype(np.uint8) * 255
    
    # --- Combine: check if flood-fill is trustworthy ---
    otsu_area = np.count_nonzero(otsu_mask)
    flood_area = np.count_nonzero(flood_leaf)
    
    if otsu_area > 0 and flood_area < otsu_area * 0.2:
        # Flood-fill leaked badly — trust Otsu alone
        mask = otsu_mask
    elif otsu_area > 0 and flood_area > otsu_area * 2.0:
        # Flood-fill is too conservative — trust Otsu alone
        mask = otsu_mask
    else:
        # Both are reasonable — intersect them for best result
        mask = cv2.bitwise_and(otsu_mask, flood_leaf)
        # If intersection is too small, fall back to Otsu
        if np.count_nonzero(mask) < otsu_area * 0.3:
            mask = otsu_mask
    
    # --- Morphological cleanup ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Fill holes: keep only largest contour filled
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        mask_filled = np.zeros_like(mask)
        cv2.drawContours(mask_filled, contours, -1, 255, -1)
        mask = mask_filled
    
    return mask


# =============================================================================
# STEP 2: SKELETON & WIDTH ANALYSIS
# =============================================================================

def compute_skeleton(mask: np.ndarray) -> np.ndarray:
    """
    Skeletonize using Zhang-Suen thinning (pure NumPy, no opencv-contrib needed).
    Falls back to cv2.ximgproc.thinning if available for speed.
    """
    # Try the fast C++ version first
    if hasattr(cv2, 'ximgproc'):
        try:
            return cv2.ximgproc.thinning(mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        except Exception:
            pass

    # Pure-NumPy Zhang-Suen thinning
    img = (mask > 0).astype(np.uint8)
    rows, cols = img.shape
    changed = True

    while changed:
        changed = False
        for step in (0, 1):
            # Pad to avoid boundary checks
            p = np.pad(img, 1, mode='constant')
            # 8-neighbors: P2=N, P3=NE, P4=E, P5=SE, P6=S, P7=SW, P8=W, P9=NW
            P2 = p[0:-2, 1:-1]
            P3 = p[0:-2, 2:]
            P4 = p[1:-1, 2:]
            P5 = p[2:,   2:]
            P6 = p[2:,   1:-1]
            P7 = p[2:,   0:-2]
            P8 = p[1:-1, 0:-2]
            P9 = p[0:-2, 0:-2]

            # B(P1) = number of non-zero neighbors
            B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9

            # A(P1) = number of 0->1 transitions in the ordered sequence P2..P9,P2
            neighbors = np.stack([P2, P3, P4, P5, P6, P7, P8, P9], axis=-1)
            neighbors_shifted = np.stack([P3, P4, P5, P6, P7, P8, P9, P2], axis=-1)
            A = np.sum((neighbors == 0) & (neighbors_shifted == 1), axis=-1)

            # Common conditions
            cond = (img == 1) & (B >= 2) & (B <= 6) & (A == 1)

            if step == 0:
                cond &= (P2 * P4 * P6 == 0)
                cond &= (P4 * P6 * P8 == 0)
            else:
                cond &= (P2 * P4 * P8 == 0)
                cond &= (P2 * P6 * P8 == 0)

            if np.any(cond):
                img[cond] = 0
                changed = True

    return (img * 255).astype(np.uint8)


def get_endpoints(skeleton: np.ndarray) -> List[Tuple[int, int]]:
    """Find skeleton endpoints using convolution (fast)."""
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    skel_bin = (skeleton > 0).astype(np.uint8)
    neighbor_count = cv2.filter2D(skel_bin, -1, kernel)
    ep_mask = (skel_bin > 0) & (neighbor_count == 1)
    ys, xs = np.where(ep_mask)
    return list(zip(xs.tolist(), ys.tolist()))


def trace_from_endpoint(skeleton: np.ndarray, endpoint: Tuple[int, int],
                        max_steps: int = 50000) -> List[Tuple[int, int]]:
    """
    Trace skeleton from endpoint inward.
    At junctions, follow the direction most aligned with current heading.
    """
    path = [endpoint]
    visited = set()
    visited.add(endpoint)
    current = endpoint
    prev = None
    h, w = skeleton.shape
    
    for _ in range(max_steps):
        x, y = current
        neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                nx, ny = x + dx, y + dy
                if (nx, ny) not in visited and 0 <= ny < h and 0 <= nx < w:
                    if skeleton[ny, nx] > 0:
                        neighbors.append((nx, ny))
        
        if not neighbors:
            break
        
        if len(neighbors) == 1:
            chosen = neighbors[0]
        else:
            # At junction: follow the straightest path
            if prev is not None:
                dx_dir, dy_dir = x - prev[0], y - prev[1]
                best_dot, chosen = -999, neighbors[0]
                for (nx, ny) in neighbors:
                    dot = (nx - x) * dx_dir + (ny - y) * dy_dir
                    if dot > best_dot:
                        best_dot = dot
                        chosen = (nx, ny)
            else:
                chosen = neighbors[0]
        
        path.append(chosen)
        visited.add(chosen)
        prev = current
        current = chosen
    
    return path


# =============================================================================
# STEP 3: TAIL/STEM DETECTION
# =============================================================================

def detect_stem(image: np.ndarray, min_stem_length_px: int = 12) -> Dict:
    """
    Detect stem/tail on a leaf image. Robust to color/lighting.
    
    Core idea: The stem is identified purely by GEOMETRY:
    - It's a narrow protrusion from the main leaf body
    - Width along the stem is much less than the leaf body width
    - The transition from stem to leaf body shows a sharp width increase
    """
    h, w = image.shape[:2]
    
    # --- Robust segmentation ---
    mask = segment_leaf_robust(image)
    
    # Find largest contour (the leaf)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _empty_result(mask, h, w)
    
    leaf_contour = max(contours, key=cv2.contourArea)
    leaf_area = cv2.contourArea(leaf_contour)
    
    # Clean mask from largest contour only
    clean_mask = np.zeros_like(mask)
    cv2.drawContours(clean_mask, [leaf_contour], -1, 255, -1)
    
    # --- Distance transform: local width at every point ---
    dist_transform = cv2.distanceTransform(clean_mask, cv2.DIST_L2, 5)
    
    # --- Skeleton ---
    skeleton = compute_skeleton(clean_mask)
    
    # Width at each skeleton pixel = 2 * distance_to_edge
    width_map = np.zeros_like(skeleton, dtype=np.float64)
    skel_ys, skel_xs = np.where(skeleton > 0)
    for (sy, sx) in zip(skel_ys, skel_xs):
        width_map[sy, sx] = dist_transform[sy, sx] * 2.0
    
    skel_widths = width_map[skeleton > 0]
    if len(skel_widths) < 5:
        return _empty_result(clean_mask, h, w, skeleton=skeleton,
                            dist_transform=dist_transform, width_map=width_map,
                            leaf_area=leaf_area, leaf_contour=leaf_contour)
    
    # --- Endpoints ---
    endpoints = get_endpoints(skeleton)
    if not endpoints:
        return _empty_result(clean_mask, h, w, skeleton=skeleton,
                            dist_transform=dist_transform, width_map=width_map,
                            endpoints=[], leaf_area=leaf_area, leaf_contour=leaf_contour)
    
    # --- Width statistics ---
    # Use p75 as robust leaf body width estimate
    # (the median can be biased low if stem pixels are a large fraction of skeleton)
    leaf_body_width = np.percentile(skel_widths, 75)
    leaf_max_width = np.max(skel_widths)
    
    # --- For each endpoint, trace inward and look for the stem->body transition ---
    best_stem = None
    best_score = 0
    
    for ep in endpoints:
        path = trace_from_endpoint(skeleton, ep)
        if len(path) < 5:
            continue
        
        # Get raw width profile along this path
        widths_raw = np.array([width_map[py, px] for (px, py) in path])
        
        # Smooth to reduce noise (adaptive kernel size)
        ks = min(9, max(3, len(widths_raw) // 5))
        if ks % 2 == 0:
            ks += 1
        if len(widths_raw) >= ks:
            widths = np.convolve(widths_raw, np.ones(ks) / ks, mode='same')
        else:
            widths = widths_raw.copy()
        
        start_width = widths[0]
        path_max = np.max(widths)
        
        # --- Filter: endpoint should be in a narrow part ---
        # Skip if this endpoint is already in the fat leaf body
        if start_width > leaf_body_width * 0.5:
            continue
        
        # --- Find stem-to-body transition ---
        # The stem region: width stays below a threshold derived from leaf body
        stem_width_limit = leaf_body_width * 0.30
        stem_width_limit = max(stem_width_limit, start_width * 2.5)
        stem_width_limit = max(stem_width_limit, 15.0)
        
        transition_idx = None
        for i in range(len(widths)):
            if widths[i] > stem_width_limit:
                transition_idx = i
                break
        
        # Fallback: gradient-based detection
        if transition_idx is None and len(widths) > 10:
            gradient = np.diff(widths)
            for i in range(3, len(gradient) - 3):
                local_grad = np.mean(gradient[max(0, i-3):i+4])
                if local_grad > start_width * 0.25 and widths[i] > start_width * 1.8:
                    if i >= min_stem_length_px:
                        transition_idx = i
                    break
        
        if transition_idx is None or transition_idx < min_stem_length_px:
            continue
        
        # --- Compute stem metrics ---
        stem_pixels = path[:transition_idx]
        stem_widths = widths_raw[:transition_idx]
        avg_stem_width = float(np.mean(stem_widths))
        
        # Euclidean length of the stem path
        if len(stem_pixels) > 1:
            pts = np.array(stem_pixels, dtype=np.float64)
            diffs = np.diff(pts, axis=0)
            eucl_length = float(np.sum(np.sqrt(np.sum(diffs ** 2, axis=1))))
        else:
            eucl_length = 0.0
        
        # Score: longer stem + narrower relative width = better
        score = eucl_length / (avg_stem_width + 1.0)
        
        if score > best_score:
            best_score = score
            best_stem = {
                "endpoint": ep,
                "path": stem_pixels,
                "length_px": len(stem_pixels),
                "euclidean_length": eucl_length,
                "avg_width": avg_stem_width,
                "start_width": float(start_width),
                "score": score
            }
    
    # --- Validation ---
    has_stem = False
    if best_stem is not None:
        long_enough = best_stem["euclidean_length"] >= min_stem_length_px
        narrow_enough = best_stem["avg_width"] < leaf_body_width * 0.5
        elongated = best_stem["euclidean_length"] / (best_stem["avg_width"] + 1) > 2.0
        
        # --- Flat-region check: reject pointed leaf tips ---
        # A real stem has a long section where width is roughly constant.
        # A pointed leaf tip just keeps widening — no flat plateau.
        # Check: the longest flat region must be >= 50% of the stem path.
        has_flat_region = False
        if len(best_stem["path"]) > 10:
            sw = np.array([width_map[py, px] for (px, py) in best_stem["path"]])
            n_sw = len(sw)
            ks_sw = max(3, n_sw // 10)
            if ks_sw % 2 == 0: ks_sw += 1
            smooth_sw = np.convolve(sw, np.ones(ks_sw)/ks_sw, mode='same')
            
            best_flat = 0
            win_sw = max(5, n_sw // 8)
            for i_sw in range(0, n_sw - win_sw):
                seg = smooth_sw[i_sw:i_sw + win_sw]
                seg_mean = np.mean(seg)
                if seg_mean > 0 and (np.max(seg) - np.min(seg)) / seg_mean < 0.15:
                    flat_end = i_sw + win_sw
                    while flat_end < n_sw:
                        ext = smooth_sw[i_sw:flat_end + 1]
                        if (np.max(ext) - np.min(ext)) / np.mean(ext) < 0.20:
                            flat_end += 1
                        else:
                            break
                    best_flat = max(best_flat, flat_end - i_sw)
            
            # A real stem must have a flat region of at least 30 pixels
            # (~0.5cm at 0.0176 cm/px). Pointed leaf tips are shorter.
            has_flat_region = best_flat >= 30
        
        has_stem = long_enough and narrow_enough and elongated and has_flat_region
    
    # --- Refine stem boundary: trim blade taper ---
    # The stem detection threshold can be generous, including the gradually
    # widening blade base as "stem". Fix: walk backwards from the junction
    # and find where width drops back to true stem width.
    # True stem width = 25th percentile of widths in the middle portion of path.
    if has_stem and best_stem and len(best_stem["path"]) > 20:
        stem_widths_along = np.array([
            width_map[py, px] for (px, py) in best_stem["path"]
        ])
        n = len(stem_widths_along)
        
        # Typical stem width: p25 of middle third of the path
        q_start = n // 4
        q_end = 3 * n // 4
        if q_end > q_start:
            stem_typical_width = np.percentile(stem_widths_along[q_start:q_end], 25)
        else:
            stem_typical_width = np.median(stem_widths_along)
        
        # Walk backwards from junction: find where width drops to <= 1.5x typical
        refined_end = n - 1
        for i in range(n - 1, 0, -1):
            if stem_widths_along[i] <= stem_typical_width * 1.5:
                refined_end = i
                break
        
        # Only trim if we're cutting less than 30% (avoid over-trimming)
        if refined_end >= n * 0.7:
            trimmed_path = best_stem["path"][:refined_end + 1]
        else:
            # Fallback: keep original (trimming too aggressive)
            trimmed_path = best_stem["path"]
        
        # Recompute length for trimmed path
        if len(trimmed_path) > 1:
            pts = np.array(trimmed_path, dtype=np.float64)
            best_stem["euclidean_length"] = float(
                np.sum(np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1)))
            )
            best_stem["path"] = trimmed_path
            best_stem["length_px"] = len(trimmed_path)
            best_stem["avg_width"] = float(np.mean([
                width_map[py, px] for (px, py) in trimmed_path
            ]))
    
    # --- Build stem mask for visualization ---
    stem_mask = np.zeros_like(clean_mask)
    if has_stem and best_stem:
        for (px, py) in best_stem["path"]:
            radius = max(int(dist_transform[py, px]) + 2, 3)
            cv2.circle(stem_mask, (px, py), radius, 255, -1)
        stem_mask = cv2.bitwise_and(stem_mask, clean_mask)
    
    # --- Classify stem length ---
    stem_class = "NONE"
    if has_stem:
        leaf_dim = max(h, w)
        ratio = best_stem["euclidean_length"] / leaf_dim
        if ratio > 0.3:
            stem_class = "LONG"
        elif ratio > 0.1:
            stem_class = "MEDIUM"
        else:
            stem_class = "SHORT"
    
    return {
        "has_stem": has_stem,
        "stem_class": stem_class,
        "stem_length_px": best_stem["length_px"] if best_stem else 0,
        "stem_euclidean_length": best_stem["euclidean_length"] if has_stem and best_stem else 0,
        "stem_avg_width": best_stem["avg_width"] if best_stem else 0,
        "stem_start_width": best_stem["start_width"] if best_stem else 0,
        "leaf_body_width_p75": float(leaf_body_width),
        "leaf_median_width": float(np.median(skel_widths)),
        "leaf_area": leaf_area,
        "mask": clean_mask,
        "skeleton": skeleton,
        "width_map": width_map,
        "dist_transform": dist_transform,
        "stem_mask": stem_mask,
        "stem_path": best_stem["path"] if best_stem else [],
        "stem_endpoint": best_stem["endpoint"] if best_stem else None,
        "endpoints": endpoints,
        "stem_score": best_score
    }


def _empty_result(mask, h, w, **kwargs):
    """Return a result dict for cases where no stem can be detected."""
    base = {
        "has_stem": False, "stem_class": "NONE",
        "stem_length_px": 0, "stem_euclidean_length": 0,
        "stem_avg_width": 0, "stem_start_width": 0,
        "leaf_body_width_p75": 0, "leaf_median_width": 0,
        "leaf_area": 0, "mask": mask,
        "skeleton": kwargs.get("skeleton", np.zeros((h, w), dtype=np.uint8)),
        "width_map": kwargs.get("width_map", np.zeros((h, w), dtype=np.float64)),
        "dist_transform": kwargs.get("dist_transform", np.zeros((h, w), dtype=np.float64)),
        "stem_mask": np.zeros((h, w), dtype=np.uint8),
        "stem_path": [], "stem_endpoint": None,
        "endpoints": kwargs.get("endpoints", []),
        "stem_score": 0
    }
    base.update({k: v for k, v in kwargs.items() if k not in base})
    return base


# =============================================================================
# STEP 4: MEASUREMENT (cm)
# =============================================================================

def _measure_blade(contour_pts, mask, detection_result, h_img, w_img):
    """
    Measure blade length and width using contour geometry + white background.
    
    For leaves WITH stem:
      - Junction = where stem meets blade
      - Blade tip = contour point farthest from junction that is NOT in stem
        region and HAS white background beyond it (outward ray check)
      - Length = junction to blade tip
      - Width = perpendicular span of contour along junction->tip axis
    
    For leaves WITHOUT stem:
      - Find the two farthest-apart contour points
      - Length = that distance, Width = perpendicular span
    """
    if detection_result["has_stem"] and detection_result["stem_path"]:
        stem_mask = detection_result["stem_mask"]
        junction = detection_result["stem_path"][-1]
        jx, jy = float(junction[0]), float(junction[1])
        
        # Leaf centroid for outward ray direction
        M = cv2.moments(mask)
        if M["m00"] > 0:
            leaf_cx, leaf_cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        else:
            leaf_cx, leaf_cy = w_img / 2.0, h_img / 2.0
        
        # Distance from junction for each contour point
        dists = np.sqrt(
            (contour_pts[:, 0] - jx) ** 2 + (contour_pts[:, 1] - jy) ** 2
        )
        
        # Exclude points in the stem region
        not_stem = np.array([
            stem_mask[min(int(py), h_img - 1), min(int(px), w_img - 1)] == 0
            for px, py in contour_pts
        ])
        
        # Check white pixels beyond each contour point (outward ray)
        ray_len = 30
        white_beyond = np.zeros(len(contour_pts))
        for i, (px, py) in enumerate(contour_pts):
            dx = float(px) - leaf_cx
            dy = float(py) - leaf_cy
            norm = np.sqrt(dx * dx + dy * dy)
            if norm < 1:
                continue
            dx, dy = dx / norm, dy / norm
            count = 0
            for d in range(1, ray_len + 1):
                rx, ry = int(px + d * dx), int(py + d * dy)
                if 0 <= rx < w_img and 0 <= ry < h_img:
                    if mask[ry, rx] == 0:
                        count += 1
                else:
                    count += 1
            white_beyond[i] = count
        
        # Score: distance from junction, excluding stem, prefer points with white beyond
        score = dists.copy()
        score[~not_stem] = 0
        score[white_beyond < ray_len * 0.5] *= 0.5
        
        blade_tip_idx = np.argmax(score)
        btx = float(contour_pts[blade_tip_idx, 0])
        bty = float(contour_pts[blade_tip_idx, 1])
        
        blade_length_px = float(np.sqrt((btx - jx) ** 2 + (bty - jy) ** 2))
        
        # Width perpendicular to junction -> tip axis
        blade_axis = np.array([btx - jx, bty - jy])
        axis_len = np.linalg.norm(blade_axis)
        if axis_len > 0:
            axis_unit = blade_axis / axis_len
            perp_unit = np.array([-axis_unit[1], axis_unit[0]])
            centered = contour_pts.astype(np.float64) - np.array([jx, jy])
            perp_proj = centered @ perp_unit
            blade_width_px = float(np.max(perp_proj) - np.min(perp_proj))
        else:
            blade_width_px = 0.0
    
    else:
        # No stem: farthest-apart pair on convex hull
        hull = cv2.convexHull(
            contour_pts.reshape(-1, 1, 2).astype(np.int32)
        ).reshape(-1, 2)
        max_dist = 0.0
        pt_a, pt_b = hull[0], hull[0]
        for i in range(len(hull)):
            for j in range(i + 1, len(hull)):
                d = float(np.sqrt(
                    (hull[i][0] - hull[j][0]) ** 2 + (hull[i][1] - hull[j][1]) ** 2
                ))
                if d > max_dist:
                    max_dist = d
                    pt_a, pt_b = hull[i], hull[j]
        
        blade_length_px = max_dist
        
        blade_axis = pt_b.astype(np.float64) - pt_a.astype(np.float64)
        axis_len = np.linalg.norm(blade_axis)
        if axis_len > 0:
            axis_unit = blade_axis / axis_len
            perp_unit = np.array([-axis_unit[1], axis_unit[0]])
            centered = contour_pts.astype(np.float64) - pt_a.astype(np.float64)
            perp_proj = centered @ perp_unit
            blade_width_px = float(np.max(perp_proj) - np.min(perp_proj))
        else:
            blade_width_px = 0.0
    
    # Always ensure length >= width
    if blade_width_px > blade_length_px:
        blade_length_px, blade_width_px = blade_width_px, blade_length_px
    
    return blade_length_px, blade_width_px


def measure_leaf(detection_result: Dict, cm_per_pixel: float = 0.0176) -> Dict:
    """
    Measure leaf and stem sizes in centimeters.
    
    Blade length logic:
    - Find the stem-to-blade junction point (where stem ends, blade begins)
    - Find the farthest point on the leaf contour from that junction = blade tip
    - Blade length = euclidean distance from junction to blade tip
    
    For leaves without stem:
    - Find the two farthest-apart contour points (diameter of the contour)
    - Longer axis = blade length, perpendicular = blade width
    
    Parameters:
    -----------
    detection_result : Dict returned by detect_stem()
    cm_per_pixel     : Calibration factor (default 0.0176 for 1080p source)
    """
    skeleton = detection_result["skeleton"]
    endpoints = detection_result["endpoints"]
    mask = detection_result["mask"]
    
    # --- Total leaf length: longest skeleton path from any endpoint ---
    total_length_px = 0.0
    for ep in endpoints:
        path = trace_from_endpoint(skeleton, ep)
        if len(path) > 1:
            pts = np.array(path, dtype=np.float64)
            path_len = float(np.sum(np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))))
            total_length_px = max(total_length_px, path_len)
    
    # --- Stem length ---
    stem_length_px = detection_result["stem_euclidean_length"]
    
    # --- Find the leaf contour ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _make_measurements(0, 0, 0, 0, 0, 0, cm_per_pixel)
    
    leaf_contour = max(contours, key=cv2.contourArea)
    contour_pts = leaf_contour.reshape(-1, 2)  # (N, 2) array of (x, y)
    h_img, w_img = mask.shape
    
    # --- Blade length & width ---
    # Uses the white background to confirm the blade tip:
    #   1. Find stem-blade junction
    #   2. For each contour point: score = distance_from_junction,
    #      but exclude stem region and penalise points without white beyond them
    #   3. Blade tip = highest scoring point
    #   4. Length = junction -> tip, Width = perpendicular span
    
    blade_length_px, blade_width_px = _measure_blade(
        contour_pts, mask, detection_result, h_img, w_img
    )
    
    # --- Leaf area ---
    leaf_area_px2 = float(np.count_nonzero(mask))
    
    return _make_measurements(total_length_px, blade_length_px, stem_length_px,
                              blade_width_px, leaf_area_px2, cm_per_pixel)


def _make_measurements(total_px, blade_px, stem_px, width_px, area_px2, cm_per_pixel):
    """Convert px measurements to cm and return dict."""
    return {
        "total_length_cm":    round(total_px * cm_per_pixel, 2),
        "blade_length_cm":    round(blade_px * cm_per_pixel, 2),
        "stem_length_cm":     round(stem_px * cm_per_pixel, 2),
        "blade_width_cm":     round(width_px * cm_per_pixel, 2),
        "leaf_area_cm2":      round(area_px2 * cm_per_pixel * cm_per_pixel, 4),
        "total_length_px":    round(total_px, 1),
        "blade_length_px":    round(blade_px, 1),
        "stem_length_px":     round(stem_px, 1),
        "blade_width_px":     round(width_px, 1),
        "leaf_area_px2":      round(area_px2, 0),
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_result(image: np.ndarray, result: Dict, measurements: Dict, title: str = "") -> np.ndarray:
    """Create 4-panel visualization: original, detection, width map, info (in cm)."""
    h, w = image.shape[:2]
    target_h = 400
    scale = target_h / h
    new_w = int(w * scale)
    new_h = target_h
    panel_w = max(new_w, 200)
    
    def resize(img):
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    def pad_panel(panel):
        if panel.shape[1] < panel_w:
            pad = np.full((new_h, panel_w - panel.shape[1], 3), 40, dtype=np.uint8)
            return np.hstack([panel, pad])
        return panel
    
    image_r = resize(image)
    
    # --- Panel 1: Detection overlay ---
    panel1 = image_r.copy()
    mask_r = resize(result["mask"])
    skeleton_r = resize(result["skeleton"])
    stem_mask_r = resize(result["stem_mask"])
    
    # Leaf contour
    cnts, _ = cv2.findContours(mask_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(panel1, cnts, -1, (255, 200, 0), 2)
    
    # Skeleton
    skel_pts = np.argwhere(skeleton_r > 0)
    for (y, x) in skel_pts:
        if 0 <= y < new_h and 0 <= x < new_w:
            panel1[y, x] = (200, 255, 255)
    
    # Stem overlay
    if result["has_stem"]:
        overlay = np.zeros_like(panel1)
        overlay[stem_mask_r > 0] = (0, 0, 255)
        panel1 = cv2.addWeighted(panel1, 0.7, overlay, 0.3, 0)
        # Re-draw skeleton in stem region
        for (y, x) in skel_pts:
            if 0 <= y < new_h and 0 <= x < new_w and stem_mask_r[y, x] > 0:
                panel1[y, x] = (0, 0, 255)
    
    # Endpoints
    for ep in result.get("endpoints", []):
        ex, ey = int(ep[0] * scale), int(ep[1] * scale)
        cv2.circle(panel1, (ex, ey), 5, (0, 255, 0), -1)
    
    # Stem tip marker
    if result.get("stem_endpoint"):
        sx = int(result["stem_endpoint"][0] * scale)
        sy = int(result["stem_endpoint"][1] * scale)
        cv2.circle(panel1, (sx, sy), 8, (0, 0, 255), 3)
    
    # --- Panel 2: Width heatmap ---
    panel2 = np.full((new_h, new_w, 3), 40, dtype=np.uint8)
    dt = result["dist_transform"]
    if dt.max() > 0:
        dt_norm = cv2.normalize(dt, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        dt_color = cv2.applyColorMap(dt_norm, cv2.COLORMAP_JET)
        dt_r = resize(dt_color)
        panel2[mask_r > 0] = dt_r[mask_r > 0]
    
    # --- Panel 3: Segmentation mask (shows robustness) ---
    panel3 = np.full((new_h, new_w, 3), 40, dtype=np.uint8)
    mask_3ch = cv2.cvtColor(mask_r, cv2.COLOR_GRAY2BGR)
    # Color code: green for leaf body, red for stem
    leaf_vis = np.zeros_like(panel3)
    leaf_vis[mask_r > 0] = (0, 180, 0)
    if result["has_stem"]:
        leaf_vis[stem_mask_r > 0] = (0, 80, 255)
    panel3 = leaf_vis
    
    # --- Panel 4: Info text ---
    info_w = 300
    panel4 = np.full((new_h, info_w, 3), 40, dtype=np.uint8)
    
    lines = []
    if title:
        lines.append((title, (100, 220, 255), 0.55))
    lines.append(("", None, 0))
    
    if result["has_stem"]:
        lines.append(("STEM DETECTED", (0, 255, 100), 0.6))
        lines.append((f"Class: {result['stem_class']} STEM", 
                      {"LONG": (0, 160, 255), "MEDIUM": (0, 255, 255), "SHORT": (100, 255, 100)}
                      .get(result["stem_class"], (255, 255, 255)), 0.5))
    else:
        lines.append(("NO STEM", (100, 100, 255), 0.6))
    
    lines.append(("", None, 0))
    lines.append(("--- Measurements ---", (180, 180, 180), 0.45))
    lines.append((f"Total length:  {measurements['total_length_cm']:.2f} cm", (255, 255, 255), 0.45))
    lines.append((f"Blade length:  {measurements['blade_length_cm']:.2f} cm", (255, 255, 255), 0.45))
    lines.append((f"Stem length:   {measurements['stem_length_cm']:.2f} cm", (255, 255, 255), 0.45))
    lines.append((f"Blade width:   {measurements['blade_width_cm']:.2f} cm", (255, 255, 255), 0.45))
    lines.append((f"Leaf area:     {measurements['leaf_area_cm2']:.3f} cm2", (255, 255, 255), 0.45))
    
    y_off = 28
    for text, color, font_scale in lines:
        if text and color:
            cv2.putText(panel4, text, (12, y_off), cv2.FONT_HERSHEY_SIMPLEX,
                       font_scale, color, 1, cv2.LINE_AA)
        y_off += int(24 * max(font_scale / 0.45, 1)) if font_scale > 0 else 12
    
    # Labels
    for p, lbl in [(panel1, "Detection"), (panel2, "Width Map"), (panel3, "Segmentation")]:
        cv2.rectangle(p, (0, 0), (len(lbl) * 10 + 8, 22), (20, 20, 20), -1)
        cv2.putText(p, lbl, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Combine
    panels = [pad_panel(p) for p in [panel1, panel2, panel3]]
    combined = np.hstack(panels + [panel4])
    
    # Vertical separators
    xpos = panel_w
    for _ in range(3):
        cv2.line(combined, (xpos, 0), (xpos, new_h), (80, 80, 80), 1)
        xpos += panel_w if _ < 2 else info_w
    
    return combined


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def process_images(image_dir: str):
    """Process all leaf images in a directory.
    Automatically creates an 'output' folder inside image_dir and saves results there.
    """
    image_dir = os.path.abspath(image_dir)
    output_dir = os.path.join(image_dir, "output")

    if not os.path.isdir(image_dir):
        print(f"ERROR: Directory not found: {image_dir}")
        return []

    os.makedirs(output_dir, exist_ok=True)

    files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
        and os.path.isfile(os.path.join(image_dir, f))
    ])

    if not files:
        print(f"No image files found in {image_dir}")
        return []

    print(f"Input:  {image_dir}  ({len(files)} images)")
    print(f"Output: {output_dir}")

    vis_panels = []
    results_list = []

    for fname in files:
        filepath = os.path.join(image_dir, fname)
        image = cv2.imread(filepath)
        if image is None:
            print(f"  SKIP: cannot read {fname}")
            continue

        print(f"\nProcessing: {fname} ({image.shape[1]}x{image.shape[0]})")
        result = detect_stem(image)
        measurements = measure_leaf(result, CM_PER_PIXEL)

        if result["has_stem"]:
            print(f"  -> STEM DETECTED ({result['stem_class']})")
            print(f"     Stem:  {measurements['stem_length_cm']:.2f} cm")
        else:
            print(f"  -> NO STEM")
        print(f"     Total: {measurements['total_length_cm']:.2f} cm  |  "
              f"Blade: {measurements['blade_length_cm']:.2f} x {measurements['blade_width_cm']:.2f} cm")

        vis = visualize_result(image, result, measurements, fname)
        vis_panels.append(vis)
        cv2.imwrite(os.path.join(output_dir, f"result_{fname}"), vis)
        results_list.append((fname, result, measurements))

    # Summary image
    if vis_panels:
        max_w = max(p.shape[1] for p in vis_panels)
        rows = []
        for p in vis_panels:
            if p.shape[1] < max_w:
                pad = np.full((p.shape[0], max_w - p.shape[1], 3), 40, dtype=np.uint8)
                p = np.hstack([p, pad])
            rows.append(p)
            rows.append(np.full((3, max_w, 3), 60, dtype=np.uint8))

        body = np.vstack(rows[:-1])

        title_h = 50
        title_bar = np.full((title_h, max_w, 3), 25, dtype=np.uint8)
        cv2.putText(title_bar, "Leaf Stem Detection - Robust (Color/Lighting Independent)",
                   (12, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        summary = np.vstack([title_bar, body])
        summary_path = os.path.join(output_dir, "summary_all_results.png")
        cv2.imwrite(summary_path, summary)
        print(f"\nSummary -> {summary_path}")

    return results_list


# =============================================================================
# CONFIG
# =============================================================================

INPUT_FOLDER  = "./sample"        # folder with leaf images
CM_PER_PIXEL  = 0.0176            # calibration: 1080p source, 0.0176 cm per pixel

# MODE:
#   "stem_only"  - detect stem, measure ONLY leaves that have a stem (skip stemless)
#   "both"       - detect stem, measure ALL leaves (with or without stem)
MODE = "both"

# =============================================================================

if __name__ == "__main__":
    results = process_images(INPUT_FOLDER)

    # Filter based on MODE
    if MODE == "stem_only":
        results = [(f, r, m) for f, r, m in results if r["has_stem"]]
        print(f"\n  MODE: stem_only (showing only leaves with stem)")
    else:
        print(f"\n  MODE: both (showing all leaves)")

    print("=" * 92)
    print("  RESULTS SUMMARY")
    print("=" * 92)
    print(f"  {'File':25s}  {'Stem':8s}  {'Total':>8s}  {'Blade L':>8s}  {'Blade W':>8s}  {'Stem':>8s}  {'Area':>10s}")
    print(f"  {'':25s}  {'':8s}  {'(cm)':>8s}  {'(cm)':>8s}  {'(cm)':>8s}  {'(cm)':>8s}  {'(cm2)':>10s}")
    print("-" * 92)
    for fname, r, m in results:
        stem_str = r['stem_class'] if r['has_stem'] else "NONE"
        print(f"  {fname:25s}  {stem_str:8s}  "
              f"{m['total_length_cm']:8.2f}  "
              f"{m['blade_length_cm']:8.2f}  "
              f"{m['blade_width_cm']:8.2f}  "
              f"{m['stem_length_cm']:8.2f}  "
              f"{m['leaf_area_cm2']:10.3f}")
    print("=" * 92)