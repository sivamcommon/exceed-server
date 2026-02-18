import cv2
import numpy as np


def _is_square(approx, aspect_tol=0.15):
    x, y, w, h = cv2.boundingRect(approx)
    if w == 0 or h == 0:
        return False, (x, y, w, h)
    aspect = w / float(h)
    return abs(aspect - 1.0) <= aspect_tol, (x, y, w, h)


def _is_red_region(image_bgr, rect, min_red_ratio=0.25):
    x, y, w, h = rect
    roi = image_bgr[y:y + h, x:x + w]
    if roi.size == 0:
        return False

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Full red range (low S/V thresholds to catch darker reds)
    lower1 = np.array([0, 40, 40], dtype=np.uint8)
    upper1 = np.array([10, 255, 255], dtype=np.uint8)
    lower2 = np.array([170, 40, 40], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    red_ratio = float(np.count_nonzero(mask)) / float(mask.size)
    return red_ratio >= min_red_ratio


def detect_red_squares(image_bgr, min_area=500, aspect_tol=0.15, min_red_ratio=0.25):
    """
    Detect squares first by shape, then verify they are red.

    Inputs:
    - image_bgr: numpy array (H, W, 3) in BGR format.
    - min_area: minimum contour area to consider.
    - aspect_tol: allowed deviation from square aspect ratio (w/h).
    - min_red_ratio: fraction of red pixels required inside the square.

    Outputs:
    - annotated: image with detected squares drawn and size labels.
    - rects: list of dicts with keys {"x","y","w","h"} in pixels.
    """
    if image_bgr is None:
        raise ValueError("image_bgr is None")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    annotated = image_bgr.copy()
    rects = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) != 4:
            continue

        is_square, rect = _is_square(approx, aspect_tol=aspect_tol)
        if not is_square:
            continue

        if not _is_red_region(image_bgr, rect, min_red_ratio=min_red_ratio):
            continue

        x, y, w, h = rect
        rects.append({"x": x, "y": y, "w": w, "h": h})

        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"w={w}px h={h}px"
        cv2.putText(
            annotated,
            label,
            (x, max(0, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return annotated, rects
