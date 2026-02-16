"""Drawing utilities for boxes and defects."""

import cv2

from .core import class_name


def draw_main_boxes(frame_bgr, xyxy, conf, cls_ids, names, keep_indices):
    if xyxy is None or len(keep_indices) == 0:
        return frame_bgr.copy()

    img = frame_bgr.copy()
    for i in keep_indices:
        if i < 0 or i >= len(xyxy):
            continue
        x1, y1, x2, y2 = xyxy[i]
        cname = class_name(names, int(cls_ids[i]))
        cname_l = str(cname).strip().lower()
        if cname_l in ("leaf", "spinach"):
            box_color = (0, 255, 0)
        elif cname_l in ("stem", "leaf stem"):
            box_color = (144, 238, 144)
        else:
            box_color = (0, 0, 255)
        label = f"{cname} {conf[i]:.2f}"
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
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


def draw_defects(annotated, defects):
    defect_found = False
    for d in defects:
        defect_found = True
        gx1, gy1, gx2, gy2 = d["box"]
        cname = d["name"]
        cname_l = str(cname).strip().lower()
        if "yellow" in cname_l:
            box_color = (0, 255, 255)
        elif "white" in cname_l:
            box_color = (255, 255, 255)
        elif "ipd" in cname_l:
            box_color = (0, 0, 255)
        else:
            box_color = (0, 0, 255)
        label = f"{cname} {d['conf']:.2f}"
        cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), box_color, 2)
        cv2.putText(
            annotated,
            label,
            (gx1, max(0, gy1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            box_color,
            1,
            cv2.LINE_AA,
        )
    return defect_found
