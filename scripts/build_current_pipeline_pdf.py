"""Build a current runtime pipeline PDF with diagrams."""

import base64
import os
import sys
import time

import requests
from fpdf import FPDF
from PIL import Image


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PDF = os.path.join(ROOT, "documents", "CURRENT_STREAMING_INFERENCE_PIPELINE.pdf")
IMG_DIR = os.path.join(ROOT, "documents", "diagram_cache_current")
os.makedirs(IMG_DIR, exist_ok=True)


def _ascii(s: str) -> str:
    repl = {
        "\u2014": "-",
        "\u2013": "-",
        "\u2212": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
        "\u00b7": "-",
        "\u00d7": "x",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return s


DIAGRAMS = [
    {
        "title": "1) High-Level Runtime Pipeline",
        "desc": (
            "Single-frame flow through the current runtime. Camera/video source provides a frame, "
            "inference runs when enabled, then the same output frame is used for MJPEG stream and recording."
        ),
        "mermaid": r"""flowchart LR
    SRC[Frame Source] --> DEC{source mode}
    DEC -->|camera| CAM[camera_pipeline_loop]
    DEC -->|video| VID[video_pipeline_loop]
    CAM --> RAW[shared.latest_raw]
    RAW --> INF[infer_worker_loop]
    VID --> RUN[run_inference]
    INF --> RUN
    RUN --> OUT[shared.latest]
    OUT --> STR[mjpeg_stream_generator]
    OUT --> REC[FFmpeg recorder_write]
""",
    },
    {
        "title": "2) Camera Pipeline Element (GStreamer -> BGR)",
        "desc": (
            "Live camera path on Jetson. The pipeline converts NVMM/NV12 camera frames to BGR "
            "for OpenCV and Python processing."
        ),
        "mermaid": r"""flowchart LR
    A[nvarguscamerasrc] --> B[video/x-raw memory:NVMM NV12 1080p]
    B --> C[queue max-buffers=1 leaky=downstream]
    C --> D[nvvidconv]
    D --> E[video/x-raw BGRx]
    E --> F[videoconvert]
    F --> G[video/x-raw BGR]
    G --> H[appsink max-buffers=1 drop=true]
    H --> I[grab_1080p]
    I --> J[camera_pipeline_loop]
""",
    },
    {
        "title": "3) Mode Matrix (stream mode + source mode)",
        "desc": (
            "Current behavior by mode combination. Raw/camera is lightweight viewing mode, "
            "infer/camera and infer/video execute model inference, while raw/video is not used as a normal session mode."
        ),
        "mermaid": r"""flowchart TB
    M1[raw + camera] --> O1[Live stream raw frames]
    M1 --> O2[Optional raw recording]
    M2[infer + camera] --> O3[Infer worker on latest_raw]
    M2 --> O4[Annotated stream + infer recording]
    M3[infer + video] --> O5[video_pipeline_loop reads mp4]
    M3 --> O6[Annotated stream + infer recording]
    M3 --> O7[EOF handler -> back to raw/camera]
    M4[raw + video] --> O8[Not normal runtime target]
""",
    },
    {
        "title": "4) Camera + Infer Mode Detailed Sequence",
        "desc": (
            "In infer/camera mode, grabbing and inference are decoupled. Grab loop keeps latest_raw fresh, "
            "inference worker consumes newest frame, and stream/record use shared.latest."
        ),
        "mermaid": r"""sequenceDiagram
    participant C as Camera
    participant G as camera_pipeline_loop
    participant R as shared.latest_raw
    participant W as infer_worker_loop
    participant I as run_inference
    participant O as shared.latest
    participant S as mjpeg_stream_generator
    participant F as FFmpeg
    C->>G: frame
    G->>R: overwrite latest_raw
    W->>R: read newest
    W->>I: run_inference(frame)
    I-->>W: annotated frame
    W->>O: update latest
    S->>O: read + JPEG encode
    O->>F: recorder_write(annotated)
""",
    },
    {
        "title": "5) Video + Infer Mode Detailed Sequence",
        "desc": (
            "In infer/video mode, one loop reads file frames, runs inference, publishes latest, writes recording, "
            "and on EOF triggers cleanup and mode reset."
        ),
        "mermaid": r"""sequenceDiagram
    participant V as cv2.VideoCapture
    participant L as video_pipeline_loop
    participant I as run_inference
    participant O as shared.latest
    participant S as MJPEG
    participant F as FFmpeg
    participant E as handle_video_eof
    L->>V: read frame
    V-->>L: frame or fail
    L->>I: run_inference(frame)
    I-->>L: annotated
    L->>O: update latest
    S->>O: stream jpeg
    O->>F: recorder_write(annotated)
    L->>L: apply speed slow level 0..10
    L->>E: on read failure/EOF
""",
    },
    {
        "title": "6) Inference Internals (Two-Stage Models)",
        "desc": (
            "run_inference delegates to infer_main two-stage logic. Main model detects/tracks leaves, "
            "defect model evaluates crops, then drawing/stats produce final annotated frame."
        ),
        "mermaid": r"""flowchart LR
    IN[Input frame] --> M1[Main model detect/segment + track]
    M1 --> F1[Leaf filtering + dedup + ROI filter]
    F1 --> C1[Per-leaf crop/mask extraction]
    C1 --> M2[Defect model on each crop]
    M2 --> V[Defect voting/aggregation]
    V --> D[Draw overlays + update stats]
    D --> OUT[Annotated frame + summary]
""",
    },
    {
        "title": "7) Output Consumers and State Buffers",
        "desc": (
            "Two shared buffers define output behavior. latest_raw is the newest source frame, latest is the rendered output "
            "consumed by stream and recorder."
        ),
        "mermaid": r"""flowchart LR
    SRC[Source frame] --> RAW[latest_raw buffer]
    RAW --> INF[Inference path]
    SRC -->|raw mode direct| OUT[latest output buffer]
    INF --> OUT
    OUT --> MJPEG[MJPEG stream clients]
    OUT --> REC[Recording writer]
    OUT --> CAP[Capture endpoint]
""",
    },
]


def mermaid_to_png(mmd: str, out_path: str) -> str:
    encoded = base64.urlsafe_b64encode(mmd.encode("utf-8")).decode("ascii")
    url = f"https://mermaid.ink/img/{encoded}?type=png&bgColor=!white"
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200 and r.content[:4] == b"\x89PNG":
                with open(out_path, "wb") as f:
                    f.write(r.content)
                return out_path
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"Failed to render diagram: {out_path}")


def build_pdf():
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.multi_cell(0, 10, _ascii("Current Streaming & Inference Pipeline"))
    pdf.ln(1)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(
        0,
        6,
        _ascii(
            "This document describes the current runtime behavior in all major modes, including "
            "camera pipeline stages, streaming, inference, recording, and EOF/mode transitions."
        ),
    )
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 8, "Contents")
    pdf.set_font("Helvetica", "", 10)
    for d in DIAGRAMS:
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 6, _ascii(f"- {d['title']}"))

    page_w = pdf.w - pdf.l_margin - pdf.r_margin

    for idx, d in enumerate(DIAGRAMS, start=1):
        img_path = os.path.join(IMG_DIR, f"current_diagram_{idx}.png")
        if not (os.path.exists(img_path) and os.path.getsize(img_path) > 2048):
            mermaid_to_png(d["mermaid"], img_path)

        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 8, _ascii(d["title"]))
        pdf.set_font("Helvetica", "", 10)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 5, _ascii(d["desc"]))
        pdf.ln(2)

        with Image.open(img_path) as img:
            iw, ih = img.size
        max_h_mm = pdf.h - pdf.get_y() - pdf.b_margin - 5
        img_w_mm = page_w
        img_h_mm = img_w_mm * ih / iw
        if img_h_mm > max_h_mm:
            img_h_mm = max_h_mm
            img_w_mm = img_h_mm * iw / ih
        x = (pdf.w - img_w_mm) / 2
        pdf.image(img_path, x=x, y=pdf.get_y(), w=img_w_mm, h=img_h_mm)

    pdf.output(OUT_PDF)
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    try:
        build_pdf()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
