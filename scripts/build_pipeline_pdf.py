"""Render Mermaid diagrams via mermaid.ink and combine into a PDF."""
import base64
import io
import os
import sys
import time
import requests
from PIL import Image
from fpdf import FPDF

OUT_PDF = os.path.join(os.path.dirname(__file__), "SERVER_PIPELINE_DIAGRAMS.pdf")
IMG_DIR = os.path.join(os.path.dirname(__file__), "_diagram_cache")
os.makedirs(IMG_DIR, exist_ok=True)

def _ascii(s: str) -> str:
    repl = {
        "\u2014": "-", "\u2013": "-", "\u2212": "-",
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2026": "...", "\u00b7": "-", "\u00d7": "x",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return s


DIAGRAMS = [
    {
        "title": "1. Overall Architecture - Camera Source",
        "desc": (
            "Camera frames flow from nvarguscamerasrc through a GStreamer pipeline "
            "into a single-slot 'latest_raw' buffer. The grab loop, inference worker, "
            "streamer, and recorder are independent threads communicating via single-slot "
            "overwrites — this is the foundation of the automatic frame-skipping."
        ),
        "mermaid": r"""flowchart LR
    subgraph HW["Camera Hardware (30 FPS)"]
        CAM[nvarguscamerasrc<br/>1080p NV12]
    end
    subgraph GST["GStreamer Pipeline"]
        Q["queue<br/>max-buffers=1<br/>leaky=downstream"]
        CONV[nvvidconv + videoconvert<br/>to BGR]
        SINK["appsink<br/>max-buffers=1<br/>drop=true"]
    end
    subgraph SHARED["SharedState per camera"]
        RAW[("latest_raw<br/>single slot")]
        OUT[("latest<br/>single slot")]
    end
    subgraph THREADS["Worker Threads"]
        GRAB[pipeline_loop<br/>30 FPS grab]
        INFER[_infer_worker<br/>GPU-bound, no cap]
        STREAM[mjpeg_generator<br/>20 FPS]
        REC[FFmpeg stdin<br/>H.264 encode]
    end
    CAM --> Q --> CONV --> SINK
    SINK -->|pull-sample| GRAB
    GRAB -->|every frame| RAW
    GRAB -->|if raw mode copy| OUT
    RAW -->|read latest| INFER
    INFER -->|annotated| OUT
    OUT --> STREAM
    OUT --> REC
    STREAM -->|JPEG 80| CLIENT[Browser MJPEG]
    REC --> MP4[(mp4 file)]
""",
    },
    {
        "title": "2. Grabber vs Inference Worker - How Skipping Happens",
        "desc": (
            "Sequence diagram showing a slow inference (~100ms per frame). The grab loop "
            "keeps overwriting latest_raw at 30 FPS. When the infer worker finishes a "
            "frame, it picks up whatever is currently in latest_raw — never a backlog. "
            "Frames 2 and 3 here are silently dropped."
        ),
        "mermaid": r"""sequenceDiagram
    participant CAM as Camera 30 FPS
    participant GRAB as Grab Loop
    participant RAW as latest_raw slot
    participant INF as Infer Worker
    participant OUT as latest slot
    participant STR as Stream 20 FPS
    Note over GRAB,INF: Inference takes ~100 ms
    CAM->>GRAB: frame 1 (t=0ms)
    GRAB->>RAW: store 1
    INF->>RAW: read 1
    Note right of INF: runs YOLO<br/>for 100ms
    CAM->>GRAB: frame 2 (t=33ms)
    GRAB->>RAW: overwrite (skip 1)
    CAM->>GRAB: frame 3 (t=66ms)
    GRAB->>RAW: overwrite (skip 2)
    CAM->>GRAB: frame 4 (t=99ms)
    GRAB->>RAW: overwrite (skip 3)
    INF->>OUT: annotated 1 (t=100ms)
    INF->>RAW: read 4 (newest)
    Note right of INF: skips 2 and 3
    STR->>OUT: read every 50ms
    STR-->>STR: encode JPEG, send
""",
    },
    {
        "title": "3. Mode and Source State Machine",
        "desc": (
            "The server has four logical states, controlled by /stream/mode and /record/*. "
            "Transitions into video mode stop the Argus camera to free NVMM buffers; "
            "returning from video re-starts Argus. stable_runtime_mode=true forces camera."
        ),
        "mermaid": r"""stateDiagram-v2
    [*] --> RawCamera: boot
    RawCamera: RAW + camera<br/>live stream<br/>no inference
    InferCamera: INFER + camera<br/>YOLO on live<br/>recording MP4
    InferVideo: INFER + video file<br/>video_pipeline_loop<br/>speed 0-10
    RawCamera --> InferCamera: mode=infer source=camera
    InferCamera --> RawCamera: mode=raw or record/stop
    RawCamera --> InferVideo: mode=infer source=video<br/>stops Argus
    InferVideo --> RawCamera: mode=raw or video EOF<br/>restarts Argus
    InferCamera --> InferVideo: source=video
    InferVideo --> InferCamera: source=camera
""",
    },
    {
        "title": "4. Video Source Path (Different from Camera Path)",
        "desc": (
            "When source=video, the camera pipeline is stopped. A single thread reads "
            "frames from the .mp4 file, runs inference, and updates latest — no grab/infer "
            "split. Playback speed is controlled by a sleep multiplier 0..10."
        ),
        "mermaid": r"""flowchart LR
    FILE[(mp4 file<br/>video_source_path)] --> VC[cv2.VideoCapture]
    VC -->|read one frame| VL[video_pipeline_loop<br/>single thread]
    VL -->|resize 1080p| RUN[run_inference<br/>YOLO main + defect]
    RUN -->|annotated| OUT[("latest slot")]
    OUT --> STR[MJPEG 20 FPS]
    OUT --> FF[FFmpeg record]
    VL -.->|sleep 1/INFER_FPS × 1+slow| VL
    VL -.->|read fails > 5| EOF[handle_video_eof<br/>persist stats<br/>reset mode]
""",
    },
    {
        "title": "5. Recording Paths - Which Frames Reach the MP4",
        "desc": (
            "Raw-mode recording gets every 30 FPS grab frame. Infer-mode recording gets "
            "only annotated frames at the inference rate, but FFmpeg is told -r 30, which "
            "is why infer-recorded videos can appear sped-up."
        ),
        "mermaid": r"""flowchart TB
    subgraph Mode1["Raw Recording - STREAM_MODE=raw"]
        G1[Grab loop 30 FPS] -->|every raw frame| F1[FFmpeg -r 30]
        F1 --> M1[(raw mp4 true 30 FPS)]
    end
    subgraph Mode2["Infer Recording - STREAM_MODE=infer"]
        G2[Grab loop 30 FPS] --> RAW2[latest_raw]
        RAW2 --> I2[Infer worker<br/>10-30 FPS]
        I2 -->|only annotated| F2[FFmpeg -r 30]
        F2 --> M2[(infer mp4<br/>labeled 30 FPS<br/>fewer real frames<br/>plays fast)]
    end
""",
    },
    {
        "title": "6. FPS Budget Summary",
        "desc": (
            "Only three stages have hard FPS caps: camera (30), stream (20), and video "
            "playback (configurable). Inference is best-effort GPU-bound. Every arrow "
            "between stages is single-slot, so slower consumers automatically drop frames."
        ),
        "mermaid": r"""flowchart LR
    A[Camera sensor<br/>30 FPS] --> B[Grab loop<br/>30 FPS fixed]
    B --> C{inferring?}
    C -->|no| D[latest slot<br/>30 FPS]
    C -->|yes| E[Infer worker<br/>GPU-bound<br/>10-25 FPS]
    E --> D
    D --> F[MJPEG stream<br/>20 FPS cap]
    D --> G[Recorder<br/>source rate]
""",
    },
]


def mermaid_to_png(mmd: str, out_path: str) -> str:
    """Render a mermaid diagram to PNG via mermaid.ink and save to out_path."""
    encoded = base64.urlsafe_b64encode(mmd.encode("utf-8")).decode("ascii")
    url = f"https://mermaid.ink/img/{encoded}?type=png&bgColor=!white"
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200 and r.content[:4] == b"\x89PNG":
                with open(out_path, "wb") as f:
                    f.write(r.content)
                return out_path
            print(f"  attempt {attempt+1}: status={r.status_code} len={len(r.content)}")
        except Exception as e:
            print(f"  attempt {attempt+1} exception: {e}")
        time.sleep(2)
    raise RuntimeError(f"Failed to render mermaid diagram to {out_path}")


def build_pdf():
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.multi_cell(0, 10, _ascii("Server2 - Processing, Streaming & Inference Pipeline"))
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(
        0, 6,
        _ascii(
            "Diagrams extracted from server.py covering how frames are captured, "
            "skipped, passed through inference, streamed via MJPEG, and recorded. "
            "Each diagram is followed by a short description."
        )
    )
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Contents", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    for d in DIAGRAMS:
        pdf.cell(0, 6, _ascii(f"- {d['title']}"), new_x="LMARGIN", new_y="NEXT")

    page_w = pdf.w - pdf.l_margin - pdf.r_margin

    for idx, d in enumerate(DIAGRAMS, start=1):
        print(f"[{idx}/{len(DIAGRAMS)}] {d['title']}")
        img_path = os.path.join(IMG_DIR, f"diagram_{idx}.png")
        if not (os.path.exists(img_path) and os.path.getsize(img_path) > 2048):
            mermaid_to_png(d["mermaid"], img_path)
        else:
            print("  cached")

        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.multi_cell(0, 8, _ascii(d["title"]))
        pdf.ln(1)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 5, _ascii(d["desc"]))
        pdf.ln(3)

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
    print(f"\nWrote {OUT_PDF}")


if __name__ == "__main__":
    try:
        build_pdf()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
