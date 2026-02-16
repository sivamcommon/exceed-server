import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import os
import copy
import time
import json
import threading
import subprocess
import cv2
import numpy as np

import infer_2model as infer

Gst.init(None)

app = FastAPI()

# Configure CORS - add your frontend origin(s) here
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve saved media files from /media
app.mount("/media", StaticFiles(directory="media"), name="media")

capture_dir = "./media/images"
video_dir = "./media/videos"
os.makedirs(capture_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)

CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080
STREAM_WIDTH = 1280
STREAM_HEIGHT = 720

SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "camera_settings.json")
INFER_STATE_PATH = os.path.join(os.path.dirname(__file__), "infer_state.json")
DEFAULT_SETTINGS = {
    "wbmode": 5,
    "exposure_us": 20000000,
    "gain": 1.0,
    "aelock": True,
    "camera_mode": "single",
}

def load_settings():
    if not os.path.exists(SETTINGS_PATH):
        return DEFAULT_SETTINGS.copy()
    try:
        with open(SETTINGS_PATH, "r") as f:
            data = json.load(f)
        settings = DEFAULT_SETTINGS.copy()
        settings.update({k: v for k, v in data.items() if k in settings})
        return settings
    except Exception:
        return DEFAULT_SETTINGS.copy()

def save_settings(settings):
    with open(SETTINGS_PATH, "w") as f:
        json.dump(settings, f)

CAMERA_SETTINGS = load_settings()
save_settings(CAMERA_SETTINGS)

def load_infer_state():
    if not os.path.exists(INFER_STATE_PATH):
        return {"stats": {}, "last_video": {}}
    try:
        with open(INFER_STATE_PATH, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"stats": {}, "last_video": {}}
        data.setdefault("stats", {})
        data.setdefault("last_video", {})
        return data
    except Exception:
        return {"stats": {}, "last_video": {}}

def save_infer_state(state):
    with open(INFER_STATE_PATH, "w") as f:
        json.dump(state, f)

INFER_STATE = load_infer_state()

AVAILABLE_CAMS = {
    0: os.path.exists("/dev/video0"),
    1: os.path.exists("/dev/video1"),
}

def get_active_cams():
    mode = CAMERA_SETTINGS.get("camera_mode", "single")
    if mode == "dual":
        return [cam_id for cam_id, ok in AVAILABLE_CAMS.items() if ok]
    # single mode defaults to cam 0 only
    return [0] if AVAILABLE_CAMS.get(0) else []

def build_pipeline(cam_id: int) -> str:
    wbmode = CAMERA_SETTINGS["wbmode"]
    # Argus exposuretimerange expects nanoseconds; value stored is treated as ns.
    exposure_ns = int(CAMERA_SETTINGS["exposure_us"])
    gain = float(CAMERA_SETTINGS["gain"])
    aelock = "true" if CAMERA_SETTINGS["aelock"] else "false"
    return (
        f"nvarguscamerasrc sensor-id={cam_id} "
        "ee-mode=0 ee-strength=0 tnr-mode=0 tnr-strength=0 "
        f"ispdigitalgainrange=\"1 1\" aelock={aelock} wbmode={wbmode} "
        f"exposuretimerange=\"{exposure_ns} {exposure_ns}\" gainrange=\"{gain} {gain}\" ! "
        f"video/x-raw(memory:NVMM),width={CAPTURE_WIDTH},height={CAPTURE_HEIGHT},format=NV12,framerate=20/1 ! "
        "queue max-size-buffers=1 leaky=downstream ! "
        "nvvidconv flip-method=0 nvbuf-memory-type=0 ! "
        "video/x-raw,format=BGRx ! "
        "videoconvert ! "
        "video/x-raw,format=BGR ! "
        f"appsink name=appsink{cam_id} max-buffers=1 drop=true sync=false"
    )

class CameraState:
    def __init__(self, cam_id: int):
        
        self.cam_id = cam_id
        self.pipeline = None
        self.appsink = None
        self.latest_frame = None
        self.latest_frame_bgr = None
        self.latest_infer_bgr = None
        self.last_infer_ts = 0.0
        self.frame_lock = threading.Lock()
        self.ffmpeg_process = None
        self.recording = False
        self.record_lock = threading.Lock()
        self.current_video_path = None
        self.record_mode = "raw"
        self.record_state = "idle"  # idle | starting | recording | stopping
        self.record_thread = None
        self.running = False
        self.grab_thread = None
        self.infer_running = False
        self.infer_thread = None

CAMERAS = {}

def init_cameras():
    global CAMERAS
    active = get_active_cams()
    CAMERAS = {cam_id: CameraState(cam_id) for cam_id in active}
    init_track_stats(active)
    return active

def init_track_stats(active_cams):
    global TRACK_STATS
    with TRACK_STATS_LOCK:
        TRACK_STATS = {
            cam_id: {"tracks": {}, "leaf_defects": {}, "summary": {}}
            for cam_id in active_cams
        }
        for cam_id in active_cams:
            saved = INFER_STATE.get("stats", {}).get(str(cam_id))
            if isinstance(saved, dict):
                TRACK_STATS[cam_id]["summary"] = saved

def reset_track_stats(cam_id=None):
    with TRACK_STATS_LOCK:
        if cam_id is None:
            for stats in TRACK_STATS.values():
                stats["tracks"] = {}
                stats["leaf_defects"] = {}
                stats["summary"] = {}
                stats.pop("frame_idx", None)
        else:
            stats = TRACK_STATS.get(cam_id)
            if stats is not None:
                stats["tracks"] = {}
                stats["leaf_defects"] = {}
                stats["summary"] = {}
                stats.pop("frame_idx", None)

def maybe_persist_stats(cam_id, stats):
    if stats is None:
        return
    now = time.time()
    last_persist = stats.get("last_persist_ts", 0.0)
    if now - last_persist < 1.0:
        return
    summary = stats.get("summary", {})
    if not isinstance(summary, dict):
        return
    INFER_STATE.setdefault("stats", {})
    INFER_STATE["stats"][str(cam_id)] = summary
    save_infer_state(INFER_STATE)
    stats["last_persist_ts"] = now

def persist_stats_now(cam_id):
    with TRACK_STATS_LOCK:
        stats = TRACK_STATS.get(cam_id)
        if stats is None:
            return
        summary = stats.get("summary", {})
        if not isinstance(summary, dict):
            return
        INFER_STATE.setdefault("stats", {})
        INFER_STATE["stats"][str(cam_id)] = summary
        save_infer_state(INFER_STATE)
        stats["last_persist_ts"] = time.time()

RECORD_FPS = 30
STREAM_FPS = 30
INFER_FPS = 30
SIZE_CONFIG_LOCK = threading.Lock()
SIZE_CONFIG = copy.deepcopy(infer.SIZE_THRESHOLDS)
SIZE_MM_PER_PX = infer.MM_PER_PX

MAIN_MODEL = None
DEFECT_MODEL = None
MODEL_LOCK = threading.Lock()
INFER_LOCK = threading.Lock()
MODEL_READY = threading.Event()
JPEG_LOCK = threading.Lock()

def get_models():
    global MAIN_MODEL, DEFECT_MODEL
    if MAIN_MODEL is None or DEFECT_MODEL is None:
        with MODEL_LOCK:
            if MAIN_MODEL is None or DEFECT_MODEL is None:
                logging.info("Loading YOLO models...")
                MAIN_MODEL, DEFECT_MODEL = infer.load_models()
                MODEL_READY.set()
                logging.info("YOLO models loaded successfully")
    return MAIN_MODEL, DEFECT_MODEL

TRACK_STATS = {}
TRACK_STATS_LOCK = threading.Lock()

STREAM_MODE = "raw"
STREAM_MODE_LOCK = threading.Lock()
INFER_SOURCE = "camera"
INFER_SOURCE_LOCK = threading.Lock()
INFER_VIDEO_PATH = os.path.join(os.path.dirname(__file__), "local_video", "sample.mp4")
INFER_VIDEO_CAP = None
VIDEO_ENDED = False
VIDEO_READ_FAILS = 0
VIDEO_CAP_LOCK = threading.Lock()
MODE_CHANGE_LOCK = threading.Lock()
LAST_MODE_CHANGE_TS = 0.0
LAST_MODE = STREAM_MODE
LAST_SOURCE = INFER_SOURCE
MODE_CHANGE_COOLDOWN = 0.75
CAMERA_RESTART_LOCK = threading.Lock()
LAST_CAMERA_RESTART_TS = 0.0
CAMERA_RESTART_COOLDOWN = 1.0

def _build_output_path(base_dir: str, input_path: str, prefix: str, ext: str) -> str:
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    ts = int(time.time() * 1000)
    filename = f"{prefix}_{base_name}_{ts}.{ext}"
    return os.path.join(base_dir, filename)

def start_ffmpeg(output_path):
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "image2pipe",
        "-vcodec", "mjpeg",
        "-r", str(RECORD_FPS),
        "-i", "-",
        "-threads", "1",
        "-vcodec", "libx264",
        "-bf", "0",
        "-tune", "zerolatency",
        "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        output_path
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)

def stop_ffmpeg(proc):
    if proc and proc.stdin:
        try:
            proc.stdin.close()
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

def write_frame_to_ffmpeg(cam_state: CameraState, frame):
    try:
        proc = cam_state.ffmpeg_process
        if proc and proc.stdin:
            proc.stdin.write(frame)
    except Exception:
        pass

def frame_grabber(cam_state: CameraState):
    while cam_state.running:
        if cam_state.appsink is None:
            time.sleep(0.01)
            continue
        sample = cam_state.appsink.emit("pull-sample")
        if sample is None:
            continue

        buf = sample.get_buffer()
        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            continue

        # Convert BGR raw frame to JPEG bytes for streaming/recording.
        raw = np.frombuffer(mapinfo.data, dtype=np.uint8)
        buf.unmap(mapinfo)
        try:
            frame_bgr = raw.reshape((CAPTURE_HEIGHT, CAPTURE_WIDTH, 3))
        except ValueError:
            continue

        with JPEG_LOCK:
            ok, jpeg = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            continue
        frame = jpeg.tobytes()

        with cam_state.frame_lock:
            cam_state.latest_frame_bgr = frame_bgr.copy()
            cam_state.latest_frame = frame

def _start_camera(cam_state: CameraState):
    cam_state.running = True
    try:
        cam_state.pipeline = Gst.parse_launch(build_pipeline(cam_state.cam_id))
        if cam_state.pipeline is None:
            logging.error(f"Failed to parse GStreamer pipeline for camera {cam_state.cam_id}")
            return False
        cam_state.appsink = cam_state.pipeline.get_by_name(f"appsink{cam_state.cam_id}")
        if cam_state.appsink is None:
            logging.error(f"Failed to get appsink from pipeline for camera {cam_state.cam_id}")
            return False
        cam_state.pipeline.set_state(Gst.State.PLAYING)
        # Give pipeline time to start
        import time as time_mod
        time_mod.sleep(0.5)
        cam_state.grab_thread = threading.Thread(target=frame_grabber, args=(cam_state,), daemon=True)
        cam_state.grab_thread.start()
        logging.info(f"Camera {cam_state.cam_id} started successfully")
        return True
    except Exception as e:
        logging.error(f"Error starting camera {cam_state.cam_id}: {e}")
        cam_state.running = False
        return False

def _stop_camera(cam_state: CameraState):
    cam_state.running = False
    if cam_state.grab_thread and cam_state.grab_thread.is_alive():
        cam_state.grab_thread.join(timeout=1.0)
    cam_state.grab_thread = None
    if cam_state.pipeline:
        cam_state.pipeline.set_state(Gst.State.NULL)
        cam_state.pipeline = None
    cam_state.appsink = None

def rebuild_cameras():
    for cam_state in CAMERAS.values():
        _stop_camera(cam_state)
    active = init_cameras()
    if not active:
        logging.warning("No cameras found for current mode")
        return
    if len(active) == 1:
        logging.warning("Only one camera available; dual camera mode is not possible")
    for cam_state in CAMERAS.values():
        _start_camera(cam_state)

def restart_cameras_in_place():
    for cam_state in CAMERAS.values():
        _stop_camera(cam_state)
    for cam_state in CAMERAS.values():
        _start_camera(cam_state)

def _camera_restart_cooldown():
    global LAST_CAMERA_RESTART_TS
    with CAMERA_RESTART_LOCK:
        now = time.time()
        wait = CAMERA_RESTART_COOLDOWN - (now - LAST_CAMERA_RESTART_TS)
        if wait > 0:
            time.sleep(wait)
        LAST_CAMERA_RESTART_TS = time.time()

def safe_stop_all_cameras():
    _camera_restart_cooldown()
    stop_all_cameras()

def safe_start_all_cameras():
    _camera_restart_cooldown()
    start_all_cameras()

def stop_all_cameras():
    for cam_state in CAMERAS.values():
        _stop_camera(cam_state)

def start_all_cameras():
    for cam_state in CAMERAS.values():
        if cam_state.pipeline is None:
            _start_camera(cam_state)

def get_latest_frame(cam_state: CameraState):
    with cam_state.frame_lock:
        return cam_state.latest_frame

def get_latest_frame_bgr(cam_state: CameraState):
    with cam_state.frame_lock:
        if cam_state.latest_frame_bgr is None:
            return None
        return cam_state.latest_frame_bgr.copy()

def get_latest_infer_bgr(cam_state: CameraState):
    with cam_state.frame_lock:
        if cam_state.latest_infer_bgr is None:
            return None
        return cam_state.latest_infer_bgr.copy()

def infer_worker(cam_state: CameraState):
    interval = 1.0 / float(INFER_FPS)
    while cam_state.infer_running:
        with INFER_SOURCE_LOCK:
            source = INFER_SOURCE
        if source == "video":
            global INFER_VIDEO_CAP
            with VIDEO_CAP_LOCK:
                if INFER_VIDEO_CAP is None or not INFER_VIDEO_CAP.isOpened():
                    INFER_VIDEO_CAP = cv2.VideoCapture(INFER_VIDEO_PATH)
                ok, vframe = INFER_VIDEO_CAP.read() if INFER_VIDEO_CAP else (False, None)
            if not ok or vframe is None:
                time.sleep(0.05)
                continue
            frame_bgr = vframe
            with cam_state.frame_lock:
                cam_state.latest_frame_bgr = frame_bgr.copy()
        else:
            frame_bgr = get_latest_frame_bgr(cam_state)
            if frame_bgr is None:
                time.sleep(0.01)
                continue
        main_model, defect_model = get_models()
        with TRACK_STATS_LOCK:
            stats = TRACK_STATS.get(cam_state.cam_id)
        with SIZE_CONFIG_LOCK:
            size_cfg = SIZE_CONFIG
            mm_per_px = SIZE_MM_PER_PX
        with INFER_LOCK:
            annotated, summary = infer.infer_frame_leaf_grouped_tracked(
                main_model,
                defect_model,
                frame_bgr,
                stats,
                size_cfg=size_cfg,
                mm_per_px=mm_per_px,
            )
        with TRACK_STATS_LOCK:
            if stats is not None:
                stats["summary"] = summary
                maybe_persist_stats(cam_state.cam_id, stats)
        with cam_state.frame_lock:
            cam_state.latest_infer_bgr = annotated.copy()
            cam_state.last_infer_ts = time.time()
        time.sleep(interval)

def update_infer_workers():
    with STREAM_MODE_LOCK:
        stream_infer = STREAM_MODE == "infer"
    for cam_state in CAMERAS.values():
        with cam_state.record_lock:
            record_infer = cam_state.recording and cam_state.record_mode == "infer"
        should_run = stream_infer or record_infer
        if should_run and not cam_state.infer_running:
            cam_state.infer_running = True
            cam_state.infer_thread = threading.Thread(
                target=infer_worker, args=(cam_state,), daemon=True
            )
            cam_state.infer_thread.start()
        elif not should_run and cam_state.infer_running:
            cam_state.infer_running = False

def record_worker(cam_state: CameraState):
    interval = 1.0 / float(RECORD_FPS)
    while True:
        with cam_state.record_lock:
            if not cam_state.recording:
                break
            mode = cam_state.record_mode
            proc = cam_state.ffmpeg_process
        if proc is None or proc.poll() is not None:
            break
        if mode == "infer":
            annotated = get_latest_infer_bgr(cam_state)
            if annotated is not None:
                with JPEG_LOCK:
                    ok, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok:
                    write_frame_to_ffmpeg(cam_state, jpeg.tobytes())
        else:
            frame = get_latest_frame(cam_state)
            if frame:
                write_frame_to_ffmpeg(cam_state, frame)
        time.sleep(interval)

def start_recording(cam_state: CameraState, mode: str):
    with cam_state.record_lock:
        if cam_state.record_state in ("starting", "recording", "stopping"):
            return False
        cam_state.record_state = "starting"
        cam_state.record_mode = mode
        suffix = "infer" if mode == "infer" else "raw"
        filename = "top.mp4" if cam_state.cam_id == 0 else "bottom.mp4"
        if suffix == "infer":
            name, ext = os.path.splitext(filename)
            filename = f"{name}_infer{ext}"
        cam_state.current_video_path = f"{video_dir}/{filename}"

    def wait_and_start():
        # Wait for at least one frame to avoid starting ffmpeg with no stream.
        deadline = time.time() + 5.0
        while time.time() < deadline:
            with cam_state.record_lock:
                if cam_state.record_state != "starting":
                    return
            frame = get_latest_infer_bgr(cam_state) if mode == "infer" else get_latest_frame_bgr(cam_state)
            if frame is not None:
                break
            time.sleep(0.05)
        with cam_state.record_lock:
            if cam_state.record_state != "starting":
                return
            cam_state.ffmpeg_process = start_ffmpeg(cam_state.current_video_path)
            cam_state.recording = True
            cam_state.record_state = "recording"
            cam_state.record_thread = threading.Thread(target=record_worker, args=(cam_state,), daemon=True)
            cam_state.record_thread.start()
        update_infer_workers()

    threading.Thread(target=wait_and_start, daemon=True).start()
    update_infer_workers()
    return True

def stop_recording(cam_state: CameraState):
    with cam_state.record_lock:
        if cam_state.record_state == "starting":
            cam_state.record_state = "idle"
            cam_state.recording = False
            return True
        if cam_state.record_state != "recording":
            return False
        cam_state.record_state = "stopping"
        cam_state.recording = False
        record_thread = cam_state.record_thread
    if record_thread and record_thread.is_alive():
        record_thread.join(timeout=2.0)
    with cam_state.record_lock:
        stop_ffmpeg(cam_state.ffmpeg_process)
        cam_state.ffmpeg_process = None
        cam_state.record_thread = None
        cam_state.record_state = "idle"
    persist_stats_now(cam_state.cam_id)
    update_infer_workers()
    return True

def stop_all_recordings():
    for cam_state in CAMERAS.values():
        stop_recording(cam_state)

def handle_video_eof():
    global INFER_VIDEO_CAP
    with VIDEO_CAP_LOCK:
        if INFER_VIDEO_CAP:
            INFER_VIDEO_CAP.release()
            INFER_VIDEO_CAP = None
    with STREAM_MODE_LOCK:
        global STREAM_MODE
        STREAM_MODE = "raw"
    with INFER_SOURCE_LOCK:
        global INFER_SOURCE, VIDEO_ENDED
        INFER_SOURCE = "camera"
        VIDEO_ENDED = True
    stop_all_recordings()
    safe_start_all_cameras()
    update_infer_workers()

def mjpeg_generator(cam_state: CameraState):
    boundary = b"--frame"
    interval = 1.0 / float(STREAM_FPS)
    while True:
        with INFER_SOURCE_LOCK:
            source = INFER_SOURCE
        if source == "video":
            global INFER_VIDEO_CAP
            with VIDEO_CAP_LOCK:
                if INFER_VIDEO_CAP is None or not INFER_VIDEO_CAP.isOpened():
                    INFER_VIDEO_CAP = cv2.VideoCapture(INFER_VIDEO_PATH)
                ok, vframe = INFER_VIDEO_CAP.read() if INFER_VIDEO_CAP else (False, None)
            if not ok or vframe is None:
                global VIDEO_READ_FAILS
                VIDEO_READ_FAILS += 1
                if VIDEO_READ_FAILS < 3:
                    time.sleep(0.05)
                    continue
                handle_video_eof()
                VIDEO_READ_FAILS = 0
                time.sleep(0.01)
                continue
            VIDEO_READ_FAILS = 0
            frame_bgr = vframe
            with JPEG_LOCK:
                ok_jpeg, jpeg = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            with cam_state.frame_lock:
                cam_state.latest_frame_bgr = frame_bgr.copy()
                cam_state.latest_frame = jpeg.tobytes() if ok_jpeg else None
        else:
            frame_bgr = get_latest_frame_bgr(cam_state)
            if frame_bgr is None:
                time.sleep(0.01)
                continue
        with STREAM_MODE_LOCK:
            mode = STREAM_MODE
        if mode == "infer":
            inferred = get_latest_infer_bgr(cam_state)
            if inferred is not None:
                frame_bgr = inferred
        if source == "video":
            # Preserve aspect ratio for video source by letterboxing to stream size.
            sw, sh = STREAM_WIDTH, STREAM_HEIGHT
            h, w = frame_bgr.shape[:2]
            if (w, h) != (sw, sh):
                scale = min(sw / float(w), sh / float(h))
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
                canvas = np.zeros((sh, sw, 3), dtype=resized.dtype)
                x0 = (sw - new_w) // 2
                y0 = (sh - new_h) // 2
                canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
                frame_bgr = canvas
        else:
            if (STREAM_WIDTH, STREAM_HEIGHT) != (CAPTURE_WIDTH, CAPTURE_HEIGHT):
                frame_bgr = cv2.resize(frame_bgr, (STREAM_WIDTH, STREAM_HEIGHT), interpolation=cv2.INTER_AREA)
        with JPEG_LOCK:
            ok, jpeg = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            time.sleep(0.01)
            continue
        frame = jpeg.tobytes()
        yield (
            boundary
            + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
            + str(len(frame)).encode()
            + b"\r\n\r\n"
            + frame
            + b"\r\n"
        )
        time.sleep(interval)
@app.on_event("startup")
def start_pipeline():
    logging.info("Starting camera pipeline...")
    active = init_cameras()
    if not active:
        logging.error("No cameras found at /dev/video0 or /dev/video1")
        return
    logging.info(f"Found cameras: {active}")
    if len(active) == 1:
        logging.info("Single camera mode enabled")
    for cam_state in CAMERAS.values():
        success = _start_camera(cam_state)
        if not success:
            logging.error(f"Failed to start camera {cam_state.cam_id}")
    logging.info("Camera pipeline startup complete")
    logging.info("YOLO model will be loaded on first inference request (lazy loading)")

@app.on_event("shutdown")
def stop_pipeline():
    for cam_state in CAMERAS.values():
        cam_state.infer_running = False
        if cam_state.infer_thread and cam_state.infer_thread.is_alive():
            cam_state.infer_thread.join(timeout=2.0)
        with cam_state.record_lock:
            cam_state.recording = False
        if cam_state.record_thread and cam_state.record_thread.is_alive():
            cam_state.record_thread.join(timeout=2.0)
        with cam_state.record_lock:
            stop_ffmpeg(cam_state.ffmpeg_process)
            cam_state.ffmpeg_process = None
            cam_state.record_thread = None
            cam_state.record_state = "idle"
        _stop_camera(cam_state)
    global INFER_VIDEO_CAP
    with VIDEO_CAP_LOCK:
        if INFER_VIDEO_CAP:
            INFER_VIDEO_CAP.release()
            INFER_VIDEO_CAP = None

import logging

# Configure logging to write to frontend_logs.log
logging.basicConfig(
    filename="frontend_logs.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    filemode="w"
)

@app.post("/log")
async def log_message(request: Request):
    data = await request.json()
    message = data.get("message", "No message provided")
    logging.info(message)
    return {"status": "success"}

@app.post("/save-local")
async def save_local(file: UploadFile = File(...), path: str = Form(...)):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(await file.read())
        return {"status": "saved", "path": path}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/camera/settings")
def get_camera_settings():
    return CAMERA_SETTINGS

@app.post("/camera/settings")
async def set_camera_settings(
    cam: int = Form(0),
    wbmode: int = Form(None),
    exposure_us: int = Form(None),
    gain: float = Form(None),
    aelock: bool = Form(None),
    camera_mode: str = Form(None),):
    
    if camera_mode is None and cam not in CAMERAS:
        raise HTTPException(status_code=404, detail="camera_not_available")

    mode_changed = False
    prev_mode = CAMERA_SETTINGS.get("camera_mode", "single")
    if wbmode is not None:
        CAMERA_SETTINGS["wbmode"] = int(wbmode)
    if exposure_us is not None:
        CAMERA_SETTINGS["exposure_us"] = int(exposure_us)
    if gain is not None:
        CAMERA_SETTINGS["gain"] = float(gain)
    if aelock is not None:
        CAMERA_SETTINGS["aelock"] = bool(aelock)
    if camera_mode is not None:
        if camera_mode not in ("single", "dual"):
            raise HTTPException(status_code=400, detail="camera_mode_must_be_single_or_dual")
        CAMERA_SETTINGS["camera_mode"] = camera_mode
        mode_changed = (camera_mode != prev_mode)

    save_settings(CAMERA_SETTINGS)

    if mode_changed:
        rebuild_cameras()
    else:
        restart_cameras_in_place()

    return {"status": "updated", "settings": CAMERA_SETTINGS}

@app.get("/stream")
def stream(cam: int = 0):
    if cam not in CAMERAS:
        raise HTTPException(status_code=404, detail="camera_not_available")
    return StreamingResponse(
        mjpeg_generator(CAMERAS[cam]),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/stream/mode")
def get_stream_mode():
    with STREAM_MODE_LOCK:
        return {"mode": STREAM_MODE}

@app.post("/stream/mode")
async def set_stream_mode(
    mode: str = Form(...),
    source: str = Form(None),
    size_config: str = Form(None),
    mm_per_px: float = Form(None),
):
    global STREAM_MODE, LAST_MODE_CHANGE_TS, LAST_MODE, LAST_SOURCE, INFER_SOURCE, INFER_VIDEO_CAP, VIDEO_ENDED, VIDEO_READ_FAILS
    if mode not in ("raw", "infer"):
        raise HTTPException(status_code=400, detail="mode_must_be_raw_or_infer")
    if source is not None and source not in ("camera", "video"):
        raise HTTPException(status_code=400, detail="source_must_be_camera_or_video")
    with INFER_SOURCE_LOCK:
        current_source = INFER_SOURCE
    desired_source = source if source is not None else current_source
    now = time.time()
    with MODE_CHANGE_LOCK:
        if (
            mode == LAST_MODE
            and desired_source == LAST_SOURCE
            and (now - LAST_MODE_CHANGE_TS) < MODE_CHANGE_COOLDOWN
        ):
            return {"status": "ok", "mode": STREAM_MODE, "ignored": True}
        LAST_MODE_CHANGE_TS = now
        LAST_MODE = mode
        LAST_SOURCE = desired_source
    with STREAM_MODE_LOCK:
        STREAM_MODE = mode
    if source is not None:
        with INFER_SOURCE_LOCK:
            INFER_SOURCE = source
            VIDEO_ENDED = False
            VIDEO_READ_FAILS = 0
            with VIDEO_CAP_LOCK:
                if INFER_VIDEO_CAP:
                    INFER_VIDEO_CAP.release()
                    INFER_VIDEO_CAP = None
        if source == "video":
            safe_stop_all_cameras()
        else:
            safe_start_all_cameras()
    if mode == "infer":
        reset_track_stats()
        for cam_state in CAMERAS.values():
            start_recording(cam_state, "infer")
    if mode == "raw":
        for cam_state in CAMERAS.values():
            persist_stats_now(cam_state.cam_id)
        stop_all_recordings()
    if size_config is not None or mm_per_px is not None:
        with SIZE_CONFIG_LOCK:
            global SIZE_CONFIG, SIZE_MM_PER_PX
            if size_config is not None:
                try:
                    parsed = json.loads(size_config)
                except Exception:
                    raise HTTPException(status_code=400, detail="invalid_size_config_json")
                if not isinstance(parsed, dict):
                    raise HTTPException(status_code=400, detail="invalid_size_config_format")
                SIZE_CONFIG = parsed
            if mm_per_px is not None:
                try:
                    mm_val = float(mm_per_px)
                except Exception:
                    raise HTTPException(status_code=400, detail="invalid_mm_per_px")
                if mm_val > 0:
                    SIZE_MM_PER_PX = mm_val
    update_infer_workers()
    return {"status": "ok", "mode": STREAM_MODE}

@app.get("/infer/stats")
def infer_stats(cam: int = 0):
    with TRACK_STATS_LOCK:
        stats = TRACK_STATS.get(cam)
        if stats is None:
            summary = {}
        else:
            summary = stats.get("summary", {})
    last_video = INFER_STATE.get("last_video", {}).get(str(cam))
    if not summary and isinstance(INFER_STATE.get("stats", {}).get(str(cam)), dict):
        summary = INFER_STATE["stats"][str(cam)]
    with INFER_SOURCE_LOCK:
        video_ended = VIDEO_ENDED
    return {"status": "ok", "summary": summary, "last_video": last_video, "video_ended": video_ended}

@app.get("/capture")
def capture(request: Request, cam: int = 0):
    if cam not in CAMERAS:
        raise HTTPException(status_code=404, detail="camera_not_available")
    frame = get_latest_frame(CAMERAS[cam])
    if frame is None:
        return {"error": "no frame"}

    filename = "top.jpg" if cam == 0 else "bottom.jpg"
    path = f"{capture_dir}/{filename}"
    with open(path, "wb") as f:
        f.write(frame)

    return {
        "status": "captured",
        "file": os.path.abspath(path),
        "url": f"/media/images/{filename}",
    }

@app.post("/record/start")
def record_start(request: Request, cam: int = 0, mode: str = Form("raw")):
    if cam not in CAMERAS:
        raise HTTPException(status_code=404, detail="camera_not_available")
    if mode not in ("raw", "infer"):
        raise HTTPException(status_code=400, detail="mode_must_be_raw_or_infer")
    cam_state = CAMERAS[cam]
    started = start_recording(cam_state, mode)
    if not started:
        return {"status": "already_recording"}

    return {
        "status": "recording_started",
        "file": os.path.abspath(cam_state.current_video_path),
        "url": f"/media/videos/{os.path.basename(cam_state.current_video_path)}",
    }

@app.post("/record/stop")
def record_stop(request: Request, cam: int = 0):
    if cam not in CAMERAS:
        raise HTTPException(status_code=404, detail="camera_not_available")
    cam_state = CAMERAS[cam]
    stopped = stop_recording(cam_state)
    if not stopped:
        return {"status": "not_recording"}

    # Return a URL for the completed video file
    if cam_state.current_video_path:
        filename = os.path.basename(cam_state.current_video_path)
        file_url = os.path.abspath(cam_state.current_video_path)
        url = f"/media/videos/{filename}"
    else:
        file_url = None
        url = None

    INFER_STATE.setdefault("last_video", {})
    INFER_STATE["last_video"][str(cam)] = {
        "file": file_url,
        "url": url,
        "mode": cam_state.record_mode,
        "ts": time.time(),
    }
    save_infer_state(INFER_STATE)

    with STREAM_MODE_LOCK:
        global STREAM_MODE
        STREAM_MODE = "raw"
    with INFER_SOURCE_LOCK:
        global INFER_SOURCE, INFER_VIDEO_CAP
        if INFER_SOURCE == "video":
            INFER_SOURCE = "camera"
            if INFER_VIDEO_CAP:
                INFER_VIDEO_CAP.release()
                INFER_VIDEO_CAP = None
            start_all_cameras()

    return {"status": "recording_stopped", "file": file_url, "url": url}
