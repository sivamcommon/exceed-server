"""
Runtime pipeline loops extracted from server.py.

This module keeps camera/video capture, inference worker, and MJPEG streaming
flow in one place while server.py injects behavior via hooks.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from queue import Empty
from typing import Callable

import cv2


@dataclass
class PipelineHooks:
    run_inference: Callable[[object, object], object]
    recorder_write: Callable[[object, object], bool]
    should_record_raw: Callable[[], bool]
    monitor_event: Callable[[str], None]
    get_infer_source: Callable[[], str]
    read_video_frame: Callable[[], tuple[bool, object | None]]
    handle_video_eof: Callable[[], None]
    is_video_worker_running: Callable[[], bool]
    set_video_worker_running: Callable[[bool], None]
    get_video_slow_level: Callable[[], int]
    get_infer_fps: Callable[[], float]
    get_record_fps: Callable[[], float]
    get_frame_generation: Callable[[object], int]


def _enqueue_latest_frame(cam_state: object, frame: object):
    """Queue newest frame for inference without blocking capture."""
    try:
        cam_state.shared.infer_queue.put_nowait(frame)
        return
    except Exception:
        pass
    try:
        cam_state.shared.infer_queue.get_nowait()
    except Empty:
        pass
    except Exception:
        pass
    try:
        cam_state.shared.infer_queue.put_nowait(frame)
    except Exception:
        pass


def _record_frame_due(cam_state: object, hooks: PipelineHooks) -> bool:
    """Throttle recorder writes using record fps setting."""
    try:
        fps = float(max(hooks.get_record_fps(), 1.0))
    except Exception:
        fps = 20.0
    interval = 1.0 / fps
    now = time.time()
    last = float(getattr(cam_state.shared, "last_record_ts", 0.0))
    if (now - last) < interval:
        return False
    cam_state.shared.last_record_ts = now
    return True


def infer_worker_loop(cam_state: object, hooks: PipelineHooks):
    """Worker that consumes queued frames and publishes annotated frames."""
    worker_generation = int(hooks.get_frame_generation(cam_state))
    while cam_state.running:
        # Hard-stop stale workers after any mode/source generation change.
        if int(hooks.get_frame_generation(cam_state)) != worker_generation:
            break
        if not bool(getattr(cam_state.shared, "inferring", False)):
            time.sleep(0.01)
            continue
        src = hooks.get_infer_source()
        frame = None
        if src == "camera" and getattr(cam_state.shared, "frame_mode", "realtime") == "buffer":
            try:
                frame = cam_state.shared.infer_queue.get(timeout=0.1)
            except Empty:
                continue
        else:
            frame = getattr(cam_state.shared, "latest_raw", None)
            if frame is None:
                time.sleep(0.005)
                continue
        try:
            annotated = hooks.run_inference(frame, cam_state)
            cam_state.shared.latest = annotated
            try:
                cam_state.shared.latest_seq = int(getattr(cam_state.shared, "latest_seq", 0)) + 1
                cam_state.shared.latest_ts = time.time()
            except Exception:
                pass
            if bool(getattr(cam_state.shared, "recording", False)) and _record_frame_due(cam_state, hooks):
                if not hooks.recorder_write(cam_state.ffmpeg_process, annotated):
                    cam_state.shared.recording = False
                    logging.warning("cam %s: recorder write failed, stopping record", cam_state.cam_id)
        except Exception as exc:
            logging.warning("infer_worker_loop cam %s error: %s", cam_state.cam_id, exc)
            time.sleep(0.01)
    hooks.monitor_event(f"INFER_WORKER_EXIT cam{cam_state.cam_id}")


def camera_pipeline_loop(cam_state: object, grab_frame: Callable[[object], object], hooks: PipelineHooks):
    """Camera grab loop; starts infer worker on demand and manages queue pressure."""
    infer_thread = None
    prev_inferring = bool(getattr(cam_state.shared, "inferring", False))
    while cam_state.running:
        frame = grab_frame(cam_state)
        if frame is None:
            time.sleep(0.01)
            continue
        inferring = bool(getattr(cam_state.shared, "inferring", False))
        if inferring and not prev_inferring:
            # Infer-only stream mode: clear previously published raw frame so the
            # stream waits for the next annotated inference result.
            cam_state.shared.latest = None
        cam_state.shared.latest_raw = frame
        if inferring:
            if getattr(cam_state.shared, "frame_mode", "realtime") == "buffer":
                _enqueue_latest_frame(cam_state, frame)
            if infer_thread is None or not infer_thread.is_alive():
                hooks.monitor_event(f"INFER_WORKER_START cam{cam_state.cam_id}")
                infer_thread = threading.Thread(
                    target=infer_worker_loop,
                    args=(cam_state, hooks),
                    daemon=True,
                )
                infer_thread.start()
        else:
            cam_state.shared.latest = frame
            if getattr(cam_state.shared, "frame_mode", "realtime") == "buffer":
                while True:
                    try:
                        cam_state.shared.infer_queue.get_nowait()
                    except Empty:
                        break
            if bool(getattr(cam_state.shared, "recording", False)) and hooks.should_record_raw() and _record_frame_due(cam_state, hooks):
                if not hooks.recorder_write(cam_state.ffmpeg_process, frame):
                    cam_state.shared.recording = False
                    logging.warning("cam %s: recorder write failed, stopping record", cam_state.cam_id)
        prev_inferring = inferring


def video_pipeline_loop(cam_state: object, hooks: PipelineHooks):
    """Video source loop — every frame is inferred, streamed, and recorded."""
    prev_inferring = False
    hooks.monitor_event("VIDEO_WORKER_START")
    worker_generation = int(hooks.get_frame_generation(cam_state))
    cam_state.running = True
    try:
        while True:
            if int(hooks.get_frame_generation(cam_state)) != worker_generation:
                break
            if not hooks.is_video_worker_running():
                break
            ok, vframe = hooks.read_video_frame()
            if not ok:
                hooks.handle_video_eof()
                break
            if vframe is None:
                time.sleep(0.005)
                continue
            if vframe.shape[:2] != (1080, 1920):
                vframe = cv2.resize(vframe, (1920, 1080), interpolation=cv2.INTER_LINEAR)
            cam_state.shared.latest_raw = vframe
            inferring = bool(getattr(cam_state.shared, "inferring", False))
            if inferring and not prev_inferring:
                cam_state.shared.latest = None
            if inferring:
                try:
                    annotated = hooks.run_inference(vframe, cam_state)
                    cam_state.shared.latest = annotated
                    try:
                        cam_state.shared.latest_seq = int(getattr(cam_state.shared, "latest_seq", 0)) + 1
                        cam_state.shared.latest_ts = time.time()
                    except Exception:
                        pass
                    if bool(getattr(cam_state.shared, "recording", False)):
                        if not hooks.recorder_write(cam_state.ffmpeg_process, annotated):
                            cam_state.shared.recording = False
                            logging.warning("cam %s: recorder write failed, stopping record", cam_state.cam_id)
                except Exception as exc:
                    logging.warning("video_pipeline_loop cam %s infer error: %s", cam_state.cam_id, exc)
                    cam_state.shared.latest = vframe
            else:
                cam_state.shared.latest = vframe
                if bool(getattr(cam_state.shared, "recording", False)) and hooks.should_record_raw():
                    if not hooks.recorder_write(cam_state.ffmpeg_process, vframe):
                        cam_state.shared.recording = False
                        logging.warning("cam %s: recorder write failed, stopping record", cam_state.cam_id)
            slow = int(max(0, hooks.get_video_slow_level()))
            if slow > 0:
                time.sleep(0.01 * slow)
            prev_inferring = inferring
    finally:
        cam_state.running = False
        hooks.set_video_worker_running(False)
        hooks.monitor_event("VIDEO_WORKER_STOP")


def mjpeg_stream_generator(
    cam_state: object,
    stream_token: int,
    on_disconnect: Callable[[bool], None],
    jpeg_quality: int = 80,
    stream_fps: float = 20.0,
):
    """Yield MJPEG chunks from latest frame."""
    interval = 1.0 / float(max(stream_fps, 1.0))
    disconnected = False
    try:
        while True:
            with cam_state.stream_lock:
                replaced = cam_state.stream_token != stream_token
                active = cam_state.stream_running
            if replaced or not active:
                on_disconnect(replaced)
                disconnected = True
                break
            output = cam_state.shared.latest
            if output is None:
                time.sleep(0.01)
                continue
            ok, jpeg = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
            if not ok:
                time.sleep(0.01)
                continue
            payload = jpeg.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"
            )
            time.sleep(interval)
    finally:
        # If the client disconnects (browser closed / network drop), the generator
        # will be cancelled and we won't hit the "replaced/not active" branch.
        # Ensure server-side stream bookkeeping is updated.
        if not disconnected:
            try:
                on_disconnect(False)
            except Exception:
                pass

