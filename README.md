# Exceed Server

FastAPI-based two-stage AI inference server for Jetson Orin NX.  
Runs YOLO segmentation (stage-1) + RF-DETR defect detection (stage-2) on a live camera or video file.

---

## Stack

- **Runtime**: Python 3.10, FastAPI, Uvicorn
- **Inference**: TensorRT engines via Ultralytics (stage-1) and native TRT Python API (stage-2)
- **Camera**: GStreamer + NvArgus (Jetson CSI)
- **Tracking**: ByteTrack (supervision)
- **Platform**: NVIDIA Jetson Orin NX

---

## Project Structure

```
server.py            — FastAPI app, routes, camera/pipeline lifecycle
infer_main.py        — Model loading, main inference pipeline entry point
infer_detect.py      — Stage-1/2 detection logic, leaf crop, dedup
infer_draw.py        — Box/mask/defect drawing, tracking stats
infer_rfdetr.py      — RF-DETR TensorRT wrapper (stage-2)
pipeline.py          — Camera/video pipeline loops, MJPEG stream
config_store.py      — Config file loading/saving (unified API)
stem_detector.py     — OpenCV-based stem length measurement
smart_monitor.py     — Runtime monitor (RAM, CPU, GPU, events)
utils.py             — JSON helpers, deep merge

config/
  server_config.json    — FPS, power mode, monitor settings
  model1_config.json    — Stage-1 model path, classes, ROI
  model2_config.json    — Stage-2 model path, classes, confidence
  pipeline_config.json  — Tracking, GC interval, warmup, video path
  camera_config.json    — Exposure, gain, white balance
  class_config.json     — (generated) merged class map

models/               — TensorRT .engine files (not in git)
media/                — Camera output images/videos
```

---

## Running

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1
```

For max performance on Jetson before starting:
```bash
sudo nvpmodel -m 0      # MAX-N power mode
sudo jetson_clocks       # lock clocks to max frequency
```

---

## Key Config Files

All configs are **hot-reloaded** — edit the file and the server picks up changes on the next frame without restart. Only model path changes require a restart.

### `config/model1_config.json`
- `model.path` — stage-1 engine file
- `model.config.imgsz` — inference resolution (default 640)
- `model.config.conf_leaf` — leaf confidence threshold
- `model.config.roi_enabled` — enable center-ROI filtering

### `config/model2_config.json`
- `model.path` — stage-2 RF-DETR engine file
- `model.config.imgsz` — must match engine export size (currently 512)
- `model.config.rfdetr_num_classes` — number of defect classes
- `classes` — per-class confidence, color, min size, role

### `config/pipeline_config.json`
- `infer.tracking_enabled` — enable ByteTrack across frames
- `infer.gc_every_n_frames` — GC + CUDA cache flush interval (default 10)
- `infer.unload_on_stop` — unload models when inference stops

### `config/server_config.json`
- `mode_related.stream_fps` — MJPEG stream FPS
- `mode_related.infer_fps` — inference target FPS

---

## Changelog

### v1.0.0 — 2026-05-14
Initial stable version pushed to this repository.

**Fixes applied:**
- `model2_config.json`: fixed `imgsz` from 640 → 512 to match RF-DETR engine export size (was crashing every stage-2 inference silently)
- `model2_config.json`: switched from `rfdetr_fp32.engine` → `rfdetr_fp16.engine` (~2× GPU speedup)
- `infer_rfdetr.py`: `non_blocking=True` on GPU buffer copy + stream-scoped sync instead of global `torch.cuda.synchronize()`
- `infer_main.py`: GC + `empty_cache()` moved to every N frames (configurable via `gc_every_n_frames`) instead of every 1 second (which fired every frame at 2–3 FPS)
- `server.py`: removed `sudo systemctl restart nvargus-daemon` at startup (was prompting for password and timing out every boot)
- `config_store.py`: added `gc_every_n_frames` field wired through pipeline config

**Known bottlenecks (not yet fixed):**
- RF-DETR runs once per detected leaf sequentially — N leaves = N×latency
- Seg masks transferred GPU→CPU once per leaf (N CUDA sync points per frame)
- `get_class_config()` called inside the per-leaf loop
