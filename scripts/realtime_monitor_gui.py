#!/usr/bin/env python3
"""
Realtime monitor GUI for legacy pipeline.

Plots 3 live graphs from monitor.log:
  1) RAM usage (%)
  2) Process RSS (MB)
  3) Detection count (cam0_tracks, fallback cam0_defcnt)

Enable/disable via config/server_config.json:
  others.live_plot_gui_enabled = true|false

Usage:
  python3 scripts/realtime_monitor_gui.py
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import sys
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
DEFAULT_MONITOR_LOG = os.path.join(ROOT_DIR, "monitor.log")
DEFAULT_SERVER_CFG = os.path.join(ROOT_DIR, "config", "server_config.json")

LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \| .*?"
    r"ram=\d+/\d+MB \((?P<ram_pct>\d+)%\) \| "
    r"rss=(?P<rss>\d+)MB(?P<tail>.*)$"
)
TRACKS_RE = re.compile(r"cam0_tracks=(\d+)")
DEFCNT_RE = re.compile(r"cam0_defcnt=(\d+)")


def _is_gui_enabled(cfg_path: str) -> bool:
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        others = data.get("others", {}) if isinstance(data, dict) else {}
        return bool(others.get("live_plot_gui_enabled", False))
    except Exception:
        return False


def _extract_detection_count(extra: str) -> int:
    if not extra:
        return 0
    m = TRACKS_RE.search(extra)
    if m:
        return int(m.group(1))
    m = DEFCNT_RE.search(extra)
    if m:
        return int(m.group(1))
    return 0


class MonitorTail:
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.pos = 0
        self.ino = None

    def read_new_points(self):
        points = []
        if not os.path.exists(self.log_path):
            return points

        st = os.stat(self.log_path)
        if self.ino is None or self.ino != st.st_ino:
            self.ino = st.st_ino
            self.pos = 0
        elif st.st_size < self.pos:
            self.pos = 0

        with open(self.log_path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(self.pos)
            chunk = f.read()
            self.pos = f.tell()

        for raw in chunk.splitlines():
            line = raw.strip()
            m = LINE_RE.match(line)
            if not m:
                continue
            try:
                ts = dt.datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S")
                ram_pct = int(m.group("ram_pct"))
                rss = int(m.group("rss"))
                det = _extract_detection_count(m.group("tail") or "")
            except Exception:
                continue
            points.append((ts, ram_pct, rss, det))
        return points


def main() -> int:
    cfg_path = DEFAULT_SERVER_CFG
    if not _is_gui_enabled(cfg_path):
        print(
            "live_plot_gui_enabled=false in config/server_config.json -> GUI not started.\n"
            "Set others.live_plot_gui_enabled=true to enable."
        )
        return 0

    log_path = DEFAULT_MONITOR_LOG
    tail = MonitorTail(log_path)

    max_points = 600
    xs = deque(maxlen=max_points)
    ram_pcts = deque(maxlen=max_points)
    rss_mbs = deque(maxlen=max_points)
    det_counts = deque(maxlen=max_points)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ax_ram, ax_rss, ax_det = axes

    (ln_ram,) = ax_ram.plot([], [], color="tab:red", linewidth=1.8, label="RAM %")
    (ln_rss,) = ax_rss.plot([], [], color="tab:blue", linewidth=1.8, label="RSS MB")
    (ln_det,) = ax_det.plot([], [], color="tab:green", linewidth=1.8, label="Detection Count")

    ax_ram.set_ylabel("RAM %")
    ax_rss.set_ylabel("RSS (MB)")
    ax_det.set_ylabel("Count")
    ax_det.set_xlabel("Time")
    ax_ram.grid(alpha=0.3)
    ax_rss.grid(alpha=0.3)
    ax_det.grid(alpha=0.3)
    ax_ram.legend(loc="upper left")
    ax_rss.legend(loc="upper left")
    ax_det.legend(loc="upper left")
    fig.suptitle("Realtime Legacy Monitor: RAM / RSS / Detection")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    def _update(_frame_idx):
        if not _is_gui_enabled(cfg_path):
            print("live_plot_gui_enabled turned off -> closing GUI.")
            plt.close(fig)
            return ln_ram, ln_rss, ln_det

        new_points = tail.read_new_points()
        for ts, ram_pct, rss, det in new_points:
            xs.append(ts)
            ram_pcts.append(ram_pct)
            rss_mbs.append(rss)
            det_counts.append(det)

        if not xs:
            return ln_ram, ln_rss, ln_det

        x_idx = list(range(len(xs)))
        labels = [t.strftime("%H:%M:%S") for t in xs]

        ln_ram.set_data(x_idx, list(ram_pcts))
        ln_rss.set_data(x_idx, list(rss_mbs))
        ln_det.set_data(x_idx, list(det_counts))

        for ax in axes:
            ax.set_xlim(0, max(1, len(x_idx) - 1))
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)

        tick_step = max(1, len(x_idx) // 8)
        ticks = list(range(0, len(x_idx), tick_step))
        ax_det.set_xticks(ticks)
        ax_det.set_xticklabels([labels[i] for i in ticks], rotation=30, ha="right")

        return ln_ram, ln_rss, ln_det

    # Keep a strong reference so animation loop keeps running.
    anim = FuncAnimation(fig, _update, interval=1000, blit=False, cache_frame_data=False)
    _ = anim
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

