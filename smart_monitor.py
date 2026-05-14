"""
Smart runtime monitoring for memory/CPU diagnostics.

Output per app run:
- monitor.log (human readable)
"""

from __future__ import annotations

import datetime as dt
import logging
import logging.handlers
import os
import signal
import threading
import time
from typing import Any, Callable, Dict, Optional

import psutil


class SmartMonitor:
    def __init__(self, base_dir: str, interval_sec: float = 5.0, ram_kill_pct: float | None = None):
        self.base_dir = base_dir
        self.interval_sec = max(1.0, float(interval_sec))
        self.log_path = os.path.join(base_dir, "monitor.log")
        # If set, kill the process when system RAM usage exceeds this percentage.
        self.ram_kill_pct: float | None = float(ram_kill_pct) if ram_kill_pct is not None else None

        self._proc = psutil.Process()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._logger: Optional[logging.Logger] = None
        self._events: list[str] = []
        self._events_lock = threading.Lock()
        self._task_seq = 0
        self._task_seq_lock = threading.Lock()
        self._tasks: Dict[int, Dict[str, Any]] = {}
        self._tasks_lock = threading.Lock()
        self._extra_provider = None
        # If set, invoked (once) when RAM exceeds ram_kill_pct instead of immediate SIGTERM.
        self._ram_kill_handler: Optional[Callable[[Dict[str, Any]], None]] = None

    def set_ram_kill_handler(self, fn: Optional[Callable[[Dict[str, Any]], None]]):
        """Register cleanup for RAM threshold; None restores default (SIGTERM only)."""
        self._ram_kill_handler = fn

    def set_extra_provider(self, fn):
        """
        Provide extra runtime fields to append to each main status row.

        fn must be a callable that returns a dict of {key: value}.
        Failures inside fn are swallowed so monitoring never breaks the app.
        """
        self._extra_provider = fn

    def _now(self) -> str:
        return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _read_gpu_mb(self) -> float:
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if "NvMapMemUsed" in line:
                        return float(int(line.split()[1]) / 1024.0)
        except Exception:
            pass
        return -1.0

    def _read_temp_c(self) -> float:
        try:
            with open("/sys/devices/virtual/thermal/thermal_zone0/temp", "r") as f:
                return float(int(f.read().strip()) / 1000.0)
        except Exception:
            return -1.0

    def _snapshot(self) -> Dict[str, Any]:
        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()
        snap = {
            "ts": self._now(),
            "cpu_pct_system": float(psutil.cpu_percent(interval=None)),
            "cpu_pct_process": float(self._proc.cpu_percent(interval=None)),
            "ram_used_mb": float(vm.used / (1024 * 1024)),
            "ram_total_mb": float(vm.total / (1024 * 1024)),
            "ram_pct": float(vm.percent),
            "swap_used_mb": float(sm.used / (1024 * 1024)),
            "rss_mb": float(self._proc.memory_info().rss / (1024 * 1024)),
            "threads": int(threading.active_count()),
            "gpu_mb": float(self._read_gpu_mb()),
            "temp_c": float(self._read_temp_c()),
        }
        if self._extra_provider is not None:
            try:
                extra = self._extra_provider()
                if isinstance(extra, dict):
                    snap["extra"] = extra
            except Exception:
                pass
        return snap

    def _init_files(self):
        open(self.log_path, "w").close()

    def start(self):
        if self._running:
            return
        self._init_files()
        self._logger = logging.getLogger("smart_monitor")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        self._logger.handlers.clear()
        fh = logging.handlers.RotatingFileHandler(self.log_path, maxBytes=50 * 1024 * 1024, backupCount=1)
        fh.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(fh)
        self._running = True
        self._logger.info("=" * 80)
        self._logger.info(f"MONITOR START {self._now()} | interval={self.interval_sec}s")
        self._logger.info("=" * 80)

        # Prime cpu_percent so next samples are meaningful.
        psutil.cpu_percent(interval=None)
        self._proc.cpu_percent(interval=None)

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        if not self._running:
            return
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._logger:
            self._logger.info("=" * 80)
            self._logger.info(f"MONITOR STOP {self._now()}")
            self._logger.info("=" * 80)

    def event(self, message: str):
        with self._events_lock:
            self._events.append(f"[{self._now()}] {message}")

    def log(self, message: str):
        """Write a line immediately to monitor.log (not queued, not delayed)."""
        if self._logger:
            self._logger.info(message)

    def task_start(self, name: str, extra: str = "") -> int:
        snap = self._snapshot()
        with self._task_seq_lock:
            self._task_seq += 1
            task_id = self._task_seq
        with self._tasks_lock:
            self._tasks[task_id] = {"name": name, "start": snap}
        self._write_task_row(task_id, name, "start", snap, duration_ms=0.0, rss_delta_mb=0.0, extra=extra)
        return task_id

    def task_end(self, task_id: int, extra: str = ""):
        end = self._snapshot()
        with self._tasks_lock:
            st = self._tasks.pop(task_id, None)
        if not st:
            return
        start = st["start"]
        duration_ms = max(0.0, (self._ts_parse(end["ts"]) - self._ts_parse(start["ts"])) * 1000.0)
        rss_delta = float(end["rss_mb"] - start["rss_mb"])
        self._write_task_row(task_id, st["name"], "end", end, duration_ms, rss_delta, extra)

    def _ts_parse(self, txt: str) -> float:
        return dt.datetime.strptime(txt, "%Y-%m-%d %H:%M:%S").timestamp()

    def _write_task_row(
        self,
        task_id: int,
        name: str,
        stage: str,
        snap: Dict[str, Any],
        duration_ms: float,
        rss_delta_mb: float,
        extra: str,
    ):
        if self._logger:
            self._logger.info(
                f"TASK {stage.upper()} #{task_id} {name} | rss={snap['rss_mb']:.0f}MB "
                f"delta={rss_delta_mb:+.0f}MB cpu={snap['cpu_pct_process']:.0f}% "
                f"dur={duration_ms:.0f}ms {extra}"
            )

    def _loop(self):
        while self._running:
            snap = self._snapshot()
            if self._logger:
                extra_txt = ""
                extra = snap.get("extra")
                if isinstance(extra, dict) and extra:
                    # Stable ordering for grepability
                    parts = []
                    for k in sorted(extra.keys()):
                        v = extra.get(k)
                        parts.append(f"{k}={v}")
                    extra_txt = " | " + " ".join(parts)
                self._logger.info(
                    f"{snap['ts']} | cpu={snap['cpu_pct_system']:.0f}% proc={snap['cpu_pct_process']:.0f}% "
                    f"| ram={snap['ram_used_mb']:.0f}/{snap['ram_total_mb']:.0f}MB ({snap['ram_pct']:.0f}%) "
                    f"| rss={snap['rss_mb']:.0f}MB | swap={snap['swap_used_mb']:.0f}MB "
                    f"| gpu={snap['gpu_mb']:.0f}MB | temp={snap['temp_c']:.0f}C | threads={snap['threads']}{extra_txt}"
                )

            with self._events_lock:
                events = self._events[:]
                self._events.clear()
            if self._logger and events:
                for ev in events:
                    self._logger.info(f"EVENT {ev}")

            # RAM kill guard: graceful handler or SIGTERM when RAM exceeds threshold.
            if self.ram_kill_pct is not None and snap["ram_pct"] >= self.ram_kill_pct:
                kill_msg = (
                    f"RAM beyond {self.ram_kill_pct:.0f}% usage — RAM {snap['ram_pct']:.1f}% "
                    f"({snap['ram_used_mb']:.0f}/{snap['ram_total_mb']:.0f}MB) — stopping inference "
                    f"and shutting down"
                )
                if self._logger:
                    self._logger.info(kill_msg)
                print(kill_msg, flush=True)
                # Stop the monitor loop before running handler so we never double-fire.
                self._running = False
                handler = self._ram_kill_handler
                if handler is not None:
                    try:
                        handler(snap)
                    except Exception:
                        logging.exception("ram_kill_handler failed; sending SIGTERM")
                        os.kill(os.getpid(), signal.SIGTERM)
                else:
                    os.kill(os.getpid(), signal.SIGTERM)
                break

            time.sleep(self.interval_sec)
