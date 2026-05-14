#!/usr/bin/env python3
"""
Log visualizer for Jetson runtime diagnostics.

Parses:
  - monitor.log
  - contig_diag.log

Shows:
  - RAM/RSS/SWAP trends
  - CMA/NvMap/contiguous (buddy high-order) trends
  - CPU/TEMP trends
  - Auto-generated insight summary

Usage:
  python3 scripts/log_insight_gui.py
  python3 scripts/log_insight_gui.py --monitor monitor.log --contig contig_diag.log
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sys
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt


MONITOR_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \| "
    r"cpu=(?P<cpu>-?\d+)% proc=(?P<proc>-?\d+)% \| "
    r"ram=(?P<ram_used>\d+)/(?P<ram_total>\d+)MB \((?P<ram_pct>\d+)%\) \| "
    r"rss=(?P<rss>-?\d+)MB \| swap=(?P<swap>-?\d+)MB \| "
    r"gpu=(?P<gpu>-?\d+)MB \| temp=(?P<temp>-?\d+)C \| threads=(?P<threads>\d+)"
)

EVENT_RE = re.compile(r"^EVENT \[(?P<ts>[^\]]+)\] (?P<msg>.+)$")

CONTIG_LINE_RE = re.compile(
    r"^(?P<ts>\d{2}:\d{2}:\d{2}) \| "
    r"avail=(?P<avail>-?\d+)MB free=(?P<free>-?\d+)MB "
    r"nvmap=(?P<nvmap>-?\d+)MB cma=(?P<cma_free>-?\d+)/(?P<cma_total>-?\d+)MB "
    r"swap_free=(?P<swap_free>-?\d+)MB rss=(?P<rss>-?\d+)MB \| "
    r"buddy DMA:o8=(?P<dma_o8>-?\d+),o9=(?P<dma_o9>-?\d+),o10=(?P<dma_o10>-?\d+) \| "
    r"Normal:o8=(?P<norm_o8>-?\d+),o9=(?P<norm_o9>-?\d+),o10=(?P<norm_o10>-?\d+)"
)


@dataclass
class MonitorPoint:
    ts: dt.datetime
    cpu: int
    proc: int
    ram_used: int
    ram_total: int
    ram_pct: int
    rss: int
    swap: int
    gpu: int
    temp: int
    threads: int


@dataclass
class EventPoint:
    ts: dt.datetime
    msg: str


@dataclass
class ContigPoint:
    ts: dt.datetime
    avail: int
    free: int
    nvmap: int
    cma_free: int
    cma_total: int
    swap_free: int
    rss: int
    dma_o8: int
    dma_o9: int
    dma_o10: int
    norm_o8: int
    norm_o9: int
    norm_o10: int


def parse_monitor(path: str) -> tuple[List[MonitorPoint], List[EventPoint]]:
    points: List[MonitorPoint] = []
    events: List[EventPoint] = []
    if not os.path.exists(path):
        return points, events

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            m = MONITOR_LINE_RE.match(line)
            if m:
                points.append(
                    MonitorPoint(
                        ts=dt.datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S"),
                        cpu=int(m.group("cpu")),
                        proc=int(m.group("proc")),
                        ram_used=int(m.group("ram_used")),
                        ram_total=int(m.group("ram_total")),
                        ram_pct=int(m.group("ram_pct")),
                        rss=int(m.group("rss")),
                        swap=int(m.group("swap")),
                        gpu=int(m.group("gpu")),
                        temp=int(m.group("temp")),
                        threads=int(m.group("threads")),
                    )
                )
                continue

            ev = EVENT_RE.match(line)
            if ev:
                ts_raw = ev.group("ts")
                try:
                    ts = dt.datetime.strptime(ts_raw, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    # Some logs may only carry HH:MM:SS
                    try:
                        t = dt.datetime.strptime(ts_raw, "%H:%M:%S").time()
                        day = points[-1].ts.date() if points else dt.date.today()
                        ts = dt.datetime.combine(day, t)
                    except ValueError:
                        continue
                events.append(EventPoint(ts=ts, msg=ev.group("msg")))

    return points, events


def parse_contig(path: str, monitor_points: List[MonitorPoint]) -> List[ContigPoint]:
    points: List[ContigPoint] = []
    if not os.path.exists(path):
        return points

    base_date = monitor_points[0].ts.date() if monitor_points else dt.date.today()
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            m = CONTIG_LINE_RE.match(line)
            if not m:
                continue
            t = dt.datetime.strptime(m.group("ts"), "%H:%M:%S").time()
            ts = dt.datetime.combine(base_date, t)
            points.append(
                ContigPoint(
                    ts=ts,
                    avail=int(m.group("avail")),
                    free=int(m.group("free")),
                    nvmap=int(m.group("nvmap")),
                    cma_free=int(m.group("cma_free")),
                    cma_total=int(m.group("cma_total")),
                    swap_free=int(m.group("swap_free")),
                    rss=int(m.group("rss")),
                    dma_o8=int(m.group("dma_o8")),
                    dma_o9=int(m.group("dma_o9")),
                    dma_o10=int(m.group("dma_o10")),
                    norm_o8=int(m.group("norm_o8")),
                    norm_o9=int(m.group("norm_o9")),
                    norm_o10=int(m.group("norm_o10")),
                )
            )
    return points


def summarize(monitor: List[MonitorPoint], contig: List[ContigPoint], events: List[EventPoint]) -> str:
    if not monitor and not contig:
        return "No parseable data found in monitor/contig logs."

    lines: List[str] = []

    if monitor:
        peak_rss = max(p.rss for p in monitor)
        peak_ram = max(p.ram_pct for p in monitor)
        peak_cpu = max(p.cpu for p in monitor)
        peak_temp = max(p.temp for p in monitor)
        lines.append(f"- Peak process RSS: {peak_rss} MB")
        lines.append(f"- Peak system RAM: {peak_ram}%")
        lines.append(f"- Peak CPU: {peak_cpu}% | Peak temp: {peak_temp} C")

    if contig:
        min_cma = min(p.cma_free for p in contig)
        low_cma_samples = sum(1 for p in contig if p.cma_free <= 8)
        high_order_zero = sum(
            1
            for p in contig
            if p.dma_o8 == 0 and p.dma_o9 == 0 and p.dma_o10 == 0
            and p.norm_o8 == 0 and p.norm_o9 == 0 and p.norm_o10 == 0
        )
        lines.append(f"- Minimum CmaFree: {min_cma} MB")
        lines.append(f"- Samples with CmaFree <= 8MB: {low_cma_samples}/{len(contig)}")
        lines.append(f"- Samples with all high-order buddy blocks zero: {high_order_zero}/{len(contig)}")
        if min_cma <= 1 and high_order_zero > 0:
            lines.append("- Interpretation: strong contiguous-memory starvation signature (likely NVMM/NvMap allocation failures).")

    if events:
        interesting = [e for e in events if any(k in e.msg for k in ("MODE_CHANGE", "RECORD_START", "RECORD_STOP", "INFER_WORKER_START", "MODEL_LOADING"))]
        if interesting:
            lines.append("- Key events observed:")
            for e in interesting[:6]:
                lines.append(f"    {e.ts.strftime('%H:%M:%S')}  {e.msg}")

    return "\n".join(lines)


def plot_logs(monitor: List[MonitorPoint], contig: List[ContigPoint], events: List[EventPoint], title: str):
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    ax1, ax2, ax3, ax4 = axes.flatten()

    # Panel 1: RAM / RSS / SWAP
    if monitor:
        t = [p.ts for p in monitor]
        ax1.plot(t, [p.ram_used for p in monitor], label="RAM Used (MB)", linewidth=2)
        ax1.plot(t, [p.rss for p in monitor], label="Process RSS (MB)", linewidth=2)
        ax1.plot(t, [p.swap for p in monitor], label="Swap Used (MB)", linewidth=1.8)
        ax1.set_title("System + Process Memory")
        ax1.set_ylabel("MB")
        ax1.legend(loc="upper left")
        ax1.grid(alpha=0.25)
    else:
        ax1.set_title("System + Process Memory (no monitor data)")

    # Panel 2: Contiguous allocator view
    if contig:
        t = [p.ts for p in contig]
        ax2.plot(t, [p.cma_free for p in contig], label="CmaFree (MB)", linewidth=2)
        ax2.plot(t, [p.cma_total for p in contig], label="CmaTotal (MB)", linestyle="--", linewidth=1.2)
        ax2.plot(t, [p.avail for p in contig], label="MemAvailable (MB)", linewidth=1.6)
        ax2.plot(t, [p.nvmap for p in contig], label="NvMap (MB)", linewidth=1.6)
        ax2.set_title("Contiguous Memory vs Available Memory")
        ax2.set_ylabel("MB")
        ax2.legend(loc="upper right")
        ax2.grid(alpha=0.25)
    else:
        ax2.set_title("Contiguous Memory (no contig data)")

    # Panel 3: Buddy high-order blocks (contiguous block availability proxy)
    if contig:
        t = [p.ts for p in contig]
        ax3.plot(t, [p.dma_o8 for p in contig], label="DMA o8", linewidth=1.8)
        ax3.plot(t, [p.dma_o9 for p in contig], label="DMA o9", linewidth=1.8)
        ax3.plot(t, [p.dma_o10 for p in contig], label="DMA o10", linewidth=1.8)
        ax3.plot(t, [p.norm_o8 for p in contig], label="Normal o8", linewidth=1.4)
        ax3.plot(t, [p.norm_o9 for p in contig], label="Normal o9", linewidth=1.4)
        ax3.plot(t, [p.norm_o10 for p in contig], label="Normal o10", linewidth=1.4)
        ax3.set_title("High-Order Buddy Blocks")
        ax3.set_ylabel("Block Count")
        ax3.legend(loc="upper right", ncol=2, fontsize=8)
        ax3.grid(alpha=0.25)
    else:
        ax3.set_title("Buddy Blocks (no contig data)")

    # Panel 4: CPU/TEMP + summary text
    if monitor:
        t = [p.ts for p in monitor]
        ax4.plot(t, [p.cpu for p in monitor], label="CPU %", linewidth=2)
        ax4.plot(t, [p.proc for p in monitor], label="Proc CPU %", linewidth=1.8)
        ax4.plot(t, [p.temp for p in monitor], label="Temp C", linewidth=1.8)
        ax4.set_ylabel("Value")
        ax4.grid(alpha=0.25)
        ax4.legend(loc="upper left")
        ax4.set_title("CPU / Process CPU / Temperature")
    else:
        ax4.set_title("CPU / Temperature (no monitor data)")

    # Event markers on top two panels
    key_events = [e for e in events if any(k in e.msg for k in ("MODE_CHANGE", "RECORD_START", "RECORD_STOP", "INFER_WORKER_START"))]
    for ev in key_events[:10]:
        for ax in (ax1, ax2):
            ax.axvline(ev.ts, color="red", alpha=0.15, linewidth=1)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])

    summary = summarize(monitor, contig, events)
    fig.text(0.01, 0.01, summary, fontsize=9, family="monospace", va="bottom")

    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize monitor.log + contig_diag.log")
    parser.add_argument("--monitor", default="monitor.log", help="Path to monitor.log")
    parser.add_argument("--contig", default="contig_diag.log", help="Path to contig_diag.log")
    parser.add_argument("--save", default="", help="Optional path to save PNG snapshot")
    args = parser.parse_args()

    monitor_points, events = parse_monitor(args.monitor)
    contig_points = parse_contig(args.contig, monitor_points)

    if not monitor_points and not contig_points:
        print("No parseable data found. Check file paths and log formats.")
        sys.exit(1)

    title = f"Runtime Log Insights: {os.path.basename(args.monitor)} + {os.path.basename(args.contig)}"
    fig = plot_logs(monitor_points, contig_points, events, title=title)

    if args.save:
        fig.savefig(args.save, dpi=160)
        print(f"Saved snapshot: {args.save}")

    plt.show()


if __name__ == "__main__":
    main()
