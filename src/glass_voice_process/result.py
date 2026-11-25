"""处理结果与保存。"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .audio import SR, write_wav
from .utils import Segment


@dataclass
class Result:
    method: str
    input: np.ndarray                  # 16 kHz 单声道原始信号
    balanced: np.ndarray               # 平衡后的完整混音
    tracks: dict                       # {"wearer": ndarray, "partner_1": ndarray, ...}（与原始等长、时间对齐）
    segments: list                     # [Segment]
    frame_gain: np.ndarray | None = None   # 每 10 ms 的增益 dB（v5 为输出/输入帧电平比估算）
    stats: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    sr: int = SR

    @property
    def speakers(self):
        return sorted(self.tracks, key=lambda s: (s != "wearer", s))

    def summary(self) -> dict:
        out = {}
        for sp in self.speakers:
            ss = [s for s in self.segments if s.label == sp]
            out[sp] = {"segments": len(ss), "seconds": round(sum(s.end - s.start for s in ss), 2),
                       "level_db": round(float(np.median([s.level_db for s in ss])), 2) if ss else None,
                       "gain_db": round(float(np.median([s.gain_db for s in ss])), 2) if ss else None}
        return out

    def save(self, out_dir, name: str = "output", plot: bool = False, save_tracks: bool = True) -> dict:
        """写出 {name}_balanced.wav、{name}_{说话人}.wav、{name}_segments.csv、{name}_result.json（和可选 report.png）。"""
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        files = {"balanced": d / f"{name}_balanced.wav"}
        write_wav(files["balanced"], self.balanced, self.sr)
        if save_tracks:
            for sp, y in self.tracks.items():
                files[sp] = d / f"{name}_{sp}.wav"
                write_wav(files[sp], y, self.sr)
        files["segments"] = d / f"{name}_segments.csv"
        write_segments(files["segments"], self.segments)
        files["result"] = d / f"{name}_result.json"
        files["result"].write_text(json.dumps({"method": self.method, "summary": self.summary(), "stats": self.stats,
                                               "config": self.config}, ensure_ascii=False, indent=2, default=float),
                                   encoding="utf-8")
        if plot:
            from .report import plot_report
            files["report"] = d / f"{name}_report.png"
            plot_report(self, files["report"])
        return files


def write_segments(path, segs):
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["speaker", "start_s", "end_s", "level_dBFS", "gain_dB", "overlap"])
        for s in segs:
            w.writerow([s.label, f"{s.start:.2f}", f"{s.end:.2f}", f"{s.level_db:.1f}", f"{s.gain_db:+.1f}",
                        "" if s.overlap is None else f"{s.overlap:.2f}"])


def read_segments(path):
    with open(path, encoding="utf-8-sig") as fh:
        return [Segment(float(r["start_s"]), float(r["end_s"]), r["speaker"], float(r.get("level_dBFS") or 0),
                        float(r.get("gain_dB") or 0), float(r["overlap"]) if r.get("overlap") else None)
                for r in csv.DictReader(fh)]
