"""分段数据结构与掩码工具。"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .audio import HOP


@dataclass
class Segment:
    """一个说话人段。label: "wearer" / "partner_1" / ..."""
    start: float
    end: float
    label: str = ""
    level_db: float = 0.0     # 该段原始（处理前）有效电平 dBFS
    gain_db: float = 0.0      # 该段所加增益 dB
    overlap: float | None = None  # pyannote 判为两人重叠的帧比例；未运行 pyannote 时为 None

    def to_dict(self):
        return asdict(self)


def runs(m):
    """把序列切成连续相同值的区间 [(值, 起, 止)]。"""
    m = np.asarray(m)
    if len(m) == 0:
        return []
    d = np.flatnonzero(m[1:] != m[:-1]) + 1
    starts = np.r_[0, d]
    ends = np.r_[d, len(m)]
    return [(m[s], int(s), int(e)) for s, e in zip(starts, ends)]


def smooth_mask(m, min_on: float, min_off: float):
    """填短间隙（< min_off 秒），去短语音（< min_on 秒）。"""
    m = np.asarray(m, bool).copy()
    for val, s, e in runs(m):
        if not val and (e - s) * HOP < min_off and s > 0 and e < len(m):
            m[s:e] = True
    for val, s, e in runs(m):
        if val and (e - s) * HOP < min_on:
            m[s:e] = False
    return m


def hysteresis(p, onset: float, offset: float, onset_alt=None, p_alt=None):
    """滞回判决：未在段内时 p > onset（或 p_alt > onset_alt）起段；段内 p > offset 继续。"""
    out = np.zeros(len(p), bool)
    on = False
    for i in range(len(p)):
        if on:
            on = p[i] > offset
        else:
            on = p[i] > onset or (p_alt is not None and p_alt[i] > onset_alt)
        out[i] = on
    return out


def merge_short(segs, min_dur: float = 0.3):
    """短于 min_dur 的段并入前一段；同一说话人、间隔 < 0.2 s 的相邻段合并。"""
    out = []
    for s in segs:
        if out and (s.end - s.start) < min_dur and abs(s.start - out[-1].end) < 0.05:
            out[-1].end = s.end
        elif out and out[-1].label == s.label and s.start - out[-1].end < 0.2:
            out[-1].end = s.end
        else:
            out.append(s)
    return out


def pad_to(a, n, fill=False):
    a = np.asarray(a)
    if len(a) >= n:
        return a[:n]
    return np.r_[a, np.full(n - len(a), fill, dtype=a.dtype)]
