"""评估指标：有真值时的分离准确率；无真值时的电平差、断音缺口/陡降。"""
from __future__ import annotations

from collections import Counter

import numpy as np

from .audio import HOP, frame_rms_db


def _frames(rows, n):
    lab = np.array([""] * n, dtype=object)
    for spk, s, e in rows:
        lab[int(s / HOP): int(e / HOP)] = spk
    return lab


def diarization_scores(truth_rows, segments, n_frames):
    """truth_rows: [(说话人, 起, 止)]，说话人 "wearer" 表示佩戴者。返回检出率、误检率、佩戴者判别、纯度、覆盖度、人数。"""
    T = _frames(truth_rows, n_frames)
    P = _frames([(s.label, s.start, s.end) for s in segments], n_frames)
    both = (T != "") & (P != "")
    tw, pw = T == "wearer", P == "wearer"
    pm = both & ~tw & ~pw
    pur = sum(Counter(T[pm & (P == c)]).most_common(1)[0][1] for c in set(P[pm])) / max(1, pm.sum())
    cov = sum(Counter(P[pm & (T == t)]).most_common(1)[0][1] for t in set(T[pm])) / max(1, pm.sum())
    return {"detection": float(np.mean(P[T != ""] != "")) if (T != "").any() else float("nan"),
            "false_alarm": float(np.mean(P[T == ""] != "")) if (T == "").any() else float("nan"),
            "wearer_accuracy": float(np.mean(tw[both] == pw[both])) if both.any() else float("nan"),
            "partner_purity": float(pur), "partner_coverage": float(cov),
            "partners_pred": len(set(P[(P != "") & ~pw])), "partners_true": len(set(T[(T != "") & ~tw]))}


def active_level(sig, segments, label):
    """某说话人在一条音频里的有效电平：其分段内较响 40% 帧的能量平均 (dBFS)。"""
    lv = frame_rms_db(sig)
    fr = [lv[int(s.start / HOP): int(s.end / HOP)] for s in segments if s.label == label]
    fr = np.concatenate(fr) if fr else np.array([])
    if len(fr) < 10:
        return None
    top = np.sort(fr)[-max(1, int(len(fr) * 0.4)):]
    return float(10 * np.log10(np.mean(10 ** (top / 10))))


def level_gaps(result, segments=None):
    """佩戴者 − 各对话者 的有效电平差（处理前 → 处理后）。segments 默认用结果自己的分段。"""
    segs = segments or result.segments
    out = {}
    w_in, w_out = active_level(result.input, segs, "wearer"), active_level(result.balanced, segs, "wearer")
    for sp in sorted({s.label for s in segs if s.label != "wearer"}):
        p_in, p_out = active_level(result.input, segs, sp), active_level(result.balanced, segs, sp)
        if None not in (w_in, w_out, p_in, p_out):
            out[sp] = (round(w_in - p_in, 1), round(w_out - p_out, 1))
    return out


def cutouts(result, max_gap=5.0, min_drop=15.0):
    """断音统计（需要 frame_gain）：
    gaps  = 两侧都被放大 ≥ 15 dB 的段之间、≤ max_gap 秒的未标注缺口里增益掉 ≥ min_drop 且原始仍有声音的个数
    steep = 50 ms 内增益下降 ≥ min_drop、处于未标注帧且原始仍有声音的次数"""
    g = result.frame_gain
    if g is None:
        return {}
    lv = frame_rms_db(result.input)[:len(g)]
    fl = np.percentile(lv, 10)
    lab = np.zeros(len(g), bool)
    for s in result.segments:
        lab[int(s.start / HOP): int(s.end / HOP)] = True
    gaps = 0
    ss = sorted(result.segments, key=lambda s: s.start)
    for a, b in zip(ss, ss[1:]):
        if a.gain_db < 15 or b.gain_db < 15:
            continue
        f0, f1 = int(a.end / HOP), int(b.start / HOP)
        if f1 - f0 < 5 or (f1 - f0) * HOP > max_gap:
            continue
        if min(a.gain_db, b.gain_db) - g[f0:f1].min() >= min_drop and np.mean(lv[f0:f1] > fl + 6) >= 0.2:
            gaps += 1
    dg = np.r_[np.zeros(5), g[:-5] - g[5:]]
    st = (dg >= min_drop) & ~lab & (lv > fl + 6)
    return {"gaps": gaps, "steep": int(np.sum(st[1:] & ~st[:-1]))}
