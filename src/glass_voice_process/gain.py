"""v3/v4 的增益流程：分段增益 + 慢速 AGC、缺口桥接、慢释放/提前起音、底噪约束、瞬态上限、尖峰压制、限幅。"""
from __future__ import annotations

import numpy as np

from .audio import HOP, SR, db
from .utils import runs


def _level_of(fr):
    """有效语音电平：较响 40% 帧的能量平均。"""
    top = np.sort(fr)[-max(1, int(len(fr) * 0.4)):]
    return float(db(np.mean(10 ** (top / 10))))


def apply_gains(x, segs, lv, target_db=-23.0, max_boost=35.0, max_cut=-15.0, ramp=0.08, block=2.0, hold=0.8,
                noise_db=None, noise_ceiling=-45.0, transient_margin=8.0, bridge_gap=2.5, bridge_min_gain=6.0,
                release=0.4, attack=0.1):
    """返回 (输出信号, 每采样点增益 dB, 每帧增益 dB, 每帧是否属于说话人段)。会写入 seg.level_db / seg.gain_db。"""
    if noise_db is not None:  # 放大后底噪不超过 noise_ceiling
        max_boost = float(np.clip(noise_ceiling - noise_db, 10.0, max_boost))
    nf = len(lv)
    g = np.zeros(nf)
    has = np.zeros(nf, bool)
    B = int(block / HOP)
    for s in segs:
        f0, f1 = int(s.start / HOP), min(nf, int(s.end / HOP))
        fr = lv[f0:f1]
        if len(fr) == 0:
            continue
        s.level_db = _level_of(fr)
        s.gain_db = float(np.clip(target_db - s.level_db, max_cut, max_boost))
        nb = max(1, round((f1 - f0) / B))
        if nb == 1:
            g[f0:f1] = s.gain_db
        else:  # 长段按约 block 秒分块（慢速 AGC），块增益不偏离整段 ±8 dB
            edges = np.linspace(f0, f1, nb + 1).astype(int)
            cen = (edges[:-1] + edges[1:]) / 2
            bg = np.clip([np.clip(target_db - _level_of(lv[a:b]), max_cut, max_boost)
                          for a, b in zip(edges[:-1], edges[1:])], s.gain_db - 8, s.gain_db + 8)
            g[f0:f1] = np.interp(np.arange(f0, f1), cen, bg)
        has[f0:f1] = True
    # 未标注帧：短停顿 (< hold) 或 两侧都被放大且缺口 ≤ bridge_gap → 两侧之间线性过渡；否则 0 dB
    for val, s0, e0 in runs(has):
        if val:
            continue
        inner = s0 > 0 and e0 < nf
        dur = (e0 - s0) * HOP
        if inner and (dur < hold or (dur <= bridge_gap and min(g[s0 - 1], g[e0]) > bridge_min_gain)):
            g[s0:e0] = np.linspace(g[s0 - 1], g[e0], e0 - s0)
        else:
            g[s0:e0] = 0.0
    k = max(1, int(ramp / HOP))
    g = np.convolve(np.pad(g, (k, k), mode="edge"), np.ones(k) / k, mode="same")[k:-k]
    # 未标注帧的速率限制：放大段后慢释放（release 秒 30 dB），下一段前提前起音（attack 秒 30 dB）
    if release and release > 0:
        rel = 30.0 * HOP / release
        att = 30.0 * HOP / max(attack, HOP)
        for i in range(1, nf):
            if not has[i] and g[i - 1] > 0 and g[i] < g[i - 1] - rel:
                g[i] = g[i - 1] - rel
        for i in range(nf - 2, -1, -1):
            if not has[i]:
                g[i] = min(max(g[i], g[i + 1] - att), g[i + 1] + att)
    # 瞬态上限：被放大的帧输出不超过 target + transient_margin
    if transient_margin is not None:
        from scipy.ndimage import maximum_filter1d
        cap = target_db + transient_margin - maximum_filter1d(lv, 3)
        cap = np.minimum.reduce([cap, np.roll(cap, 1), np.roll(cap, -1)])
        g = np.where(g > 0, np.minimum(g, np.maximum(cap, 0.0)), g)
    gs = np.interp(np.arange(len(x)) / SR, np.arange(nf) * HOP, g)
    return x * 10 ** (gs / 20), gs, g, has


def suppress_clicks(y, gs, min_gain_db=6.0, crest=4.0, blk=0.005, ctx=0.1, abs_db=-14.0):
    """只在被放大 (> min_gain_db) 的区域压制 < 10 ms 的尖峰：5 ms 块峰值超过 ±100 ms 中位数的 crest 倍
    且超过 abs_db 时压回。返回 (信号, 压制块数)。"""
    from scipy.ndimage import median_filter
    b = int(blk * SR)
    nb = len(y) // b
    if nb < 3:
        return y, 0
    pk = np.abs(y[: nb * b]).reshape(nb, b).max(1)
    ref = np.maximum(crest * median_filter(pk, size=2 * int(ctx / blk) + 1, mode="nearest"), 10 ** (abs_db / 20))
    hit = (gs[: nb * b].reshape(nb, b).mean(1) > min_gain_db) & (pk > ref)
    gc = np.ones(nb)
    gc[hit] = ref[hit] / pk[hit]
    gc = np.minimum.reduce([gc, np.roll(gc, 1), np.roll(gc, -1)])
    return y * np.interp(np.arange(len(y)), np.arange(nb) * b + b / 2, gc), int(hit.sum())


def limiter(y, ceiling_db=-1.0, release=0.05):
    """前瞻 5 ms + 一阶释放的峰值限幅。"""
    from scipy.ndimage import minimum_filter1d
    from scipy.signal import lfilter
    c = 10 ** (ceiling_db / 20)
    need = np.minimum(1.0, c / np.maximum(np.abs(y), 1e-9))
    need = minimum_filter1d(need, int(0.005 * SR))
    alpha = np.exp(-1 / (release * SR))
    sm = lfilter([1 - alpha], [1, -alpha], need - 1) + 1
    return y * np.minimum(sm, need)
