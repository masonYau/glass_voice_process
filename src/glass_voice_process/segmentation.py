"""pyannote segmentation-3.0（ONNX，sherpa-onnx 转换）的逐帧解码：语音、重叠、换人点。"""
from __future__ import annotations

import numpy as np

from .audio import HOP, SR, n_frames
from .utils import hysteresis, runs, smooth_mask
from .vad import fast_agc_signal


class PyannoteSegmenter:
    def __init__(self, model_file):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(model_file), providers=["CPUExecutionProvider"])
        meta = self.sess.get_modelmeta().custom_metadata_map
        self.window = int(meta.get("window_size", 160000))
        self.shift = int(meta.get("receptive_field_shift", 270))
        self.rf = int(meta.get("receptive_field_size", 991))

    def __call__(self, x, use_fast_agc=True, step=2.5, min_turn=0.2, max_turn_gap=0.35, batch=8):
        """10 s 窗口、step 秒步长；每窗口约 16.9 ms 一帧、7 类 powerset 概率
        （无人 / 局部说话人 A、B、C / AB、AC、BC 重叠）。汇总到 10 ms 时间轴：
          speech  = 1 − P(无人)；overlap = P(两人同时)；
          change  = 判为"局部主导说话人切换"（前后都 ≥ min_turn 秒、中间静音 ≤ max_turn_gap 秒）的窗口比例。"""
        W, shift, rf = self.window, self.shift, self.rf
        sig = fast_agc_signal(x) if use_fast_agc else np.asarray(x, np.float32)
        S = int(step * SR)
        starts = list(range(0, max(1, len(sig) - W + S), S))
        nf = n_frames(len(x))
        acc = np.zeros((3, nf))
        cnt = np.zeros(nf)
        grid = np.arange(nf) * HOP
        mt, mg = int(min_turn * SR / shift), int(max_turn_gap * SR / shift)
        for b in range(0, len(starts), batch):
            chunk = starts[b:b + batch]
            xs = np.stack([np.pad(sig[s0:s0 + W], (0, W - len(sig[s0:s0 + W]))) for s0 in chunk])
            y = self.sess.run(None, {"x": xs[:, None, :].astype(np.float32)})[0]
            prob = np.exp(y) if y.max() <= 0 else y
            for k, s0 in enumerate(chunk):
                P = prob[k]
                F = P.shape[0]
                tc = (s0 + np.arange(F) * shift + rf / 2) / SR
                speech = 1 - P[:, 0]
                overlap = P[:, 4:7].sum(1)
                act = np.stack([P[:, 1] + P[:, 4] + P[:, 5], P[:, 2] + P[:, 4] + P[:, 6], P[:, 3] + P[:, 5] + P[:, 6]])
                dom = np.where(speech > 0.5, act.argmax(0), -1)
                ch = np.zeros(F)
                rr = [(v, s, e) for v, s, e in runs(dom) if v >= 0]
                for (v1, s1, e1), (v2, s2, e2) in zip(rr, rr[1:]):
                    if v1 != v2 and s2 - e1 <= mg and e1 - s1 >= mt and e2 - s2 >= mt:
                        ch[(e1 + s2) // 2] = 1
                valid = tc <= (s0 + min(W, len(sig) - s0)) / SR
                lo, hi = tc[valid][0], tc[valid][-1]
                gi = np.where((grid >= lo) & (grid <= hi))[0]
                acc[0, gi] += np.interp(grid[gi], tc, speech)
                acc[1, gi] += np.interp(grid[gi], tc, overlap)
                for f in np.where(ch > 0)[0]:
                    c = int(round(tc[f] / HOP))
                    acc[2, max(0, c - 2): c + 3] += 1
                cnt[gi] += 1
        cnt = np.maximum(cnt, 1)
        return {"speech": acc[0] / cnt, "overlap": acc[1] / cnt, "change": np.minimum(1, acc[2] / cnt)}


def speech_mask(seg, onset=0.5, offset=0.5, min_on=0.18, min_off=0.2):
    """语音概率 → 语音帧（滞回 + 最短发言/最短静音，参考 Codex 0.18 s / 0.2 s）。"""
    return smooth_mask(hysteresis(seg["speech"], onset, offset), min_on, min_off)


def change_points(seg, thr=0.5):
    """换人得分 ≥ thr 的每个区间只留峰值 1 帧。"""
    c = seg["change"]
    out = np.zeros(len(c), bool)
    for v, s, e in runs(c >= thr):
        if v:
            out[s + int(np.argmax(c[s:e]))] = True
    return out
