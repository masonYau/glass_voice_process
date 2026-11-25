"""与 torchaudio.compliance.kaldi.fbank 等价的 numpy 实现（CAM++ 的输入特征），免去 torch/torchaudio 依赖。
默认参数与 kaldi 一致：25 ms 窗 / 10 ms 移、去直流、预加重 0.97、povey 窗、FFT 补到 2 的幂、
功率谱、梅尔 20 Hz ~ Nyquist、取对数；dither 固定为 0（结果可复现）。"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

_EPS = np.finfo(np.float32).eps


@lru_cache(maxsize=4)
def _mel_banks(num_bins: int, padded: int, sr: int, low: float = 20.0, high: float = 0.0):
    nyq = sr / 2
    if high <= 0:
        high = nyq + high
    mel = lambda f: 1127.0 * np.log(1.0 + f / 700.0)
    n_fft_bins = padded // 2
    fft_w = sr / padded
    mlow, mhigh = mel(low), mel(high)
    delta = (mhigh - mlow) / (num_bins + 1)
    b = np.arange(num_bins)[:, None]
    left, center, right = mlow + b * delta, mlow + (b + 1) * delta, mlow + (b + 2) * delta
    m = mel(fft_w * np.arange(n_fft_bins))[None, :]
    up = (m - left) / (center - left)
    down = (right - m) / (right - center)
    banks = np.maximum(0.0, np.minimum(up, down))
    return np.pad(banks, ((0, 0), (0, 1))).astype(np.float64)  # 补 Nyquist 列


def fbank(x: np.ndarray, sr: int = 16000, num_mel_bins: int = 80, frame_length_ms: float = 25.0,
          frame_shift_ms: float = 10.0, preemph: float = 0.97) -> np.ndarray:
    """返回 (帧数, num_mel_bins) 的对数梅尔能量。"""
    x = np.asarray(x, dtype=np.float64)
    L, S = int(sr * frame_length_ms / 1000), int(sr * frame_shift_ms / 1000)
    if len(x) < L:
        return np.zeros((0, num_mel_bins), np.float32)
    m = 1 + (len(x) - L) // S
    idx = np.arange(m)[:, None] * S + np.arange(L)[None, :]
    fr = x[idx]
    fr = fr - fr.mean(1, keepdims=True)                     # 去直流
    prev = np.concatenate([fr[:, :1], fr[:, :-1]], axis=1)  # 预加重（首点复制）
    fr = fr - preemph * prev
    win = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(L) / (L - 1))) ** 0.85  # povey
    fr = fr * win
    padded = 1 << int(np.ceil(np.log2(L)))
    spec = np.abs(np.fft.rfft(fr, padded)) ** 2
    mel = spec @ _mel_banks(num_mel_bins, padded, sr).T
    return np.log(np.maximum(mel, _EPS)).astype(np.float32)
