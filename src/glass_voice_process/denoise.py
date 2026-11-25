"""平稳谱门限降噪（noisereduce）。噪声样本取 WebRTC 判为非语音且电平处于全段较低一半的帧。"""
from __future__ import annotations

import numpy as np

from .audio import HOP_N, SR, frame_rms_db
from .vad import webrtc_vad


def noise_mask(x):
    """返回"非噪声"帧（True = 可能是语音或较响，不用于噪声估计）。"""
    lv = frame_rms_db(x)
    return webrtc_vad(x) | (lv > np.percentile(lv, 50))


def denoise(x, strength=0.85, speech_mask=None):
    if strength <= 0:
        return np.asarray(x, np.float32)
    import noisereduce as nr
    if speech_mask is None:
        speech_mask = noise_mask(x)
    nonspeech = np.repeat(~speech_mask, HOP_N)[: len(x)]
    noise_clip = x[: len(nonspeech)][nonspeech] if nonspeech.sum() > SR * 0.5 else None
    y = nr.reduce_noise(y=x, sr=SR, y_noise=noise_clip, stationary=True, prop_decrease=strength, n_fft=512)
    return y.astype(np.float32)
