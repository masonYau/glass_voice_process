"""音频读写与基础电平计算。内部统一使用 16 kHz 单声道 float32，时间轴以 10 ms 为一帧。"""
from __future__ import annotations

import os

import numpy as np

SR = 16000
HOP = 0.01  # 10 ms 帧
HOP_N = int(HOP * SR)


def db(x, eps=1e-10):
    """功率 → dB。"""
    return 10 * np.log10(np.maximum(x, eps))


def to_mono_16k(x: np.ndarray, sr: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1) if x.shape[1] <= 8 else x.mean(axis=0)
    if sr != SR:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    return x.astype(np.float32)


def load_audio(src, sr: int | None = None) -> np.ndarray:
    """读取文件路径或接收 ndarray（需同时给 sr），返回 16 kHz 单声道 float32。"""
    if isinstance(src, (str, os.PathLike)):
        import soundfile as sf
        x, fs = sf.read(str(src), dtype="float32", always_2d=True)
        return to_mono_16k(x, fs)
    if sr is None:
        raise ValueError("传入 ndarray 时必须同时给出采样率 sr")
    return to_mono_16k(np.asarray(src), sr)


def write_wav(path, y: np.ndarray, sr: int = SR):
    import soundfile as sf
    sf.write(str(path), np.clip(y, -1, 1).astype(np.float32), sr, subtype="PCM_16")


def n_frames(n_samples: int) -> int:
    return 1 + (n_samples - 1) // HOP_N


def frame_rms_db(x: np.ndarray, win: float = 0.03) -> np.ndarray:
    """每 10 ms 一帧的 RMS 电平 (dBFS)。"""
    n = int(win * SR)
    pad = np.pad(x, (n // 2, n // 2))
    nf = n_frames(len(x))
    idx = np.arange(nf)[:, None] * HOP_N + np.arange(n)[None, :]
    idx = np.minimum(idx, len(pad) - 1)
    return db(np.mean(pad[idx] ** 2, axis=1))
