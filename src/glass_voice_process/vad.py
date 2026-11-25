"""语音检测：WebRTC（仅用于噪声估计）、Silero（ONNX 直接推理，免 torch）、双路检测、谐波检查。"""
from __future__ import annotations

import numpy as np

from .audio import HOP, HOP_N, SR, frame_rms_db, n_frames
from .utils import runs, smooth_mask


# ---------------------------------------------------------------- WebRTC
def webrtc_vad(x, aggressiveness=2, min_speech=0.25, min_gap=0.3, rel_floor_db=12):
    """WebRTC VAD + 能量门限（比 10% 分位底噪高 rel_floor_db）+ 平滑。返回每 10 ms 一个 bool。"""
    import webrtcvad
    v = webrtcvad.Vad(aggressiveness)
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    fl = int(0.03 * SR)
    out = np.zeros(n_frames(len(x)), bool)
    for i in range(0, len(pcm) - fl, fl):
        f0 = i // HOP_N
        out[f0: f0 + 3] = v.is_speech(pcm[i: i + fl].tobytes(), SR)
    lv = frame_rms_db(x)
    out &= lv > np.percentile(lv, 10) + rel_floor_db
    return smooth_mask(out, min_speech, min_gap)


# ---------------------------------------------------------------- Silero（ONNX）
class SileroVAD:
    """Silero VAD v5 的 ONNX 推理（与 silero-vad 包的 OnnxWrapper 逐块等价：512 点 + 64 点上下文，带 RNN 状态）。"""

    def __init__(self, model_file):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(str(model_file), sess_options=opts, providers=["CPUExecutionProvider"])

    def probs(self, sig: np.ndarray) -> np.ndarray:
        """整段信号每 32 ms 的语音概率，展开到 10 ms 帧（取覆盖该帧的最大值）。"""
        sig = np.clip(np.asarray(sig, np.float32), -1, 1)
        state = np.zeros((2, 1, 128), np.float32)
        ctx = np.zeros((1, 64), np.float32)
        sr = np.array(SR, dtype=np.int64)
        p = np.zeros(n_frames(len(sig)))
        for i in range(0, len(sig) - 512, 512):
            x = np.concatenate([ctx, sig[None, i:i + 512]], axis=1)
            out, state = self.sess.run(None, {"input": x, "state": state, "sr": sr})
            ctx = x[:, -64:]
            sl = slice(i // HOP_N, (i + 512) // HOP_N + 1)
            p[sl] = np.maximum(p[sl], float(out.reshape(-1)[0]))
        return p


def slow_agc_signal(x_dn):
    """慢速 AGC 预增益（0.5 s 电平，只抬高比底噪高 4 dB 以上的部分到约 -26 dBFS，最多 +35 dB）。"""
    lv = frame_rms_db(x_dn, win=0.5)
    floor = np.percentile(lv, 10)
    g_db = np.clip(-26 - lv, 0, 35) * (lv > floor + 4)
    return x_dn * 10 ** (np.interp(np.arange(len(x_dn)) / SR, np.arange(len(lv)) * HOP, g_db) / 20)


def fast_agc_signal(x):
    """Codex 的检测信号：90 Hz 高通 + 50 ms 块快速 AGC（±200 ms 最大电平，增益 0.5~24 倍，到约 -23 dBFS）+ tanh。
    不降噪——降噪会把只比底噪高几 dB 的弱语音一起压掉。"""
    from scipy import ndimage, signal
    y = signal.sosfilt(signal.butter(2, 90, fs=SR, btype="highpass", output="sos"), x)
    fr = np.pad(y, (0, -len(y) % 800)).reshape(-1, 800)
    level = ndimage.maximum_filter1d(np.sqrt(np.mean(fr ** 2, 1)), 9, mode="nearest")
    gain = np.clip(0.07 / np.maximum(level, 1e-6), 0.5, 24)
    return np.tanh(y * np.interp(np.arange(len(y)), 800 * np.arange(len(gain)) + 400, gain)).astype(np.float32)


def vad_silero(silero: SileroVAD, x_dn, threshold=0.35, min_speech=0.25, min_gap=0.3):
    """单路 Silero：降噪后信号 + 慢速 AGC 预增益。"""
    p = silero.probs(slow_agc_signal(x_dn))
    return smooth_mask(p > threshold, min_speech, min_gap)


def vad_dual(silero: SileroVAD, x, x_dn, onset=0.35, offset=0.3, pad=0.0, min_speech=0.25, min_gap=0.3,
             onset2=0.5):
    """双路 Silero + 滞回（v3）：
    p1 = 降噪 + 慢速 AGC，p2 = 原始高通 + 快速 AGC；起段 p1 > onset 或 max > onset2；段内 max > offset 继续。"""
    p1 = silero.probs(slow_agc_signal(x_dn))
    p2 = silero.probs(fast_agc_signal(x))
    n = min(len(p1), len(p2))
    p = np.maximum(p1[:n], p2[:n])
    out = np.zeros(n, bool)
    on = False
    for i in range(n):
        on = (p[i] > offset) if on else (p1[i] > onset or p[i] > onset2)
        out[i] = on
    if pad > 0:
        from scipy.ndimage import binary_dilation
        out = binary_dilation(out, iterations=int(pad / HOP))
    nf = n_frames(len(x_dn))
    out = np.r_[out, np.zeros(max(0, nf - n), bool)][:nf]
    return smooth_mask(out, min_speech, min_gap)


# ---------------------------------------------------------------- 谐波检查
def voicing_track(x, lo=70, hi=400):
    """每 10 ms 一帧的周期性强度 (0~1)：带通 60~1000 Hz 后 40 ms 窗归一化自相关在基频范围的最大值。
    浊音约 0.5~0.9；敲击、碰撞、摩擦等瞬态/宽带噪声通常 < 0.3。"""
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, [60, 1000], btype="band", fs=SR, output="sos")
    xb = sosfiltfilt(sos, x).astype(np.float32)
    n = int(0.04 * SR)
    nf = n_frames(len(x))
    pad = np.pad(xb, (n // 2, n))
    idx = np.arange(nf)[:, None] * HOP_N + np.arange(n)[None, :]
    fr = pad[idx]
    fr = fr - fr.mean(1, keepdims=True)
    F = np.fft.rfft(fr, 2 * n)
    ac = np.fft.irfft(np.abs(F) ** 2, 2 * n)[:, :n]
    r = ac[:, SR // hi: SR // lo].max(1) / (ac[:, 0] + 1e-12)
    return np.clip(r, 0, 1)


def refine_mask(mask, lv, vt, min_voiced=0.25, voiced_r=0.45, floor=None):
    """剔除瞬态噪声段：段内比底噪高 3 dB 的帧中，周期性 > voiced_r 的比例低于 min_voiced 则丢弃。"""
    if floor is None:
        floor = np.percentile(lv, 10)
    m = mask.copy()
    dropped = 0
    for val, s, e in runs(mask):
        if not val:
            continue
        act = lv[s:e] > floor + 3
        if act.sum() < 5 or np.mean(vt[s:e][act] > voiced_r) < min_voiced:
            m[s:e] = False
            dropped += 1
    return m, dropped
