"""音色特征：窗口级 MFCC 统计（佩戴者二分也在这里算电平/直混比），CAM++（ONNX）与可选 ECAPA 说话人嵌入。"""
from __future__ import annotations

import numpy as np

from .audio import HOP, SR
from .features import fbank
from .utils import runs


def window_feats(x, lv, mask, win=1.2, hop=0.4):
    """在语音区内切 1.2 s / 0.4 s 窗口。返回 [(f0, f1)] 帧区间、电平、直混比代理、MFCC 统计。"""
    import librosa
    mfcc = librosa.feature.mfcc(y=x, sr=SR, n_mfcc=24, n_fft=512,
                                hop_length=int(HOP * SR), win_length=int(0.025 * SR))[:, : len(mask)]
    W, H = int(win / HOP), int(hop / HOP)
    wins, levels, drr, timbre = [], [], [], []
    for val, s, e in runs(mask):
        if not val:
            continue
        for f0 in range(s, max(s + 1, e - W + 1), H):
            f1 = min(e, f0 + W)
            if f1 - f0 < 20:
                continue
            seg_lv = lv[f0:f1]
            wins.append((f0, f1))
            levels.append(np.mean(np.sort(seg_lv)[-max(3, (f1 - f0) // 3):]))
            # 直混比代理：近讲语音帧间起伏大，混响把能量谷"填平"
            drr.append(np.percentile(seg_lv, 90) - np.percentile(seg_lv, 30))
            act = seg_lv >= np.median(seg_lv)  # 只用较响一半帧统计音色，避免底噪主导
            m = mfcc[1:, f0:f1][:, act]
            timbre.append(np.concatenate([m.mean(1), m.std(1)]))
    return wins, np.array(levels), np.array(drr), np.array(timbre)


class CamPlus:
    """3D-Speaker CAM++ 中文通用模型（ONNX）。特征：80 维 kaldi fbank 减时间均值；与 modelscope 的 torch 实现输出一致。"""
    name = "campplus"
    partner_sim = 0.45    # 片段级平均链接聚类阈值
    partner_merge = 0.5   # 簇质心相似度 ≥ 此值则合并

    def __init__(self, model_file):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(model_file), providers=["CPUExecutionProvider"])

    def __call__(self, seg: np.ndarray) -> np.ndarray:
        f = fbank(seg, SR, 80)
        f = f - f.mean(0, keepdims=True)
        return self.sess.run(None, {"x": f[None].astype(np.float32)})[0].reshape(-1)


class Ecapa:
    """可选：speechbrain ECAPA（VoxCeleb，英文更好）。需要 pip install .[ecapa]，首次运行会从 HuggingFace 下载。"""
    name = "ecapa"
    partner_sim = 0.35
    partner_merge = 0.5

    def __init__(self, savedir):
        from speechbrain.inference.speaker import EncoderClassifier
        kw = {}
        try:
            from speechbrain.utils.fetching import LocalStrategy
            kw["local_strategy"] = LocalStrategy.COPY
        except ImportError:
            pass
        self.enc = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb", savedir=str(savedir),
                                                  run_opts={"device": "cpu"}, **kw)

    def __call__(self, seg):
        import torch
        with torch.no_grad():
            return self.enc.encode_batch(torch.from_numpy(seg).unsqueeze(0)).squeeze().numpy()


def embed_segments(model, x, segs, min_len=1.5, norm_db=-26.0):
    """segs: [(s0, s1)] 采样点区间 → L2 归一化嵌入 (n, d)。
    低信噪比预处理：按较响一半样本的 RMS 归一化到 norm_db；不足 min_len 秒循环拼接补足。"""
    out = []
    for s0, s1 in segs:
        seg = np.asarray(x[s0:s1], np.float32)
        if len(seg) < int(0.2 * SR):
            seg = np.pad(seg, (0, int(0.2 * SR) - len(seg)))
        e2 = seg ** 2
        ref = np.sqrt(np.mean(np.sort(e2)[len(e2) // 2:]) + 1e-12)
        seg = seg * (10 ** (norm_db / 20) / ref)
        if len(seg) < int(min_len * SR):
            seg = np.tile(seg, int(np.ceil(min_len * SR / len(seg))))[: int(min_len * SR)]
        e = model(np.clip(seg, -1, 1).astype(np.float32))
        out.append(e / (np.linalg.norm(e) + 1e-9))
    return np.array(out)
