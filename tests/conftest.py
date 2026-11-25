"""测试夹具：测试音频全部在测试中合成，不使用、不提交任何录音。"""
import os

import numpy as np
import pytest

from glass_voice_process import models

SR = 16000


def _voice(dur, f0, formants, seed, level_db):
    """简单的"类语音"信号：带抖动的声门脉冲串 + 共振峰滤波 + 音节包络。"""
    from scipy.signal import lfilter
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = f0 * (1 + 0.05 * np.sin(2 * np.pi * 3 * t) + 0.01 * rng.standard_normal(n).cumsum() / np.sqrt(n))
    phase = np.cumsum(f) / SR
    src = (np.diff(np.floor(phase), prepend=0) > 0).astype(float)
    y = src
    for fc, bw in formants:
        r = np.exp(-np.pi * bw / SR)
        y = lfilter([1 - r], [1, -2 * r * np.cos(2 * np.pi * fc / SR), r * r], y)
    env = (0.5 - 0.5 * np.cos(2 * np.pi * np.minimum(1, (t % 0.25) / 0.25))) ** 0.5  # 4 Hz 音节
    y = y * env
    y = y / (np.sqrt(np.mean(y ** 2)) + 1e-12) * 10 ** (level_db / 20)
    return y.astype(np.float32)


@pytest.fixture(scope="session")
def synth_conversation():
    """佩戴者（近讲，-16 dBFS）与对话者（远处，-40 dBFS，带混响）轮流说话，加 -62 dBFS 底噪。
    返回 (信号, 真值 [(说话人, 起, 止)])。"""
    rng = np.random.default_rng(0)
    parts, truth, t = [], [], 0.5
    rir = rng.standard_normal(int(0.4 * SR)) * np.exp(-6.9 * np.arange(int(0.4 * SR)) / SR / 0.45)
    rir[0] = 4.0
    turns = [("wearer", 3.0), ("partner_A", 3.0), ("wearer", 2.5), ("partner_A", 3.0), ("wearer", 3.0), ("partner_A", 2.5)]
    total = int((sum(d for _, d in turns) + 0.7 * len(turns) + 1.5) * SR)
    x = np.zeros(total, np.float32)
    for i, (spk, d) in enumerate(turns):
        if spk == "wearer":
            v = _voice(d, 125, [(700, 90), (1200, 110), (2600, 160)], i, -16)
        else:
            v = _voice(d, 210, [(500, 80), (1800, 120), (2800, 170)], i, -40)
            v = np.convolve(v, rir)[: len(v)]
            v = v / (np.sqrt(np.mean(v ** 2)) + 1e-12) * 10 ** (-40 / 20)
        s = int(t * SR)
        x[s:s + len(v)] += v
        truth.append((spk, t, t + d))
        t += d + 0.7
    x += (rng.standard_normal(total) * 10 ** (-62 / 20)).astype(np.float32)
    return x.astype(np.float32), truth


def need_models(*names):
    missing = [n for n in names if not models.has_model(n)]
    return pytest.mark.skipif(bool(missing), reason=f"缺少模型 {missing}（glass-voice download-models）")


def need_module(mod):
    try:
        __import__(mod)
        return pytest.mark.skipif(False, reason="")
    except ImportError:
        return pytest.mark.skipif(True, reason=f"未安装 {mod}")


@pytest.fixture(scope="session")
def public_conversation():
    """可选：用公开样本合成的对话（tools/make_test_conversation.py 生成），
    设置环境变量 GLASS_VOICE_TEST_CONVERSATION=<wav 路径>（同目录需有 *_truth.csv）后启用。"""
    p = os.environ.get("GLASS_VOICE_TEST_CONVERSATION")
    if not p or not os.path.exists(p):
        pytest.skip("未设置 GLASS_VOICE_TEST_CONVERSATION")
    import csv
    truth_csv = p[:-4] + "_truth.csv"
    with open(truth_csv, encoding="utf-8-sig") as f:
        truth = [(r["speaker"], float(r["start"]), float(r["end"])) for r in csv.DictReader(f)]
    return p, truth
