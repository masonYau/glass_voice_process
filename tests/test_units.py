"""不需要模型的单元测试。"""
import json

import numpy as np
import pytest

from glass_voice_process import Config, Segment
from glass_voice_process.audio import SR, frame_rms_db, load_audio
from glass_voice_process.features import fbank
from glass_voice_process.gain import apply_gains, limiter, suppress_clicks
from glass_voice_process.utils import runs, smooth_mask


def test_runs_and_smooth():
    m = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0], bool)
    assert runs(m)[0] == (False, 0, 2)
    # 0.01 s 的间隙被填上，0.02 s 的语音被去掉
    out = smooth_mask(np.r_[np.ones(50), np.zeros(1), np.ones(50), np.zeros(20), np.ones(2), np.zeros(10)].astype(bool),
                      min_on=0.05, min_off=0.05)
    assert out[50] and not out[-12:].any()


def test_load_audio_resample_and_mono():
    x = np.random.default_rng(0).standard_normal((44100, 2)).astype(np.float32) * 0.1
    y = load_audio(x, sr=44100)
    assert y.ndim == 1 and abs(len(y) - SR) <= 2
    with pytest.raises(ValueError):
        load_audio(x)


def test_fbank_shape_and_finite():
    x = np.random.default_rng(1).standard_normal(SR).astype(np.float32) * 0.05
    f = fbank(x)
    assert f.shape == (98, 80) and np.isfinite(f).all()


def test_limiter_ceiling():
    y = np.sin(np.linspace(0, 200 * np.pi, SR)).astype(np.float32) * 3.0
    out = limiter(y, ceiling_db=-1.0)
    assert np.max(np.abs(out)) <= 10 ** (-1 / 20) + 1e-3


def test_gain_bridging_and_release():
    """两侧都被放大的 1.5 s 缺口应被桥接（不掉回 0 dB）；长静音最终回到 0 dB。"""
    n = 12 * SR
    x = np.random.default_rng(2).standard_normal(n).astype(np.float32) * 10 ** (-55 / 20)
    lv = frame_rms_db(x)
    segs = [Segment(1.0, 3.0, "partner_1"), Segment(4.5, 6.0, "partner_1")]
    _, gs, g, has = apply_gains(x, segs, lv, target_db=-23, max_boost=30, transient_margin=None)
    assert g[int(3.75 / 0.01)] > 20          # 缺口中间保持放大
    assert abs(g[int(10.5 / 0.01)]) < 1e-6   # 长静音回到 0 dB
    # 关闭桥接与释放时缺口中间掉回 0
    _, _, g0, _ = apply_gains(x, [Segment(1.0, 3.0, "partner_1"), Segment(4.5, 6.0, "partner_1")], lv,
                              target_db=-23, max_boost=30, transient_margin=None, bridge_gap=0, release=0)
    assert abs(g0[int(3.75 / 0.01)]) < 1e-6


def test_click_suppression_only_in_boosted_regions():
    y = np.zeros(SR, np.float32)
    y[8000:8040] = 0.9
    gs = np.zeros(SR)
    out, n = suppress_clicks(y.copy(), gs)
    assert n == 0 and np.max(np.abs(out)) == pytest.approx(0.9)
    out, n = suppress_clicks(y.copy(), gs + 20)
    assert n > 0 and np.max(np.abs(out)) < 0.9


def test_config_preset_and_roundtrip(tmp_path):
    c = Config.preset("v3", target_db=-20)
    assert c.segmenter == "none" and c.target_db == -20
    assert Config.preset("v4").segmenter == "pyannote"
    p = tmp_path / "cfg.json"
    c.save(p)
    assert Config.load(p) == c
    assert json.loads(p.read_text(encoding="utf-8"))["method"] == "v3"
    with pytest.raises(ValueError):
        Config.preset("v9")
    with pytest.raises(TypeError):
        Config.preset("v4", not_a_param=1)
