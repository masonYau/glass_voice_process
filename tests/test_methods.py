"""端到端：三种方法能跑通、输出合法；有模型时才运行。"""
import numpy as np
import pytest

from glass_voice_process import Pipeline, process
from glass_voice_process.metrics import cutouts, diarization_scores, level_gaps
from glass_voice_process.pipeline import quiet

from .conftest import need_models, need_module

SR = 16000


def _check(r, x):
    assert len(r.balanced) == len(x)
    assert np.isfinite(r.balanced).all()
    assert np.max(np.abs(r.balanced)) <= 10 ** (-1 / 20) + 1e-3   # 限幅生效（v5 为 -1.5 dBFS）
    for y in r.tracks.values():
        assert len(y) == len(x) and np.isfinite(y).all()
    for s in r.segments:
        assert 0 <= s.start < s.end <= len(x) / SR + 0.05


@need_models("silero_vad", "campplus")
def test_v3_runs(synth_conversation, tmp_path):
    x, _ = synth_conversation
    r = process(x, sr=SR, method="v3", log=quiet)
    _check(r, x)
    files = r.save(tmp_path, "synth")
    assert files["balanced"].exists() and files["segments"].exists()


@need_models("pyannote_segmentation", "campplus")
def test_v4_runs(synth_conversation):
    x, _ = synth_conversation
    r = process(x, sr=SR, method="v4", log=quiet)
    _check(r, x)
    assert all(s.overlap is not None for s in r.segments)
    assert "gaps" in cutouts(r)


@need_models("pyannote_segmentation", "campplus")
@need_module("sherpa_onnx")
def test_v5_runs(synth_conversation):
    x, _ = synth_conversation
    r = process(x, sr=SR, method="v5", log=quiet, v5_separation="off")
    _check(r, x)
    assert r.stats["output_peak_dbfs"] <= -1.4


@need_models("silero_vad", "campplus")
def test_wearer_quieter_after_v3(synth_conversation):
    """处理后佩戴者与对话者的电平差应明显缩小（合成信号未必被所有检测器当成语音，检出不到则跳过）。"""
    x, truth = synth_conversation
    r = process(x, sr=SR, method="v3", log=quiet)
    if not any(s.label != "wearer" for s in r.segments):
        pytest.skip("合成信号的对话者未被检出")
    gaps = level_gaps(r)
    assert gaps and all(after < before for before, after in gaps.values())


@need_models("silero_vad", "pyannote_segmentation", "campplus")
def test_public_conversation_wearer_accuracy(public_conversation):
    """用公开样本合成的对话（有真值）：佩戴者/他人判别 ≥ 98%。"""
    path, truth = public_conversation
    p = Pipeline(method="v4", log=quiet)
    r = p(path)
    sc = diarization_scores(truth, r.segments, len(r.input) // 160 + 1)
    assert sc["wearer_accuracy"] >= 0.98
    gaps = level_gaps(r)
    assert all(after < 6 for _, after in gaps.values())
