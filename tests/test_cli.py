import numpy as np
import soundfile as sf

from glass_voice_process.cli import main

from .conftest import need_models


def test_cli_models_lists_all(capsys):
    assert main(["models"]) == 0
    out = capsys.readouterr().out
    for name in ("silero_vad", "pyannote_segmentation", "campplus", "mossformer2"):
        assert name in out


@need_models("pyannote_segmentation", "campplus")
def test_cli_process(synth_conversation, tmp_path):
    x, _ = synth_conversation
    wav = tmp_path / "synth.wav"
    sf.write(wav, x, 16000, subtype="PCM_16")
    out = tmp_path / "out"
    cfg = tmp_path / "used.json"
    assert main(["process", str(wav), "-o", str(out), "-m", "v4", "-q", "--target-db", "-22",
                 "--save-config", str(cfg)]) == 0
    y, sr = sf.read(out / "synth_balanced.wav")
    assert sr == 16000 and len(y) == len(x) and np.isfinite(y).all()
    assert (out / "synth_segments.csv").exists() and (out / "synth_result.json").exists()
    assert '"target_db": -22.0' in cfg.read_text(encoding="utf-8")
