"""统一入口：process(...) / Pipeline。"""
from __future__ import annotations

from pathlib import Path

from .audio import load_audio
from .config import METHODS, Config
from .result import Result


class Pipeline:
    """可复用的处理器（模型在多次调用间缓存）。

    >>> p = Pipeline(method="v4")
    >>> r = p("input.wav")
    >>> r.save("outputs", "input")
    """

    def __init__(self, config: Config | None = None, method: str | None = None, log=print, **params):
        if config is None:
            config = Config.preset(method or "v4", **params)
        elif method or params:
            d = config.to_dict()
            d.update(params)
            if method:
                d["method"] = method
            config = Config(**d)
        if config.method not in METHODS:
            raise ValueError(f"method 必须是 {METHODS} 之一")
        self.config = config
        self.log = log

    def __call__(self, src, sr: int | None = None) -> Result:
        x = load_audio(src, sr)
        if len(x) < 16000:
            raise ValueError("音频至少需要 1 秒")
        cfg = Config(**self.config.to_dict())  # 每次调用用独立副本
        if cfg.method in ("v3", "v4"):
            from .methods import ours
            return ours.run(x, cfg, self.log)
        from .methods import codex
        return codex.run(x, cfg, self.log)

    def process_file(self, path, out_dir, plot=False, save_tracks=True) -> dict:
        r = self(path)
        return r.save(out_dir, Path(path).stem, plot=plot, save_tracks=save_tracks)


def process(src, sr: int | None = None, method: str = "v4", config: Config | None = None, log=print,
            **params) -> Result:
    """处理一段录音。

    src    : 文件路径，或 ndarray（此时需给 sr）
    method : "v3" / "v4"（默认）/ "v5"
    params : 覆盖 Config 中的任意参数，例如 target_db=-20, num_partners=1
    返回 Result（balanced、tracks、segments、frame_gain、stats）。"""
    return Pipeline(config, method=method if config is None else None, log=log, **params)(src, sr)


def iter_inputs(paths):
    exts = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
    for p in paths:
        p = Path(p)
        if p.is_dir():
            yield from sorted(q for q in p.iterdir() if q.suffix.lower() in exts)
        else:
            yield p


def quiet(*_a, **_k):
    """不输出日志：Pipeline(log=quiet)。"""


__all__ = ["Pipeline", "process", "iter_inputs", "quiet"]
