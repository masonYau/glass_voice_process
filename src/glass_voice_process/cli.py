"""命令行：
  glass-voice process <输入文件或目录...> -o <输出目录> [--method v3|v4|v5] [参数]
  glass-voice download-models [--only silero_vad pyannote_segmentation campplus mossformer2] [--dir 目录]
  glass-voice models            查看模型目录与状态
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields

from . import models
from .config import METHODS, Config
from .pipeline import Pipeline, iter_inputs, quiet

# 命令行可覆盖的 Config 参数（自动生成 --xxx-yyy 选项）
_SKIP = {"method", "model_dir", "v5_trim"}


def _add_config_args(ap):
    g = ap.add_argument_group("处理参数（不填则用所选方法的默认值，详见 README 参数表）")
    for f in fields(Config):
        if f.name in _SKIP:
            continue
        opt = "--" + f.name.replace("_", "-")
        if f.type in ("bool", bool):
            g.add_argument(opt, dest=f.name, default=None, action=argparse.BooleanOptionalAction)
        elif "int" in str(f.type) and "float" not in str(f.type):
            g.add_argument(opt, dest=f.name, type=int, default=None)
        elif "float" in str(f.type):
            g.add_argument(opt, dest=f.name, type=float, default=None)
        else:
            g.add_argument(opt, dest=f.name, type=str, default=None)


def _cmd_process(a):
    if a.config:
        cfg = Config.load(a.config)
        if a.method:
            cfg.method = a.method
    else:
        cfg = Config.preset(a.method or "v4")
    for f in fields(Config):
        v = getattr(a, f.name, None)
        if f.name not in _SKIP and v is not None:
            setattr(cfg, f.name, v)
    if a.model_dir:
        cfg.model_dir = a.model_dir
    if a.trim:
        cfg.v5_trim = {int(k): float(v) for k, v in (t.split("=") for t in a.trim)}
    pipe = Pipeline(cfg, log=quiet if a.quiet else print)
    files = list(iter_inputs(a.inputs))
    if not files:
        print("没有找到输入音频", file=sys.stderr)
        return 2
    for p in files:
        print(f"== {p}  （{cfg.method}）")
        r = pipe(p)
        out = r.save(a.out, p.stem, plot=a.plot, save_tracks=not a.no_tracks)
        for sp, d in r.summary().items():
            print(f"  {sp:10s} {d['segments']:3d} 段 {d['seconds']:7.1f}s  原始电平 {d['level_db']} dBFS  增益 {d['gain_db']} dB")
        print(f"  已写出 {out['balanced']}")
    if a.save_config:
        cfg.save(a.save_config)
    return 0


def _cmd_download(a):
    models.download(a.only, a.dir, overwrite=a.overwrite)
    return 0


def _cmd_models(a):
    d = models.model_dir(a.dir)
    print(f"模型目录：{d}")
    for name, info in models.MODELS.items():
        ok = models.has_model(name, a.dir)
        print(f"  [{'✓' if ok else ' '}] {name:22s} {info.filename:34s} {info.license:11s} {info.note}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="glass-voice", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("process", help="处理录音")
    p.add_argument("inputs", nargs="+", help="音频文件或目录")
    p.add_argument("-o", "--out", default="outputs", help="输出目录")
    p.add_argument("-m", "--method", choices=METHODS, default=None, help="v3 / v4（默认）/ v5")
    p.add_argument("--config", help="从 JSON/YAML 读取参数")
    p.add_argument("--save-config", help="把本次使用的参数保存为 JSON/YAML")
    p.add_argument("--model-dir", help="模型目录（默认 GLASS_VOICE_MODELS 或 ~/.cache/glass_voice_process/models）")
    p.add_argument("--plot", action="store_true", help="输出 report.png（需要 matplotlib）")
    p.add_argument("--no-tracks", action="store_true", help="不输出每个说话人的分轨")
    p.add_argument("--trim", action="append", default=[], metavar="ID=DB", help="v5：某轨额外增益，可重复")
    p.add_argument("-q", "--quiet", action="store_true")
    _add_config_args(p)
    p.set_defaults(func=_cmd_process)
    d = sub.add_parser("download-models", help="下载模型")
    d.add_argument("--only", nargs="+", choices=list(models.MODELS), help="只下载这些（默认：silero_vad pyannote_segmentation campplus）")
    d.add_argument("--dir", help="模型目录")
    d.add_argument("--overwrite", action="store_true")
    d.set_defaults(func=_cmd_download)
    m = sub.add_parser("models", help="查看模型状态")
    m.add_argument("--dir")
    m.set_defaults(func=_cmd_models)
    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
