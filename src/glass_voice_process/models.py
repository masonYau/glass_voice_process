"""模型文件管理。

模型不随代码分发，首次使用前用 `glass-voice download-models`（或 scripts/download_models.py）下载，
或把已有文件放进模型目录。模型目录按以下顺序确定：
  1. 函数参数 / Config.model_dir
  2. 环境变量 GLASS_VOICE_MODELS
  3. ~/.cache/glass_voice_process/models
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelInfo:
    filename: str
    url: str
    license: str
    note: str
    archive_member: str | None = None  # 若 url 是压缩包，取其中这个文件


MODELS = {
    "silero_vad": ModelInfo(
        "silero_vad.onnx",
        "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx",
        "MIT", "Silero VAD v5（语音检测，v3/v4 使用）"),
    "pyannote_segmentation": ModelInfo(
        "pyannote_segmentation_3.0.onnx",
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/"
        "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
        "MIT", "pyannote segmentation-3.0 的 ONNX 版（sherpa-onnx 转换；v4/v5 使用，无需 HuggingFace token）",
        archive_member="sherpa-onnx-pyannote-segmentation-3-0/model.onnx"),
    "campplus": ModelInfo(
        "campplus_zh_cn_16k_common.onnx",
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/"
        "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx",
        "Apache-2.0", "3D-Speaker CAM++ 中文通用说话人嵌入（iic/speech_campplus_sv_zh-cn_16k-common；v3/v4/v5 使用）"),
    "mossformer2": ModelInfo(
        "mossformer2_ss_16k.onnx",
        "https://huggingface.co/iazShadow/mossformer2-ss-16k-onnx/resolve/main/mossformer2_ss_16k.onnx",
        "Apache-2.0", "MossFormer2 双人语音分离（可选，v5 的重叠分离使用，约 218 MB）"),
}


class ModelNotFound(FileNotFoundError):
    pass


def model_dir(explicit: str | os.PathLike | None = None) -> Path:
    if explicit:
        return Path(explicit)
    env = os.environ.get("GLASS_VOICE_MODELS")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "glass_voice_process" / "models"


def model_path(name: str, directory=None, required: bool = True) -> Path | None:
    info = MODELS[name]
    p = model_dir(directory) / info.filename
    if p.exists():
        return p
    if required:
        raise ModelNotFound(
            f"缺少模型 {name}（{info.note}）：{p}\n"
            f"  下载：glass-voice download-models --only {name}   （来源 {info.url}，许可 {info.license}）\n"
            f"  或手动放到上面的路径；也可用环境变量 GLASS_VOICE_MODELS 指定模型目录。")
    return None


def has_model(name: str, directory=None) -> bool:
    return model_path(name, directory, required=False) is not None


def download(names=None, directory=None, overwrite=False, log=print) -> dict:
    """下载模型到模型目录。返回 {name: path}。"""
    import tarfile
    import tempfile
    import urllib.request

    d = model_dir(directory)
    d.mkdir(parents=True, exist_ok=True)
    out = {}
    for name in names or ["silero_vad", "pyannote_segmentation", "campplus"]:
        info = MODELS[name]
        dst = d / info.filename
        if dst.exists() and not overwrite:
            log(f"[models] 已存在 {dst}")
            out[name] = dst
            continue
        log(f"[models] 下载 {name}：{info.url}")
        with tempfile.TemporaryDirectory() as tmp:
            tmpf = Path(tmp) / "download"
            urllib.request.urlretrieve(info.url, tmpf)
            if info.archive_member:
                with tarfile.open(tmpf) as tf:
                    member = tf.getmember(info.archive_member)
                    src = tf.extractfile(member)
                    dst.write_bytes(src.read())
            else:
                dst.write_bytes(tmpf.read_bytes())
        log(f"[models] 已保存 {dst}（许可 {info.license}）")
        out[name] = dst
    return out
