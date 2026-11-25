"""处理参数。三种方法共用一个 Config；Config.preset(method) 给出该方法的默认值。可保存/读取 JSON（装了 PyYAML 也可用 YAML）。"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

METHODS = ("v3", "v4", "v5")


@dataclass
class Config:
    method: str = "v4"
    model_dir: str | None = None          # 模型目录；默认取环境变量 GLASS_VOICE_MODELS 或 ~/.cache/glass_voice_process/models

    # ---- 电平目标（三种方法共用）
    target_db: float = -23.0              # 每个说话人统一到的语音电平 (dBFS)

    # ---- v3 / v4：我们的流程
    denoise: float = 0.85                 # 平稳谱门限降噪强度 0~1，0 关闭
    vad: str = "dual"                     # dual / silero / webrtc
    vad_threshold: float = 0.35           # Silero 起段门限
    vad_offset: float = 0.3               # dual：段内延续门限（滞回）
    vad_pad: float = 0.0                  # dual：每段前后补齐秒数
    segmenter: str = "pyannote"           # v4：pyannote / union / split / none（v3 固定 none）
    py_onset: float = 0.5                 # pyannote 语音概率起段门限
    py_offset: float = 0.5                # pyannote 语音概率延续门限
    change_thr: float = 0.5               # pyannote 换人得分门限
    min_voiced: float = 0.25              # 语音段内浊音帧最低比例（瞬态剔除），0 关闭
    turn_split: bool = True               # 段内按电平切分佩戴者/对话者换人
    embedder: str = "campplus"            # campplus / ecapa / mfcc
    num_partners: int | None = None       # 已知对话者人数
    partner_threshold: float | None = None  # 仅 mfcc：余弦距离阈值
    partner_sim: float | None = None      # campplus/ecapa：片段级聚类相似度阈值（默认 0.45 / 0.35）
    partner_merge: float | None = None    # campplus/ecapa：质心相似度 ≥ 此值合并（默认 0.5）
    max_boost: float = 35.0               # 最大放大 dB
    max_cut: float = -15.0                # 最大衰减 dB
    noise_ceiling: float = -45.0          # 放大后底噪上限 dBFS
    bridge_gap: float = 2.5               # 两侧都放大的未标注缺口 ≤ 此秒数时桥接，0 关闭
    release: float = 0.4                  # 放大段后增益每 release 秒最多降 30 dB，0 = 直接回落
    limiter_ceiling: float = -1.0         # 限幅 dBFS

    # ---- v5：Codex 方法
    v5_num_speakers: int = -1             # 已知总人数（含佩戴者），-1 自动
    v5_cluster_threshold: float = 0.70    # FastClustering 阈值，越大越容易合并
    v5_max_boost: float = 26.0            # 每轨最大提升 dB（最大衰减固定 18 dB）
    v5_separation: str = "auto"           # 两人重叠段 MossFormer2 分离：auto（有模型就用）/ on / off
    v5_check_interruptions: bool = False  # 复查短插话（更慢）
    v5_trim: dict = field(default_factory=dict)  # 每轨额外增益 {轨号: dB}

    @classmethod
    def preset(cls, method: str = "v4", **overrides) -> "Config":
        if method not in METHODS:
            raise ValueError(f"method 必须是 {METHODS} 之一")
        c = cls(method=method)
        if method == "v3":
            c.segmenter = "none"
        for k, v in overrides.items():
            if not hasattr(c, k):
                raise TypeError(f"未知参数 {k}")
            setattr(c, k, v)
        if method == "v3" and overrides.get("segmenter") not in (None, "none"):
            raise ValueError("v3 不使用 pyannote；需要 pyannote 分段请用 method='v4'")
        return c

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        path = Path(path)
        d = self.to_dict()
        if path.suffix in (".yaml", ".yml"):
            import yaml
            path.write_text(yaml.safe_dump(d, allow_unicode=True, sort_keys=False), encoding="utf-8")
        else:
            path.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "Config":
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix in (".yaml", ".yml"):
            import yaml
            d = yaml.safe_load(text)
        else:
            d = json.loads(text)
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise TypeError(f"未知参数 {sorted(unknown)}")
        return cls(**d)
