# glass_voice_process

智能眼镜录音预处理：把**佩戴者**（嘴离麦克风几厘米，很响）和**对话者**（1~2 m 外，常常只比底噪高 5~15 dB）分开，并把两者拉到相近的音量。

- 三种方法可选：**v3**、**v4**（默认，推荐）、**v5**（Codex 方法）
- 纯本地离线推理（onnxruntime），不需要 GPU，不需要 HuggingFace token，录音不离开本机
- Python API 与命令行两种用法；输出平衡后的混音、每个说话人的分轨、分段表

## 三种方法

| | v3 | v4（默认） | v5（Codex 方法） |
|---|---|---|---|
| 降噪 | 平稳谱门限（noisereduce） | 同 v3 | 不降噪 |
| 语音检测 | 双路 Silero（降噪+慢速 AGC / 原始高通+快速 AGC）+ 滞回 | pyannote segmentation-3.0（ONNX） | pyannote segmentation-3.0（sherpa-onnx） |
| 佩戴者判定 | 电平 + 直混比（GMM 二分） | 同 v3 | 输入电平最高的一轨（推测） |
| 对话者聚类 | CAM++ 两步聚类（片段级 + 质心合并） | 同 v3，并在 pyannote 换人点处切分 | CAM++ + FastClustering |
| 换人切分 | 段内按电平 | 段内按电平，需 pyannote 换人点佐证 | pyannote 逐帧 |
| 增益 | 分段增益 + 2 s 慢速 AGC、缺口桥接、慢释放、底噪约束、瞬态上限、尖峰压制 | 同 v3 | 每人一个固定增益，段外保持原音量 |
| 重叠语音 | 不处理 | 标记（segments.csv 的 overlap 列） | MossFormer2 分离两人重叠（可选模型） |
| 速度（5 分钟录音，CPU） | 约 10~20 s | 约 6 s | 约 2 分钟 |

**怎么选**
- 一般用 **v4**：小声对话者漏检最少、佩戴者判定稳定、音量最平衡、没有突然断音。
- 想让输出更"原汁原味"（不降噪、弱音节保留完整、两人抢话时尽量分开）用 **v5**；代价是段内底噪也被放大，漏检部分保持原音量，佩戴者与对话者的电平差通常比 v4 大。
- **v3** 不依赖 pyannote 模型，作为对照或备选。

在 7 段实采录音（单声道 16 kHz，各 5 分钟）上的示例结果见 [docs/methods.md](docs/methods.md)。

## 安装

```bash
pip install -e .                 # 核心（v3 / v4）
pip install -e ".[v5]"           # 加上 v5 需要的 sherpa-onnx
pip install -e ".[report]"       # 输出 report.png 需要 matplotlib
pip install -e ".[ecapa]"        # 可选的 ECAPA 嵌入（需要 torch / speechbrain）
pip install -e ".[dev]"          # 测试
```

Python ≥ 3.10。核心依赖：numpy、scipy、soundfile、librosa、noisereduce、webrtcvad-wheels、scikit-learn、onnxruntime（**不需要 torch**）。

## 模型

模型不随代码分发。首次使用前下载到模型目录（默认 `~/.cache/glass_voice_process/models`，可用环境变量 `GLASS_VOICE_MODELS` 或参数 `model_dir` 指定）：

```bash
glass-voice download-models                       # silero_vad + pyannote_segmentation + campplus（约 37 MB）
glass-voice download-models --only mossformer2    # v5 的重叠分离（约 218 MB，可选）
glass-voice models                                # 查看状态
```

| 名称 | 文件 | 用途 | 来源 | 许可 |
|---|---|---|---|---|
| silero_vad | silero_vad.onnx | v3 语音检测 | [snakers4/silero-vad](https://github.com/snakers4/silero-vad) | MIT |
| pyannote_segmentation | pyannote_segmentation_3.0.onnx | v4/v5 分段 | [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)，[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-segmentation-models) 转换的 ONNX | MIT |
| campplus | campplus_zh_cn_16k_common.onnx | 说话人嵌入 | [3D-Speaker CAM++ 中文通用](https://www.modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common)，[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models) 导出 | Apache-2.0 |
| mossformer2 | mossformer2_ss_16k.onnx | v5 两人重叠分离（可选） | [MossFormer2_SS_16K](https://huggingface.co/alibabasglab/MossFormer2_SS_16K)，[ONNX 版](https://huggingface.co/iazShadow/mossformer2-ss-16k-onnx) | Apache-2.0 |

已经有这些文件时，直接放进模型目录（按上表文件名）即可，无需联网。

## Python 用法

```python
from glass_voice_process import process, Pipeline, Config

# 一次性处理
r = process("input.wav", method="v4")             # 也可传 ndarray：process(x, sr=48000)
r.save("outputs", "input", plot=True)             # 写出 wav / csv / json / png
print(r.summary())                                # 各说话人段数、时长、原始电平、增益
r.balanced          # 平衡后的混音（16 kHz float32）
r.tracks            # {"wearer": ..., "partner_1": ...}，与原始等长、时间对齐
r.segments          # [Segment(start, end, label, level_db, gain_db, overlap)]

# 批量处理时复用模型
p = Pipeline(method="v4", num_partners=1, target_db=-22)
for f in files:
    p(f).save("outputs", Path(f).stem)

# 参数可保存 / 读取
cfg = Config.preset("v5", v5_max_boost=20)
cfg.save("my_config.json")
r = process("input.wav", config=Config.load("my_config.json"))
```

评估工具在 `glass_voice_process.metrics`：`diarization_scores`（有真值时的检出/误检/佩戴者判别/纯度/覆盖度）、`level_gaps`（佩戴者−对话者电平差，处理前→后）、`cutouts`（断音缺口和陡降）。

## 命令行

```bash
glass-voice process input.wav -o outputs                  # v4
glass-voice process recordings/ -o outputs -m v5 --plot   # 目录里的所有音频
glass-voice process a.wav -o out -m v3 --num-partners 1 --target-db -22 --save-config used.json
glass-voice process a.wav -o out --config used.json
glass-voice process a.wav -o out -m v5 --trim 2=3         # v5：第 2 轨再提高 3 dB
python -m glass_voice_process --help
```

输出（每个输入一组）：

| 文件 | 内容 |
|---|---|
| `<名>_balanced.wav` | 平衡后的完整录音（最常用） |
| `<名>_wearer.wav` / `<名>_partner_N.wav` | 每个说话人一条轨，时间对齐 |
| `<名>_segments.csv` | speaker, start_s, end_s, level_dBFS（处理前）, gain_dB, overlap |
| `<名>_result.json` | 方法、各说话人汇总、统计量、所用参数 |
| `<名>_report.png` | 处理前后波形 + 分段（`--plot`） |

## 参数

所有参数都在 `Config` 中，命令行用 `--参数-名`（下划线换成短横线）。

| 参数 | 默认 | 作用 / 什么时候调 |
|---|---|---|
| `target_db` | -23 | 每个说话人统一到的语音电平 (dBFS) |
| `num_partners` | 自动 | 已知对话者人数时设定（最常用 1） |
| **v3 / v4** | | |
| `denoise` | 0.85 | 降噪强度 0~1；放大后底噪大调到 0.95，声音发闷调到 0.6，0 关闭 |
| `segmenter` | v4: pyannote | pyannote / union（双路 Silero ∪ pyannote）/ split（只用换人点）/ none |
| `py_onset` / `py_offset` | 0.5 / 0.5 | pyannote 语音概率门限；小声对话者仍漏检时降到 0.4 / 0.35 |
| `change_thr` | 0.5 | pyannote 换人得分门限；段内换人没切开时降到 0.3 |
| `vad` | dual | v3 的检测：dual / silero / webrtc |
| `vad_threshold` / `vad_offset` | 0.35 / 0.3 | dual 的起段 / 延续门限 |
| `min_voiced` | 0.25 | 段内浊音帧最低比例，低于则视为敲击/碰撞丢弃；0 关闭 |
| `turn_split` | true | 段内按电平切分佩戴者/对话者换人 |
| `embedder` | campplus | campplus / ecapa（英文更好，需 `.[ecapa]`）/ mfcc（无需模型） |
| `partner_sim` / `partner_merge` | 0.45 / 0.5 | CAM++ 两步聚类阈值；同一人被拆开时把 merge 降到 0.4，不同人被合并时升到 0.6 |
| `max_boost` / `max_cut` | 35 / -15 | 最大放大 / 衰减 dB |
| `noise_ceiling` | -45 | 放大后底噪上限 (dBFS)；想让小声对话者更响、能接受底噪时设 -40 |
| `bridge_gap` | 2.5 | 两侧都被放大的未标注缺口 ≤ 此秒数时桥接，避免突然断音；0 关闭 |
| `release` | 0.4 | 放大段结束后增益每 0.4 s 最多降 30 dB；0 = 直接回落 |
| **v5** | | |
| `v5_num_speakers` | -1 | 已知总人数（含佩戴者），-1 自动 |
| `v5_cluster_threshold` | 0.70 | 调大更容易合并声纹相近的类 |
| `v5_max_boost` | 26 | 每轨最大提升 dB（最大衰减固定 18 dB） |
| `v5_separation` | auto | 两人重叠段用 MossFormer2 分离：auto（有模型就用）/ on / off |
| `v5_check_interruptions` | false | 复查短 A-B-A 插话（更慢） |
| `v5_trim` | {} | 每轨额外增益，命令行 `--trim 轨号=dB` |

## 测试

```bash
pip install -e ".[dev]"
glass-voice download-models
pytest
```

测试音频全部在测试中合成，不使用任何录音；缺少模型时相关测试自动跳过。另可用公开语音样本合成带真值的对话来做回归：

```bash
python tools/make_test_conversation.py --wearer w1.wav w2.wav --partner p1.wav --partner2 p2.wav --out conv.wav
GLASS_VOICE_TEST_CONVERSATION=conv.wav pytest      # 检查佩戴者判别 ≥ 98%、处理后电平差 < 6 dB
```

## 已知局限

- 低信噪比下（对话者只比底噪高 2~6 dB、几乎没有谐波）仍有漏检；继续放宽门限会明显增加误检。
- 同一段里两人同时说话：v3/v4 只能整段归给一个人（v4 会在 overlap 列标出）；v5 可用 MossFormer2 分离。
- CAM++ 是中文模型，对两个声音相近的英文说话人区分度有限（可试 `embedder="ecapa"`）。
- 无法区分"对话者"和"背景人声"（例如旁人、电视），它们同样会被放大。
- v5 的佩戴者是"输入电平最高的一轨"的推测；佩戴者被拆成两轨时可能判错。

## 来源与致谢

- v3 / v4 是本项目在实采录音上迭代得到的方法（降噪、双路检测、瞬态剔除、两步聚类、缺口桥接等），详见 [docs/methods.md](docs/methods.md)。
- v5 移植自 Mason 用 Codex 编写的 `process_audio.py`（pyannote 分段 + CAM++ 聚类 + 每人固定增益 + MossFormer2 重叠分离），算法保持一致。
- 模型：Silero VAD、pyannote segmentation-3.0（CNRS）、3D-Speaker CAM++（阿里巴巴通义实验室）、MossFormer2（阿里巴巴），ONNX 转换来自 sherpa-onnx（k2-fsa）。许可见上表。

更新记录见 [CHANGELOG.md](CHANGELOG.md)。
