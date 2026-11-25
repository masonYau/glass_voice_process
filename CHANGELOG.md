# 更新记录

## 0.1.0（2026-11-25）

首个工程化版本：把实验脚本整理成可调用的 Python 包 `glass_voice_process`。

- 统一入口：`process()` / `Pipeline` / `Config`（可存取 JSON/YAML）/ `Result.save()`；命令行 `glass-voice process | download-models | models`。
- 三种方法：
  - **v3**：降噪 + 双路 Silero（滞回）+ 瞬态剔除 + 电平/直混比判佩戴者 + CAM++ 两步聚类 + 段内电平切分 + 增益流程（缺口桥接、慢释放、底噪约束、瞬态上限、尖峰压制、限幅）。
  - **v4（默认）**：v3 的语音检测改为 pyannote segmentation-3.0（本地 ONNX），按其换人点切分、标记重叠。
  - **v5**：移植 Codex 方法（sherpa-onnx 说话人分离 + 每人固定增益 + MossFormer2 重叠分离）。
- 核心依赖去掉 torch：Silero VAD、CAM++ 改用 onnxruntime 直接推理，kaldi fbank 用 numpy 实现；与原实现逐样本一致。
- 模型集中管理（`GLASS_VOICE_MODELS`），`download-models` 从官方发布地址下载。
- 评估工具：`metrics.diarization_scores` / `level_gaps` / `cutouts`；`tools/make_test_conversation.py` 用公开样本合成带真值的测试对话。
- 测试：pytest，测试音频在测试中合成；缺少模型时自动跳过。
