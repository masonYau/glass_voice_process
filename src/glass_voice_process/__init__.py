"""glass_voice_process：智能眼镜录音预处理——分离佩戴者与对话者并平衡音量。

快速使用
    from glass_voice_process import process
    r = process("input.wav", method="v4")        # v3 / v4（默认）/ v5
    r.save("outputs", "input", plot=True)
    print(r.summary())

三种方法
    v3  降噪 + 双路 Silero 检测 + 电平/直混比判佩戴者 + CAM++ 聚类 + 段内电平切分 + 增益流程（桥接/慢释放/尖峰压制）
    v4  v3 的语音检测改用 pyannote segmentation-3.0（ONNX），按其换人点切分并标记重叠（推荐，默认）
    v5  Codex 方法：pyannote 分段 + CAM++ 聚类（sherpa-onnx）、每人固定增益、段外保持原音量、可选 MossFormer2 重叠分离
"""
from .config import METHODS, Config
from .models import download as download_models
from .pipeline import Pipeline, process
from .result import Result
from .utils import Segment

__version__ = "0.1.0"
__all__ = ["process", "Pipeline", "Config", "Result", "Segment", "METHODS", "download_models", "__version__"]
