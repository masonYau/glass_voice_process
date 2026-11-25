"""用公开语音样本合成"智能眼镜视角"的测试对话（带真值），用于回归测试与调参。不包含、也不提交任何音频。

佩戴者：近讲、声音大、无混响；对话者：1~2 m 外、小 20~30 dB、有房间混响；背景：可选噪声文件或白噪声。

用法
  python tools/make_test_conversation.py --wearer a.wav b.wav --partner c.wav [--partner2 d.wav] --out test_conv.wav
  # 之后：GLASS_VOICE_TEST_CONVERSATION=test_conv.wav pytest
输出：test_conv.wav 与 test_conv_truth.csv（speaker,start,end）
"""
import argparse

import numpy as np
import soundfile as sf

from glass_voice_process.audio import load_audio

SR = 16000


def set_rms_db(x, d):
    return x * (10 ** (d / 20) / (np.sqrt(np.mean(x ** 2)) + 1e-12))


def synth_rir(rt60=0.45, length=0.6, seed=0):
    rng = np.random.default_rng(seed)
    n = int(length * SR)
    t = np.arange(n) / SR
    rir = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    rir[: int(0.003 * SR)] = 0
    rir[0] = 1.0
    return rir / np.max(np.abs(rir))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wearer", nargs="+", required=True, help="佩戴者的语音样本（1~多条）")
    ap.add_argument("--partner", nargs="+", required=True, help="对话者 A 的语音样本")
    ap.add_argument("--partner2", nargs="*", default=[], help="可选：对话者 B")
    ap.add_argument("--noise", help="可选：背景噪声文件（默认白噪声）")
    ap.add_argument("--wearer-db", type=float, default=-16)
    ap.add_argument("--partner-db", type=float, default=-40)
    ap.add_argument("--noise-db", type=float, default=-52)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    W = [load_audio(p) for p in a.wearer]
    A = [load_audio(p) for p in a.partner]
    B = [load_audio(p) for p in a.partner2]
    far = lambda x, rir: np.convolve(x, rir)[: len(x) + int(0.3 * SR)]
    rir_a, rir_b = synth_rir(0.45, seed=1), synth_rir(0.6, seed=2)
    turns = []
    for i in range(4):
        turns.append(("wearer", W[i % len(W)], a.wearer_db, None))
        if B and i % 2:
            turns.append(("partner_B", far(B[i % len(B)], rir_b), a.partner_db + 2, True))
        else:
            turns.append(("partner_A", far(A[i % len(A)], rir_a), a.partner_db, True))
    gaps = [0.5, 0.6, 0.5, 0.7, 0.4, 0.5, 0.5, 0.6]
    total = int(sum(len(x) / SR + g for (_, x, _, _), g in zip(turns, gaps)) * SR) + SR
    mix = np.zeros(total, np.float32)
    rows, t = [], 0.0
    for (spk, x, lvl, reverb), g in zip(turns, gaps):
        t += g
        y = set_rms_db(x, lvl)
        st = int(t * SR)
        mix[st: st + len(y)] += y
        dur = (len(x) - (int(0.3 * SR) if reverb else 0)) / SR  # 真值不含混响拖尾
        rows.append((spk, t, t + dur))
        t += len(x) / SR
    noise = load_audio(a.noise) if a.noise else np.random.default_rng(0).standard_normal(total).astype(np.float32)
    noise = np.tile(noise, int(np.ceil(total / len(noise))))[:total]
    mix = mix + set_rms_db(noise, a.noise_db)
    sf.write(a.out, mix, SR, subtype="PCM_16")
    with open(a.out[:-4] + "_truth.csv", "w", encoding="utf-8") as f:
        f.write("speaker,start,end\n")
        for spk, s, e in rows:
            f.write(f"{spk},{s:.2f},{e:.2f}\n")
    print("wrote", a.out, f"{len(mix) / SR:.1f}s")


if __name__ == "__main__":
    main()
