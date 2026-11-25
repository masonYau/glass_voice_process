"""v5：Codex 方法（移植自 Mason 用 Codex 编写的 process_audio.py，算法保持一致）。

流程
  1. 检测信号 = 原始录音 90 Hz 高通 + 50 ms 块快速 AGC + tanh（不降噪）
  2. sherpa-onnx 说话人分离：pyannote segmentation-3.0 分段（窗口步长 0.25）+ CAM++ 声纹 + FastClustering（阈值 0.7），
     最短发言 0.18 s / 最短静音 0.2 s
  3. 每人一条轨：单人区间直接从原声按时间取出（10 ms 门控平滑）；两人重叠区间用 MossFormer2 估计两路声音，
     按声纹和邻近单人语音分配给两人（可选，需 mossformer2 模型）；可选复查短 A-B-A 插话
  4. 每轨一个固定增益：活跃帧 10 ms RMS 的第 70 百分位调向 target（-18 ~ +max_boost dB），混音 = 原声 + Σ 轨 ×（增益 − 1），
     未检出的部分保持原音量
  5. 混音与分轨共享 −1.5 dBFS 峰值限幅
与原脚本的差别：不写文件（由 Result.save 统一输出）；佩戴者取输入电平最高的一轨（推测），其余按轨号记为 partner_1..N。"""
from __future__ import annotations

from itertools import groupby

import numpy as np

from .. import models
from ..audio import SR, frame_rms_db, n_frames
from ..config import Config
from ..result import Result
from ..utils import Segment
from ..vad import fast_agc_signal

RATE = SR
HOP = 160  # 10 ms（采样点）


def db(value):
    return 20 * np.log10(np.maximum(value, 1e-10))


def blocks(x, size):
    return np.pad(x, (0, -len(x) % size)).reshape(-1, size)


def engines(cfg: Config):
    import sherpa_onnx as sherpa
    seg_model = str(models.model_path("pyannote_segmentation", cfg.model_dir))
    emb = sherpa.SpeakerEmbeddingExtractorConfig(model=str(models.model_path("campplus", cfg.model_dir)), num_threads=2)
    seg = sherpa.OfflineSpeakerSegmentationPyannoteModelConfig(model=seg_model, window_shift_ratio=0.25)
    config = sherpa.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa.OfflineSpeakerSegmentationModelConfig(pyannote=seg, num_threads=2),
        embedding=emb,
        clustering=sherpa.FastClusteringConfig(num_clusters=cfg.v5_num_speakers, threshold=cfg.v5_cluster_threshold),
        min_duration_on=0.18, min_duration_off=0.20)
    if not config.validate():
        raise ValueError("sherpa-onnx 配置无效，请检查模型路径")
    return sherpa.OfflineSpeakerDiarization(config), sherpa.SpeakerEmbeddingExtractor(emb)


def embedding(extractor, x):
    if len(x) < RATE // 2 or np.max(np.abs(x)) < 1e-5:
        return None
    stream = extractor.create_stream()
    stream.accept_waveform(RATE, x / max(np.max(np.abs(x)), 1e-5) * 0.8)
    stream.input_finished()
    if not extractor.is_ready(stream):
        return None
    vec = np.asarray(extractor.compute(stream))
    return vec / max(np.linalg.norm(vec), 1e-8)


def separate(session, wave):
    length = len(wave)
    padded = 2048 * max(1, (length - 8 + 2047) // 2048) + 8
    level = max(np.max(np.abs(wave)), 1e-5)
    inp = np.pad(wave / level, (0, padded - length)).astype(np.float32)
    pair = session.run(None, {"mix": inp[None]})[0][0, :, :length].copy()
    coeff = np.linalg.lstsq(pair.T, wave, rcond=1e-4)[0]  # 恢复两路声音的幅度和极性
    return pair * coeff[:, None]


def source_score(source, person, timeline, start, profiles, extractor):
    clean = timeline == (1 << person)
    mask = np.repeat(clean[start // HOP:(start + len(source) + HOP - 1) // HOP], HOP)[:len(source)]
    voice = embedding(extractor, source)
    similarity = float(voice @ profiles[person]) if voice is not None and person in profiles else 0.0
    if mask.sum() >= RATE // 8:
        local_energy = np.mean(source[mask] ** 2)
        similarity += 0.08 * np.clip(10 * np.log10((local_energy + 1e-12) / (np.mean(source ** 2) + 1e-12)), -18, 10)
    return similarity


def check_interruptions(x, tracks, timeline, turns, profiles, extractor, get_separator):
    """复查短 A-B-A 插话：一个人发言中间夹了另一个人的短插话时，尝试把延续的轻声分离出来。"""
    for turn in turns:
        p = turn["speaker"] - 1
        a, b = turn["start"], turn["end"]
        if p not in profiles or not 0.8 <= b - a <= 7:
            continue
        before = {s["speaker"] - 1 for s in turns if s["speaker"] != p + 1 and -0.35 <= a - s["end"] <= 1}
        after = {s["speaker"] - 1 for s in turns if s["speaker"] != p + 1 and -0.35 <= s["start"] - b <= 1}
        others = before & after & profiles.keys()
        if len(others) != 1:
            continue
        q = others.pop()
        first, last = int(round(a * RATE / HOP)), min(len(timeline), int(round(b * RATE / HOP)))
        lo, hi = first * HOP, min(len(x), last * HOP)
        left, right = max(0, lo - RATE), min(len(x), hi + RATE)
        pair = separate(get_separator(), x[left:right])[:, lo - left:hi - left]
        vectors = [embedding(extractor, src) for src in pair]
        if any(v is None for v in vectors):
            continue
        scores = np.array([[v @ profiles[k] for k in (p, q)] for v in vectors])
        if scores[0, 0] + scores[1, 1] < scores[1, 0] + scores[0, 1]:
            pair, scores = pair[::-1], scores[::-1]
        if min(scores.diagonal()) < 0.32 or min(scores[0, 0] - scores[0, 1], scores[1, 1] - scores[1, 0]) < 0.12:
            continue
        rms = np.sqrt(np.mean(blocks(pair[1], HOP) ** 2, axis=1))
        present = (rms > max(1e-5, 0.06 * np.percentile(rms, 85))) & (timeline[first:last] == (1 << p))
        env = np.interp(np.arange(hi - lo), HOP * np.arange(len(present)) + HOP // 2, present.astype("f"))
        env *= np.minimum(1., np.minimum(np.arange(hi - lo), np.arange(hi - lo)[::-1]) / HOP)
        for k, src in zip((p, q), pair):
            old = tracks[k][lo:hi]
            old[:] = old * (1 - env) + src * env
        timeline[first:last] |= present.astype(np.int64) << q


def make_tracks(x, diarizer, extractor, get_separator, recheck, log=print):
    segments = diarizer.process(fast_agc_signal(x)).sort_by_start_time()
    ids = list(dict.fromkeys(s.speaker for s in segments))
    if len(ids) > 62:
        raise ValueError("一段录音里聚出了超过 62 类")
    timeline = np.zeros((len(x) + HOP - 1) // HOP, dtype=np.int64)
    turns = []
    for s in segments:
        p = ids.index(s.speaker)
        start = max(0, int(np.floor(s.start * RATE / HOP)))
        end = min(len(timeline), int(np.ceil(s.end * RATE / HOP)))
        timeline[start:end] |= 1 << p
        turns.append(dict(speaker=p + 1, start=round(s.start, 3), end=round(min(s.end, len(x) / RATE), 3)))
    spans, index = [], 0
    for code, items in groupby(timeline):
        end = index + sum(1 for _ in items)
        spans.append((int(code), index * HOP, min(end * HOP, len(x))))
        index = end
    tracks, profiles = [], {}
    for p in range(len(ids)):
        mask = (timeline == (1 << p)).astype("f")
        env = np.interp(np.arange(len(x)), HOP * np.arange(len(mask)) + HOP // 2, mask)
        tracks.append((x * env).astype("f"))
        examples = sorted(((b - a, a, b) for c, a, b in spans if c == 1 << p and b - a >= RATE // 2), reverse=True)
        vectors = [embedding(extractor, x[a:min(b, a + 4 * RATE)]) for _, a, b in examples[:6]]
        vectors = [v for v in vectors if v is not None]
        if vectors:
            avg = np.mean(vectors, axis=0)
            profiles[p] = avg / max(np.linalg.norm(avg), 1e-8)
    separated = 0.0
    if get_separator is not None:
        for code, a, b in spans:
            if int(code).bit_count() != 2:
                continue
            people = [p for p in range(len(ids)) if code & (1 << p)]
            for lo in range(a, b, 6 * RATE):
                hi = min(b, lo + 6 * RATE)
                left, right = max(0, lo - RATE), min(len(x), hi + RATE)
                pair = separate(get_separator(), x[left:right])
                scores = np.array([[source_score(src, p, timeline, left, profiles, extractor) for p in people]
                                   for src in pair])
                if scores[0, 0] + scores[1, 1] < scores[0, 1] + scores[1, 0]:
                    pair = pair[::-1]
                fade = np.minimum(1., np.minimum(np.arange(1, hi - lo + 1), np.arange(hi - lo, 0, -1)) / (HOP / 2))
                for p, src in zip(people, pair):
                    old = tracks[p][lo:hi]
                    old[:] = old * (1 - fade) + src[lo - left:hi - left] * fade
                separated += (hi - lo) / RATE
    if recheck and get_separator is not None:
        check_interruptions(x, tracks, timeline, turns, profiles, extractor, get_separator)
        turns = []
        for p in range(len(tracks)):
            active = (timeline & (1 << p)) != 0
            boundaries = np.flatnonzero(np.diff(active.astype(int), prepend=0, append=0))
            for first, last in boundaries.reshape(-1, 2):
                turns.append(dict(speaker=p + 1, start=first * HOP / RATE, end=min(len(x), last * HOP) / RATE))
        turns.sort(key=lambda t: t["start"])
    return tracks, timeline, turns, separated


def level_of_speech(x, active):
    energy = np.sqrt(np.mean(blocks(x, HOP) ** 2, axis=1))
    values = energy[active & (energy > 1e-6)]
    return float(db(np.percentile(values, 70))) if values.size else None


def peak_limiter(mix, tracks=()):
    peaks = np.max(np.abs(blocks(mix, HOP)), axis=1)
    for track in tracks:
        peaks = np.maximum(peaks, np.max(np.abs(blocks(track, HOP)), axis=1))
    safe = np.minimum(1., 10 ** (-1.5 / 20) / np.maximum(peaks, 1e-8))
    from scipy import ndimage
    safe = ndimage.minimum_filter1d(safe, 3, mode="nearest")
    for i in range(1, len(safe)):
        safe[i] = min(safe[i], safe[i - 1] + 0.055)
    return np.interp(np.arange(len(mix)), HOP * np.arange(len(safe)) + HOP // 2, safe).astype("f")


def _separator_factory(cfg: Config, log):
    if cfg.v5_separation == "off":
        return None
    p = models.model_path("mossformer2", cfg.model_dir, required=(cfg.v5_separation == "on"))
    if p is None:
        log("[v5] 未找到 MossFormer2 模型，两人重叠区间不做分离（下载：glass-voice download-models --only mossformer2）")
        return None
    holder = {}

    def get():
        if "s" not in holder:
            import onnxruntime as ort
            o = ort.SessionOptions()
            o.intra_op_num_threads = 4
            o.enable_cpu_mem_arena = False
            holder["s"] = ort.InferenceSession(str(p), o, providers=["CPUExecutionProvider"])
        return holder["s"]
    return get


def run(x: np.ndarray, cfg: Config, log=print) -> Result:
    x = np.asarray(x, np.float32)
    diarizer, extractor = engines(cfg)
    get_sep = _separator_factory(cfg, log)
    tracks, timeline, turns, separated = make_tracks(x, diarizer, extractor, get_sep, cfg.v5_check_interruptions, log)
    mix = x.copy()
    stats_tracks = []
    for p, track in enumerate(tracks):
        active = (timeline & (1 << p)) != 0
        level = level_of_speech(track, active)
        gain = float(np.clip(cfg.target_db - level, -18, cfg.v5_max_boost)) if level is not None else 0.
        gain += float(cfg.v5_trim.get(p + 1, cfg.v5_trim.get(str(p + 1), 0.)))
        mix += track * (10 ** (gain / 20) - 1)
        track *= 10 ** (gain / 20)
        stats_tracks.append(dict(speaker=p + 1, active_seconds=round(active.sum() * HOP / RATE, 2),
                                 input_dbfs=round(level, 2) if level is not None else None, gain_db=round(gain, 2)))
    limiter = peak_limiter(mix, tracks)
    for track in tracks:
        track *= limiter
    mix = (mix * limiter).astype(np.float32)

    # ---- 统一成 Result：佩戴者 = 输入电平最高的一轨（推测），其余按轨号 partner_1..N
    known = [t for t in stats_tracks if t["input_dbfs"] is not None]
    wearer = max(known, key=lambda t: t["input_dbfs"])["speaker"] if known else None
    others = [t["speaker"] for t in stats_tracks if t["speaker"] != wearer]
    names = {wearer: "wearer", **{k: f"partner_{i + 1}" for i, k in enumerate(others)}}
    by_num = {t["speaker"]: t for t in stats_tracks}
    segs = []
    counts = np.array([int(v).bit_count() for v in timeline])
    for t in turns:
        st = by_num[t["speaker"]]
        f0, f1 = int(t["start"] * RATE / HOP), max(int(t["start"] * RATE / HOP) + 1, int(t["end"] * RATE / HOP))
        segs.append(Segment(t["start"], t["end"], names[t["speaker"]], st["input_dbfs"] or 0.0, st["gain_db"],
                            float(np.mean(counts[f0:f1] >= 2)) if f0 < len(counts) else 0.0))
    out_tracks = {names[p + 1]: tr.astype(np.float32) for p, tr in enumerate(tracks)}
    lx, ly = frame_rms_db(x), frame_rms_db(mix)
    from scipy.ndimage import median_filter
    est_gain = median_filter(ly - lx, size=5).astype(np.float32)  # 估算的逐帧增益（输出/输入电平差）
    stats = {"tracks": stats_tracks, "track_names": {str(k): v for k, v in names.items() if k is not None},
             "two_voice_seconds": round(float(np.count_nonzero(counts == 2) * HOP / RATE), 2),
             "three_or_more_seconds": round(float(np.count_nonzero(counts >= 3) * HOP / RATE), 2),
             "separated_seconds": round(separated, 2),
             "output_peak_dbfs": round(float(db(np.max(np.abs(mix)))), 2),
             "speech_ratio": float(np.mean(counts[:n_frames(len(x))] > 0))}
    return Result("v5", x, mix, out_tracks, segs, est_gain, stats, cfg.to_dict())
