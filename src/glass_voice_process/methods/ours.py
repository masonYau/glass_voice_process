"""v3 / v4：我们的流程。
v3 = 降噪 → 双路 Silero（滞回）→ 谐波检查 → 佩戴者/他人（电平+直混比）→ CAM++ 聚类 → 段内电平切分 → 增益流程
v4 = v3，但语音检测改用 pyannote segmentation-3.0，并在其换人点切分、标记重叠（段内电平切分需换人点佐证）。"""
from __future__ import annotations

import numpy as np

from .. import models
from ..audio import SR, frame_rms_db, n_frames
from ..config import Config
from ..denoise import denoise, noise_mask
from ..diarize import diarize, drop_unvoiced, split_turns
from ..gain import apply_gains, limiter, suppress_clicks
from ..result import Result
from ..utils import pad_to
from ..vad import SileroVAD, refine_mask, vad_dual, vad_silero, voicing_track, webrtc_vad  # noqa: F401

_CACHE = {}


def _cached(key, factory):
    if key not in _CACHE:
        _CACHE[key] = factory()
    return _CACHE[key]


def _embedder(cfg: Config, log):
    if cfg.embedder == "mfcc":
        return None
    try:
        if cfg.embedder == "campplus":
            from ..embedding import CamPlus
            p = models.model_path("campplus", cfg.model_dir)
            return _cached(("campplus", str(p)), lambda: CamPlus(p))
        if cfg.embedder == "ecapa":
            from ..embedding import Ecapa
            d = models.model_dir(cfg.model_dir) / "ecapa"
            return _cached(("ecapa", str(d)), lambda: Ecapa(d))
    except (models.ModelNotFound, ImportError) as ex:
        log(f"[embedder] {cfg.embedder} 不可用，回退到 mfcc：{ex}")
        return None
    raise ValueError(f"未知 embedder {cfg.embedder}")


def run(x: np.ndarray, cfg: Config, log=print) -> Result:
    if cfg.method == "v3":
        cfg.segmenter = "none"
    stats = {}
    x_dn = denoise(x, cfg.denoise, noise_mask(x)) if cfg.denoise > 0 else x
    lv = frame_rms_db(x_dn)
    nf = n_frames(len(x))

    # ---- 语音检测
    seg_py = chg = None
    if cfg.vad == "webrtc":
        mask = webrtc_vad(x_dn)
    elif cfg.segmenter == "pyannote":
        mask = None  # 只用 pyannote，省去 Silero
    else:
        sil = _cached(("silero", str(models.model_path("silero_vad", cfg.model_dir))),
                      lambda: SileroVAD(models.model_path("silero_vad", cfg.model_dir)))
        mask = vad_dual(sil, x, x_dn, cfg.vad_threshold, cfg.vad_offset, cfg.vad_pad) if cfg.vad == "dual" \
            else vad_silero(sil, x_dn, cfg.vad_threshold)
    if cfg.segmenter != "none":
        from ..segmentation import PyannoteSegmenter, change_points, speech_mask
        p = models.model_path("pyannote_segmentation", cfg.model_dir)
        seg_py = _cached(("pyannote", str(p)), lambda: PyannoteSegmenter(p))(x)
        pm = pad_to(speech_mask(seg_py, cfg.py_onset, cfg.py_offset), nf)
        if cfg.segmenter == "pyannote":
            mask = pm
        elif cfg.segmenter == "union":
            mask = mask | pm
        chg = pad_to(change_points(seg_py, cfg.change_thr), nf)
        stats["change_points"] = int(chg.sum())
        stats["overlap_ratio"] = float(np.mean(seg_py["overlap"] > 0.5))
    mask = pad_to(mask, nf)
    stats["speech_ratio_detected"] = float(mask.mean())
    log(f"[{cfg.method}] 语音检测（{cfg.segmenter if cfg.segmenter != 'none' else cfg.vad}）：语音占 {mask.mean():.1%}")

    vt = voicing_track(x_dn)
    if cfg.min_voiced > 0:
        mask, nd = refine_mask(mask, lv, vt, cfg.min_voiced)
        stats["transients_dropped"] = nd
    noise_db = float(np.percentile(lv[~mask], 20)) if (~mask).sum() > 100 else None
    stats["noise_floor_db"] = noise_db

    # ---- 说话人划分
    emb = _embedder(cfg, log)
    segs, diag = diarize(x_dn, lv, mask, emb, cfg.num_partners, cfg.partner_threshold, cfg.partner_sim,
                         cfg.partner_merge, split_at=chg, log=log)
    stats["diarize"] = diag
    if cfg.min_voiced > 0:
        segs = drop_unvoiced(segs, lv, vt, cfg.min_voiced)
    if cfg.turn_split and segs:
        segs, nsp = split_turns(segs, lv, vt, change=chg)
        stats["turn_splits"] = nsp
    if seg_py is not None:
        ov = pad_to(seg_py["overlap"] > 0.5, nf)
        for s in segs:
            f0 = int(s.start / 0.01)
            f1 = max(f0 + 1, int(s.end / 0.01))
            s.overlap = float(np.mean(ov[f0:f1])) if f0 < nf else 0.0

    # ---- 增益
    y, gs, frame_gain, labeled = apply_gains(x_dn, segs, lv, cfg.target_db, cfg.max_boost, cfg.max_cut,
                                             noise_db=noise_db, noise_ceiling=cfg.noise_ceiling,
                                             bridge_gap=cfg.bridge_gap, release=cfg.release)
    y, nclk = suppress_clicks(y, gs)
    stats["clicks_suppressed"] = nclk
    y = limiter(y, cfg.limiter_ceiling).astype(np.float32)

    tracks = {}
    f = int(0.02 * SR)
    for sp in sorted({s.label for s in segs}):
        m = np.zeros(len(y))
        for s in segs:
            if s.label == sp:
                m[int(s.start * SR): int(s.end * SR)] = 1
        tracks[sp] = (y * np.convolve(m, np.ones(f) / f, mode="same")).astype(np.float32)
    stats["speech_ratio"] = float(sum(s.end - s.start for s in segs) / (len(x) / SR))
    return Result(cfg.method, x, y, tracks, segs, frame_gain.astype(np.float32), stats, cfg.to_dict())
