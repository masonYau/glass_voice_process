"""说话人划分（v3/v4）：佩戴者/他人二分（电平 + 直混比）、对话者聚类（CAM++/ECAPA/MFCC）、段内换人切分。"""
from __future__ import annotations

import numpy as np

from .audio import HOP, SR
from .embedding import embed_segments, window_feats
from .utils import Segment, merge_short, runs


def split_wearer(levels, drr, min_gap_db=6.0):
    """GMM 二分；返回 (is_wearer, 两类电平差)；差距 < min_gap_db 返回 None。"""
    from sklearn.mixture import GaussianMixture
    if len(levels) < 4:
        return None
    X = np.c_[levels, drr * 0.5]
    gm = GaussianMixture(2, random_state=0, n_init=5).fit(X)
    lab = gm.predict(X)
    hi = int(np.argmax(gm.means_[:, 0]))
    gap = gm.means_[hi, 0] - gm.means_[1 - hi, 0]
    if gap < min_gap_db:
        return None
    return lab == hi, float(gap)


def cluster_mfcc(emb, n=None, threshold=None, max_k=4, min_sil=0.2, min_windows=10):
    """MFCC 统计的保守聚类：k=2..max_k 中轮廓系数最高且 ≥ min_sil、每类 ≥ min_windows 才接受，否则视为 1 人。"""
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    if len(emb) <= 2:
        return np.zeros(len(emb), int)
    e = StandardScaler().fit_transform(emb)
    e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-9
    if n:
        return AgglomerativeClustering(n_clusters=min(n, len(e)), metric="cosine", linkage="average").fit_predict(e)
    if threshold is not None:
        return AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, metric="cosine",
                                       linkage="average").fit_predict(e)
    best, best_sil = np.zeros(len(e), int), min_sil
    for k in range(2, min(max_k, len(e) // min_windows) + 1):
        lab = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(e)
        if np.bincount(lab).min() < min_windows:
            continue
        sil = silhouette_score(e, lab, metric="cosine")
        if sil >= best_sil:
            best, best_sil = lab, sil
    return best


def cluster_neural(E, dur, n=None, sim_thr=0.45, min_dur=4.0, merge_sim=0.5):
    """神经嵌入两步聚类：① 平均链接 AHC（簇间平均相似度 < sim_thr 才分开）；
    ② 簇质心相似度 ≥ merge_sim 的合并；总时长 < min_dur 秒的小簇并入最相似的簇。返回 (标签, 诊断)。"""
    from sklearn.cluster import AgglomerativeClustering
    dur = np.asarray(dur, float)
    if len(E) < 2:
        lab = np.zeros(len(E), int)
    elif n:
        lab = AgglomerativeClustering(n_clusters=min(n, len(E)), metric="cosine", linkage="average").fit_predict(E)
    else:
        lab = AgglomerativeClustering(n_clusters=None, distance_threshold=1 - sim_thr, metric="cosine",
                                      linkage="average").fit_predict(E)

    def cen(c):
        v = E[lab == c].mean(0)
        return v / (np.linalg.norm(v) + 1e-9)
    while not n and merge_sim is not None and len(set(lab)) > 1:
        ids = sorted(set(lab))
        C = {c: cen(c) for c in ids}
        a, b = max(((a, b) for i, a in enumerate(ids) for b in ids[i + 1:]), key=lambda p: C[p[0]] @ C[p[1]])
        if C[a] @ C[b] < merge_sim:
            break
        lab[lab == b] = a
    while not n and len(set(lab)) > 1:
        ids = sorted(set(lab))
        d = {c: dur[lab == c].sum() for c in ids}
        small = min(ids, key=lambda c: d[c])
        if d[small] >= min_dur:
            break
        C = {c: cen(c) for c in ids}
        lab[lab == small] = max((c for c in ids if c != small), key=lambda c: C[small] @ C[c])
    ids = sorted(set(lab))
    lab = np.array([ids.index(c) for c in lab])
    cens = np.array([E[lab == c].mean(0) for c in range(len(ids))]) if len(E) else np.zeros((0, 1))
    if len(cens):
        cens /= np.linalg.norm(cens, axis=1, keepdims=True) + 1e-9
    diag = {"k": len(ids), "dur_s": [round(float(dur[lab == c].sum()), 1) for c in range(len(ids))],
            "centroid_sim": np.round(cens @ cens.T, 3).tolist() if len(cens) else []}
    return lab, diag


def partner_units(wins, idx, max_len=3.0, chunk=2.0):
    """把相邻/重叠的对话者窗口合并成连续片段，长片段切成约 chunk 秒。返回帧区间和窗口所属单元。"""
    units, owner = [], {}
    for i in sorted(idx, key=lambda i: wins[i][0]):
        f0, f1 = wins[i]
        if units and f0 <= units[-1][1]:
            units[-1][1] = max(units[-1][1], f1)
            units[-1][2].append(i)
        else:
            units.append([f0, f1, [i]])
    out = []
    C, M = int(chunk / HOP), int(max_len / HOP)
    for f0, f1, members in units:
        if f1 - f0 <= M:
            pieces = [(f0, f1)]
        else:
            ed = np.linspace(f0, f1, max(1, round((f1 - f0) / C)) + 1).astype(int)
            pieces = list(zip(ed[:-1], ed[1:]))
        for i in members:
            c = (wins[i][0] + wins[i][1]) / 2
            owner[i] = len(out) + next(k for k, (a, b) in enumerate(pieces) if c < b or k == len(pieces) - 1)
        out.extend(pieces)
    return out, owner


def remap_by_time(lab, starts, prefix="partner_"):
    order = {}
    for l, s in sorted(zip(lab, starts), key=lambda t: t[1]):
        if l not in order and l != -1:
            order[l] = f"{prefix}{len(order) + 1}"
    return [order.get(l, "wearer") for l in lab]


def frames_from_windows(wins, labels, nframes, mask, smooth=0.3):
    """窗口投票 → 帧标签（0.3 s 平滑）→ 段。"""
    names = sorted(set(labels))
    votes = np.zeros((nframes, len(names)))
    for (f0, f1), l in zip(wins, labels):
        votes[f0:f1, names.index(l)] += 1
    k = max(1, int(smooth / HOP))
    for j in range(len(names)):
        votes[:, j] = np.convolve(votes[:, j], np.ones(k) / k, mode="same")
    fl = np.where(mask, votes.argmax(1), -1)
    fl[(votes.sum(1) == 0) & mask] = -1
    return merge_short([Segment(s * HOP, e * HOP, names[v]) for v, s, e in runs(fl) if v >= 0])


def diarize(x_dn, lv, mask, embedder=None, num_partners=None, partner_threshold=None, partner_sim=None,
            partner_merge=None, split_at=None, log=print):
    """embedder：CamPlus / Ecapa 实例，None 表示用 MFCC。split_at：换人点帧（先切开再切窗口）。
    返回 (段列表, 诊断字典)。"""
    diag = {}
    wmask = mask
    if split_at is not None and np.any(split_at):
        from scipy.ndimage import binary_dilation
        wmask = mask & ~binary_dilation(split_at[:len(mask)], iterations=1)
    wins, levels, drr, timbre = window_feats(x_dn, lv, wmask)
    if not wins:
        return [], diag
    labels = np.empty(len(wins), object)
    res = split_wearer(levels, drr)
    if res is not None:
        is_w, gap = res
        diag["wearer_gap_db"] = round(gap, 1)
        log(f"[diarize] 佩戴者/他人 电平差 ≈ {gap:.1f} dB")
        labels[is_w] = "wearer"
        idx = np.where(~is_w)[0]
        if len(idx):
            if embedder is not None:
                units, owner = partner_units(wins, idx)
                E = embed_segments(embedder, x_dn, [(a * int(HOP * SR), b * int(HOP * SR)) for a, b in units])
                ul, cd = cluster_neural(E, [(b - a) * HOP for a, b in units], num_partners,
                                        partner_sim if partner_sim is not None else embedder.partner_sim,
                                        merge_sim=partner_merge if partner_merge is not None else embedder.partner_merge)
                diag.update(cd, embedder=embedder.name, n_units=len(units))
                pl = np.array([ul[owner[i]] for i in idx])
            else:
                pl = cluster_mfcc(timbre[idx], num_partners, partner_threshold)
                diag["embedder"] = "mfcc"
            labels[idx] = remap_by_time(pl, [wins[i][0] for i in idx])
    else:
        log("[diarize] 电平差不明显，改用音色聚类，最响的一类视为佩戴者")
        pl = cluster_mfcc(timbre, (num_partners + 1) if num_partners else None, partner_threshold)
        loud = max(set(pl), key=lambda c: np.median(levels[pl == c]))
        names = remap_by_time(np.where(pl == loud, -1, pl), [w[0] for w in wins])
        labels[:] = ["wearer" if c == loud else nm for c, nm in zip(pl, names)]
    return frames_from_windows(wins, labels, len(mask), mask), diag


def drop_unvoiced(segs, lv, vt, min_voiced=0.25):
    """说话人段级的谐波检查：去掉"佩戴者语音 + 紧跟碰撞/摩擦尾巴"被划出来的噪声段。"""
    fl = np.percentile(lv, 10)
    keep = []
    for s in segs:
        f0, f1 = int(s.start / HOP), int(s.end / HOP)
        act = lv[f0:f1] > fl + 3
        if act.sum() >= 5 and np.mean(vt[f0:f1][act] > 0.45) >= min_voiced:
            keep.append(s)
    return keep


def split_turns(segs, lv, vt, min_dur=0.3, margin=8.0, min_voiced=0.3, change=None, tol=0.4):
    """段内按电平切分换人：佩戴者段里连续 ≥ min_dur 秒、电平 ≤ 对话者电平 + margin、有谐波的部分改为对话者（反之亦然）。
    change 给出（pyannote 换人点）时，切分区间的起点或终点 ±tol 秒内必须有换人点佐证。返回 (段, 切分次数)。"""
    def lvl(label):
        fr = [lv[int(s.start / HOP): int(s.end / HOP)] for s in segs if s.label == label]
        fr = np.concatenate(fr) if fr else np.array([])
        return float(np.percentile(fr, 80)) if len(fr) > 20 else None
    partners = [s for s in segs if s.label != "wearer"]
    pl = [lvl(l) for l in {s.label for s in partners}]
    pl = [v for v in pl if v is not None]
    Lw, Lp = lvl("wearer"), (float(np.median(pl)) if pl else None)
    if Lw is None or Lp is None or Lw - Lp < 15:
        return segs, 0
    k = max(1, int(0.15 / HOP))
    sm = np.convolve(lv, np.ones(k) / k, mode="same")
    floor = np.percentile(lv, 10)
    T = int(tol / HOP)
    out, nsplit = [], 0
    for s in segs:
        f0, f1 = int(s.start / HOP), int(s.end / HOP)
        cand = ((sm[f0:f1] <= Lp + margin) & (lv[f0:f1] > floor + 6)) if s.label == "wearer" else (sm[f0:f1] >= Lw - margin)
        pieces, i = [], 0
        while i < len(cand):
            if not cand[i]:
                i += 1
                continue
            j = i
            while j < len(cand) and (cand[j] or (j + 10 < len(cand) and cand[j:j + 10].any())):
                j += 1
            ok = (j - i) * HOP >= min_dur and np.mean(vt[f0 + i:f0 + j] > 0.45) >= min_voiced
            if ok and change is not None:
                near = lambda c: bool(change[max(0, c - T): c + T + 1].any())
                ok = near(f0 + i) or near(f0 + j)
            if ok:
                pieces.append((f0 + i, f0 + j))
            i = j
        if not pieces:
            out.append(s)
            continue
        nsplit += len(pieces)
        cur = f0
        for a, b in pieces:
            if a > cur:
                out.append(Segment(cur * HOP, a * HOP, s.label))
            if s.label == "wearer":
                mid = (a + b) / 2 * HOP
                near_p = min(partners, key=lambda p: min(abs(p.start - mid), abs(p.end - mid)))
                out.append(Segment(a * HOP, b * HOP, near_p.label))
            else:
                out.append(Segment(a * HOP, b * HOP, "wearer"))
            cur = b
        if cur < f1:
            out.append(Segment(cur * HOP, f1 * HOP, s.label))
    out = [s for s in out if s.end - s.start >= 0.05]
    return merge_short(sorted(out, key=lambda s: s.start)), nsplit
