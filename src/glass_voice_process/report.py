"""处理前后波形 + 说话人分段的对比图（需要 matplotlib）。"""
from __future__ import annotations

import numpy as np

PALETTE = ["#e0812b", "#2fa36b", "#b0459b", "#c9b52a", "#17a2b8", "#8c564b", "#d62728"]


def plot_report(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x, y, sr = result.input, result.balanced, result.sr
    colors = {"wearer": "#2a6fdb"}
    t = np.arange(len(x)) / sr
    fig, ax = plt.subplots(2, 1, figsize=(12, 5), sharex=True, sharey=True)
    for a, sig, title in [(ax[0], x, "Before (raw)"), (ax[1], y, f"After (balanced, {result.method})")]:
        a.plot(t[::8], sig[::8], lw=0.4, color="#555")
        for s in result.segments:
            c = colors.setdefault(s.label, PALETTE[(len(colors) - 1) % len(PALETTE)])
            a.axvspan(s.start, s.end, color=c, alpha=0.18, lw=0)
        a.set_title(title, loc="left", fontsize=10)
        a.set_ylim(-1, 1)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.4) for c in colors.values()]
    ax[0].legend(handles, list(colors), loc="upper right", fontsize=8, ncol=len(colors))
    ax[1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
