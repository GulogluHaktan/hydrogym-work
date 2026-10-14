"""Figure generation for the journal manuscript (n=5, seeds 0-4).
Outputs vector PDFs into paper_tex/figures/ for \\includegraphics.
Run: ../venv/bin/python3 plot_paper_n5.py
"""
import json
import math
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
})

COLORS = {"ppo": "#1f77b4", "gppo": "#2ca02c", "sac": "#9467bd", "gsac": "#d62728"}
LABELS = {"ppo": "PPO", "gppo": "GPPO", "sac": "SAC", "gsac": "GSAC"}
ORDER = ["ppo", "sac", "gppo", "gsac"]
SEEDS = [0, 1, 2, 3, 4]
OUT = "paper_tex/figures"
os.makedirs(OUT, exist_ok=True)


def load(seed, cfg):
    fn = f"results_500_{cfg}.json" if seed == 0 else f"results_500_seed{seed}_{cfg}.json"
    if not os.path.exists(fn):
        fn2 = f"results_500_seed{seed}_{cfg}.json"
        fn = fn2 if os.path.exists(fn2) else fn
    d = json.load(open(fn))
    if cfg in d:
        return d[cfg]
    return next(iter(d.values()))


def first_nan(rs):
    rs = np.asarray(rs, dtype=float)
    idx = np.where(~np.isfinite(rs))[0]
    return int(idx[0]) + 1 if len(idx) else None


def smooth(x, w=15):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return np.arange(1, len(x) + 1), x
    k = np.ones(w) / w
    sm = np.convolve(x, k, mode="valid")
    return np.arange(w, len(x) + 1), sm


# ---------------------------------------------------------------------------
# Figure: Re=45 learning curves (seed-averaged, min-max band), on/off-policy panels
# ---------------------------------------------------------------------------
def fig_re45_curves():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    groups = [("On-policy", ["ppo", "gppo"]), ("Off-policy", ["sac", "gsac"])]
    for ax, (title, algos) in zip(axes, groups):
        for a in algos:
            curves = []
            for s in SEEDS:
                r = np.asarray(load(s, f"{a}_low")["episode_rewards"], dtype=float)
                ep, sm = smooth(r, 15)
                curves.append(sm)
            L = min(len(c) for c in curves)
            curves = np.vstack([c[:L] for c in curves])
            ep = ep[:L]
            mean = np.nanmean(curves, axis=0)
            lo, hi = np.nanmin(curves, axis=0), np.nanmax(curves, axis=0)
            ax.plot(ep, mean, color=COLORS[a], label=LABELS[a], lw=1.6)
            ax.fill_between(ep, lo, hi, color=COLORS[a], alpha=0.15, lw=0)
        ax.set_title(f"{title} (Re = 45)")
        ax.set_xlabel("Episode")
        ax.legend(loc="lower right", frameon=True)
    axes[0].set_ylabel("Smoothed episode reward")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_re45_curves.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure: Re=45 final performance bars (mean +/- std over 5 seeds)
# ---------------------------------------------------------------------------
def fig_re45_final():
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    means, stds = [], []
    for a in ORDER:
        vals = []
        for s in SEEDS:
            r = np.asarray(load(s, f"{a}_low")["episode_rewards"], dtype=float)
            vals.append(np.nanmean(r[-5:]))
        means.append(np.mean(vals))
        stds.append(np.std(vals, ddof=1))
    x = np.arange(len(ORDER))
    ax.bar(x, means, yerr=stds, capsize=3,
           color=[COLORS[a] for a in ORDER], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[a] for a in ORDER])
    ax.set_ylabel("Final reward (last-5-ep. mean)")
    ax.set_title("Re = 45 final performance\n(mean $\\pm$ s.d., $n=5$ seeds)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_re45_final.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure: Re=300 breaking point -- strip plot, 4 algos x 5 seeds (THE figure)
# ---------------------------------------------------------------------------
def fig_re300_breakpoint():
    bp = {a: [first_nan(load(s, f"{a}_high")["episode_rewards"]) for s in SEEDS]
          for a in ORDER}
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    rng = np.random.default_rng(0)
    for i, a in enumerate(ORDER):
        ys = np.array(bp[a], dtype=float)
        xs = i + (rng.random(len(ys)) - 0.5) * 0.22
        order = np.argsort(ys)
        for rank, j in enumerate(order):
            xs[j] = i + (rank - (len(ys) - 1) / 2) * 0.13
        ax.scatter(xs, ys, s=44, color=COLORS[a], edgecolor="k", lw=0.5, zorder=3)
        med = np.median(ys)
        ax.hlines(med, i - 0.32, i + 0.32, color="k", lw=2, zorder=4)
        ax.text(i + 0.36, med, f"md {med:.0f}", va="center", fontsize=8)
        for s, x, y in zip(SEEDS, xs, ys):
            ax.annotate(f"s{s}", (x, y), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=6.5, color="0.35")
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([LABELS[a] for a in ORDER])
    ax.set_ylabel("First-NaN episode at Re = 300")
    ax.set_title("Breaking point is seed-dominated, not algorithm-dominated\n"
                 "(each point = one seed; bar = median; K--W $p=0.57$, Friedman $p=0.77$)")
    ax.set_ylim(-6, 162)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_re300_breakpoint.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure: Re=300 example curves -- sudden precursor-free break
# ---------------------------------------------------------------------------
def fig_re300_curves():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    panels = [(4, ["sac", "gsac"], "seed = 4 (long runway)"),
              (1, ["ppo", "gppo"], "seed = 1")]
    for ax, (s, algos, title) in zip(axes, panels):
        for a in algos:
            r = np.asarray(load(s, f"{a}_high")["episode_rewards"], dtype=float)
            fn = first_nan(r)
            ax.plot(np.arange(1, len(r) + 1), r, color=COLORS[a], lw=0.8,
                    label=f"{LABELS[a]} (break @ {fn})")
            if fn:
                ax.axvline(fn, color=COLORS[a], ls=":", lw=1.2)
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.legend(loc="lower left", frameon=True)
        ax.set_xlim(0, 150)
    axes[0].set_ylabel("Episode reward")
    fig.suptitle("Re = 300: divergence is abrupt, with no ramp in the reward beforehand", y=1.03)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_re300_curves.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure: mechanism -- (a) flat pre-break window, (b) buffer-dilution survival prob
# ---------------------------------------------------------------------------
def _surv_prob(trigger, updates=10, batch=64, bad_steps=15, step_per_ep=15):
    buf = max(trigger * step_per_ep, bad_steps)
    good_frac = 1.0 - bad_steps / buf
    return good_frac ** (updates * batch)


def fig_mechanism():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))

    # (a) pre-break window, seed=4 gsac_high
    ax = axes[0]
    r = np.asarray(load(4, "gsac_high")["episode_rewards"], dtype=float)
    fn = first_nan(r)
    w0 = max(1, fn - 40)
    seg = r[w0 - 1:fn - 1]
    ep = np.arange(w0, fn)
    ax.plot(ep, seg, color=COLORS["gsac"], lw=1.0)
    ax.axhline(np.nanmean(seg), color="k", ls="--", lw=1,
              label=f"mean {np.nanmean(seg):.2f}, s.d. {np.nanstd(seg):.2f}")
    ax.axvline(fn, color="k", ls=":", lw=1.2, label=f"break @ {fn}")
    ax.set_title("(a) 40-episode window before the break\n(GSAC, Re = 300, seed = 4)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode reward")
    ax.legend(loc="lower left")

    # (b) survival probability vs trigger episode
    ax = axes[1]
    trig = np.arange(2, 210)
    ax.plot(trig, [_surv_prob(t) for t in trig], color="0.2", lw=1.6)
    obs = [("SAC s4", 131, (-38, -15), "left"),
           ("GSAC s1", 63, (6, -3), "left"),
           ("GSAC s4", 122, (-50, 7), "left"),
           ("GSAC s0", 141, (12, -13), "left")]
    for name, t, off, ha in obs:
        p = _surv_prob(t)
        ax.scatter([t], [p], s=34, zorder=3,
                   color=COLORS["gsac"] if "GSAC" in name else COLORS["sac"])
        ax.annotate(name, (t, p), textcoords="offset points", xytext=off,
                    fontsize=7, ha=ha)
    ax.scatter([141], [_surv_prob(141)], s=130, facecolors="none",
               edgecolors="k", lw=1.5, zorder=4, label="observed recovery window")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 3)
    ax.set_xlim(0, 210)
    ax.set_title("(b) P(no poisoned sample in 640 draws)\nvs. trigger episode")
    ax.set_xlabel("Trigger episode")
    ax.set_ylabel("Survival probability")
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_mechanism.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure: analytic-gradient norm stability, Re=45 vs Re=300
# ---------------------------------------------------------------------------
def fig_gradnorm():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=False)
    for ax, (sfx, ttl) in zip(axes, [("low", "Re = 45 (regular)"),
                                     ("high", "Re = 300 (chaotic)")]):
        for a in ["gppo", "gsac"]:
            for s in SEEDS:
                d = load(s, f"{a}_{sfx}")
                g = np.asarray(d["grad_norms"], dtype=float)
                r = np.asarray(d["episode_rewards"], dtype=float)
                fn = first_nan(r)
                if fn:
                    g = g[:fn - 1]
                ax.plot(np.arange(1, len(g) + 1), g, color=COLORS[a], lw=0.7,
                        alpha=0.7)
        ax.set_title(ttl)
        ax.set_xlabel("Episode")
        ax.set_xlim(0, 150 if sfx == "high" else 500)
    axes[0].set_ylabel("Analytic-gradient norm $\\|g_{\\mathrm{an}}\\|$")
    from matplotlib.lines import Line2D
    axes[1].legend(handles=[Line2D([0], [0], color=COLORS["gppo"], label="GPPO"),
                            Line2D([0], [0], color=COLORS["gsac"], label="GSAC")],
                   loc="upper right")
    fig.suptitle("Analytic-gradient norm stays bounded even up to the break", y=1.03)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_gradnorm.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_re45_curves()
    fig_re45_final()
    fig_re300_breakpoint()
    fig_re300_curves()
    fig_mechanism()
    fig_gradnorm()
    # print the breaking-point table for cross-check
    print("algo   " + " ".join(f"s{s}" for s in SEEDS) + "   median mean")
    for a in ORDER:
        v = [first_nan(load(s, f"{a}_high")["episode_rewards"]) for s in SEEDS]
        va = np.array(v, dtype=float)
        print(f"{a:5s}  " + " ".join(f"{x:3d}" for x in v) +
              f"   {np.median(va):5.0f} {va.mean():5.1f}")
    print("figures written to", OUT)
