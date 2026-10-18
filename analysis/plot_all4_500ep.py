import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {
    "ppo": "#1f77b4", "gppo": "#2ca02c",
    "sac": "#9467bd", "gsac": "#d62728",
}
LABELS = {"ppo": "PPO", "gppo": "GPPO", "sac": "SAC", "gsac": "GSAC"}


def smooth(x, w=15):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x, np.arange(1, len(x) + 1)
    kernel = np.ones(w) / w
    sm = np.convolve(x, kernel, mode="valid")
    eps = np.arange(w, len(x) + 1)
    return sm, eps


def first_nan_ep(r):
    r = np.asarray(r, dtype=float)
    idx = np.where(np.isnan(r))[0]
    return int(idx[0]) + 1 if len(idx) else None


def main():
    R = json.load(open("results/results_500_all8_merged.json"))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # ---- Panel A: Re=45, all 4 algorithms, smoothed ----
    ax = axes[0]
    for name in ["ppo_low", "gppo_low", "sac_low", "gsac_low"]:
        algo = name.split("_")[0]
        r = np.array(R[name]["episode_rewards"], dtype=float)
        eps = np.arange(1, len(r) + 1)
        ax.plot(eps, r, color=COLORS[algo], alpha=0.12, lw=0.7)
        sm, sm_eps = smooth(r, w=15)
        ax.plot(sm_eps, sm, color=COLORS[algo], lw=2.2, label=LABELS[algo])
    ax.set_xlabel("Training episodes [-]")
    ax.set_ylabel("Reward [-]")
    ax.set_title("Re=45 (regular flow) — 4 algorithms, 500 episodes")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim(0, 500)

    # ---- Panel B: Re=300, all 4 algorithms, raw + NaN-onset markers ----
    ax = axes[1]
    max_nan_ep = 0
    nan_eps = {}
    for name in ["ppo_high", "gppo_high", "sac_high", "gsac_high"]:
        algo = name.split("_")[0]
        r = np.array(R[name]["episode_rewards"], dtype=float)
        eps = np.arange(1, len(r) + 1)
        ax.plot(eps, r, color=COLORS[algo], alpha=0.75, lw=1.1, marker="o", ms=2.5, label=LABELS[algo])
        fn = first_nan_ep(r)
        if fn:
            nan_eps[algo] = fn
            max_nan_ep = max(max_nan_ep, fn)
            ax.axvline(fn, color=COLORS[algo], ls="--", lw=1.6, alpha=0.8)

    ax.set_xlabel("Training episodes [-]")
    ax.set_ylabel("Reward [-]")
    ax.set_title("Re=300 (chaotic flow) — 4 algorithms, first-NaN episode marked with dashed line")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim(0, max(max_nan_ep + 30, 30))

    txt = "  ".join(f"{LABELS[a]} ep{e}" for a, e in sorted(nan_eps.items(), key=lambda kv: kv[1]))
    ax.text(0.02, 0.03, "First NaN: " + txt, transform=ax.transAxes, fontsize=8.5,
            bbox=dict(boxstyle="round", fc="wheat", alpha=0.85))

    plt.tight_layout()
    plt.savefig("plots/all4_500ep_comparison.png", dpi=150)
    print("Saved plots/all4_500ep_comparison.png")

    # ---- Standalone stability-delay bar chart ----
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    order = ["ppo", "sac", "gppo", "gsac"]
    vals = [nan_eps[a] for a in order]
    bars = ax2.bar([LABELS[a] for a in order], vals,
                   color=[COLORS[a] for a in order])
    ppo_ep = nan_eps["ppo"]
    for b, a in zip(bars, order):
        mult = nan_eps[a] / ppo_ep
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 3,
                  f"ep{nan_eps[a]}\n({mult:.1f}x)", ha="center", fontsize=9)
    ax2.set_ylabel("First-NaN episode at Re=300 [-]")
    ax2.set_title("Time-to-instability in the chaotic regime\n(delay multiplier relative to PPO)")
    plt.tight_layout()
    plt.savefig("plots/stability_delay_500ep.png", dpi=150)
    print("Saved plots/stability_delay_500ep.png")

    print("\nFirst-NaN episodes at Re=300:", nan_eps)
    print("Delay multipliers vs PPO:", {a: round(nan_eps[a] / ppo_ep, 2) for a in order})
    print("SAC/PPO (off-policy alone):", round(nan_eps["sac"] / nan_eps["ppo"], 2))
    print("GPPO/PPO (analytic grad alone, on-policy):", round(nan_eps["gppo"] / nan_eps["ppo"], 2))
    print("GSAC/SAC (analytic grad alone, off-policy):", round(nan_eps["gsac"] / nan_eps["sac"], 2))
    print("GSAC/GPPO (off-policy alone, with analytic grad):", round(nan_eps["gsac"] / nan_eps["gppo"], 2))


if __name__ == "__main__":
    main()
