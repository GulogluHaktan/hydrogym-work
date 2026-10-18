import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"ppo": "#1f77b4", "gppo": "#2ca02c", "sac": "#9467bd", "gsac": "#d62728"}
LABELS = {"ppo": "PPO", "gppo": "GPPO", "sac": "SAC", "gsac": "GSAC"}
ORDER = ["ppo", "sac", "gppo", "gsac"]


def first_nan_ep(r):
    r = np.asarray(r, dtype=float)
    idx = np.where(np.isnan(r))[0]
    return int(idx[0]) + 1 if len(idx) else None


def load_high(seed_suffix, algo):
    fname = f"results/results_500{seed_suffix}_{algo}_high.json"
    return json.load(open(fname))[f"{algo}_high"]["episode_rewards"]


def main():
    seed0 = {a: first_nan_ep(load_high("", a)) for a in ORDER}
    seed1 = {a: first_nan_ep(load_high("_seed1", a)) for a in ORDER}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # ---- Panel A: seed=0 vs seed=1 breaking point, grouped bars ----
    ax = axes[0]
    x = np.arange(len(ORDER))
    w = 0.35
    ax.bar(x - w / 2, [seed0[a] for a in ORDER], w, label="seed=0",
           color=[COLORS[a] for a in ORDER], alpha=0.55)
    ax.bar(x + w / 2, [seed1[a] for a in ORDER], w, label="seed=1",
           color=[COLORS[a] for a in ORDER], alpha=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[a] for a in ORDER])
    ax.set_ylabel("First-NaN episode at Re=300 [-]")
    ax.set_title("Breaking point: seed=0 vs. seed=1\n(same 4 algorithms, independent random seed)")
    ax.legend(fontsize=9)
    for i, a in enumerate(ORDER):
        ax.text(x[i] - w / 2, seed0[a] + 2, str(seed0[a]), ha="center", fontsize=8.5)
        ax.text(x[i] + w / 2, seed1[a] + 2, str(seed1[a]), ha="center", fontsize=8.5)

    # ---- Panel B: delay multiplier vs PPO, seed=0 vs seed=1 ----
    ax = axes[1]
    mult0 = {a: seed0[a] / seed0["ppo"] for a in ORDER}
    mult1 = {a: seed1[a] / seed1["ppo"] for a in ORDER}
    ax.bar(x - w / 2, [mult0[a] for a in ORDER], w, label="seed=0",
           color=[COLORS[a] for a in ORDER], alpha=0.55)
    ax.bar(x + w / 2, [mult1[a] for a in ORDER], w, label="seed=1",
           color=[COLORS[a] for a in ORDER], alpha=1.0)
    ax.axhline(1.0, color="gray", ls=":", lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[a] for a in ORDER])
    ax.set_ylabel("Delay multiplier vs. PPO [-]")
    ax.set_title("Delay multiplier vs. PPO: seed=0 vs. seed=1\n(dotted line = PPO reference, 1.0×)")
    ax.legend(fontsize=9)
    for i, a in enumerate(ORDER):
        ax.text(x[i] - w / 2, mult0[a] + 0.15, f"{mult0[a]:.2f}x", ha="center", fontsize=8)
        ax.text(x[i] + w / 2, mult1[a] + 0.15, f"{mult1[a]:.2f}x", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig("plots/seed_comparison.png", dpi=150)
    print("Saved plots/seed_comparison.png")
    print("seed=0 first-NaN:", seed0)
    print("seed=1 first-NaN:", seed1)
    print("seed=0 delay multipliers:", {k: round(v, 2) for k, v in mult0.items()})
    print("seed=1 delay multipliers:", {k: round(v, 2) for k, v in mult1.items()})


if __name__ == "__main__":
    main()
