import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"ppo": "#1f77b4", "gppo": "#2ca02c", "sac": "#9467bd", "gsac": "#d62728"}
LABELS = {"ppo": "PPO", "gppo": "GPPO", "sac": "SAC", "gsac": "GSAC"}
ORDER = ["ppo", "sac", "gppo", "gsac"]

TRIGGER = {"ppo": 12, "sac": 36, "gppo": 83, "gsac": 141}
COLLAPSE = {"ppo": 12, "sac": 36, "gppo": 83, "gsac": 144}
BAD_FRACTION_PCT = {"sac": 2.8, "gsac": 0.7}  # off-policy only (buffer dilution)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ---- Panel A: trigger episode vs permanent-collapse episode ----
    ax = axes[0]
    x = np.arange(len(ORDER))
    w = 0.35
    trig_vals = [TRIGGER[a] for a in ORDER]
    coll_vals = [COLLAPSE[a] for a in ORDER]
    ax.bar(x - w / 2, trig_vals, w, label="Trigger episode (first NaN)",
           color=[COLORS[a] for a in ORDER], alpha=0.55)
    ax.bar(x + w / 2, coll_vals, w, label="Permanent-collapse episode",
           color=[COLORS[a] for a in ORDER], alpha=1.0)
    for i, a in enumerate(ORDER):
        recovered = COLLAPSE[a] - TRIGGER[a] - 1
        if recovered > 0:
            ax.annotate(f"{recovered} recovered ep.", xy=(x[i] + w / 2, COLLAPSE[a] + 3),
                        ha="center", fontsize=9, color=COLORS[a], fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[a] for a in ORDER])
    ax.set_ylabel("Episode index [-]")
    ax.set_title("Trigger vs. permanent-collapse episode\n(Re=300, seed=0)")
    ax.legend(fontsize=8.5, loc="upper left")

    # ---- Panel B: replay-buffer "bad transition" fraction at the trigger episode ----
    ax = axes[1]
    algos = ["sac", "gsac"]
    fracs = [BAD_FRACTION_PCT[a] for a in algos]
    bars = ax.bar([LABELS[a] for a in algos], fracs, color=[COLORS[a] for a in algos])
    for b, f in zip(bars, fracs):
        ax.text(b.get_x() + b.get_width() / 2, f + 0.05, f"{f:.1f}%",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Bad-transition fraction of replay buffer at trigger [%]")
    ax.set_title("Buffer-dilution effect\n(off-policy algorithms only)")
    ax.set_ylim(0, max(fracs) * 1.4)

    plt.tight_layout()
    plt.savefig("plots/mechanism_analysis.png", dpi=150)
    print("Saved plots/mechanism_analysis.png")


if __name__ == "__main__":
    main()
