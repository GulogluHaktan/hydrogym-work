import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def smooth(x, w=15):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x, np.arange(1, len(x) + 1)
    kernel = np.ones(w) / w
    sm = np.convolve(x, kernel, mode="valid")
    eps = np.arange(w, len(x) + 1)
    return sm, eps


def main():
    R = json.load(open("results_500_merged.json"))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # ---------------- Panel A: Re=45 (low, matches paper-style comparison) ----------------
    ax = axes[0]
    for name, color, label, ls in [
        ("ppo_low", "#1f77b4", "PPO", "-"),
        ("gppo_low", "#2ca02c", "GPPO", "-"),
    ]:
        r = np.array(R[name]["episode_rewards"], dtype=float)
        eps = np.arange(1, len(r) + 1)
        ax.plot(eps, r, color=color, alpha=0.18, lw=0.8)
        sm, sm_eps = smooth(r, w=15)
        ax.plot(sm_eps, sm, color=color, lw=2.2, label=label, linestyle=ls)

    # sample-efficiency style markers (paper-style dots + dashed guide lines)
    ppo_r, _ = smooth(R["ppo_low"]["episode_rewards"], w=15)
    gppo_r, _ = smooth(R["gppo_low"]["episode_rewards"], w=15)
    target = np.max(ppo_r)  # best smoothed PPO ever reaches
    gppo_ep = int(np.argmax(gppo_r >= target)) + 15 if np.any(gppo_r >= target) else None
    ppo_ep = int(np.argmax(ppo_r >= target)) + 15
    if gppo_ep:
        ax.axvline(gppo_ep, color="#2ca02c", ls=":", alpha=0.6)
        ax.axvline(ppo_ep, color="#1f77b4", ls=":", alpha=0.6)
        ax.axhline(target, color="gray", ls=":", alpha=0.4)
        ax.plot([gppo_ep], [target], "o", color="#2ca02c", ms=8, zorder=5)
        ax.plot([ppo_ep], [target], "o", color="#1f77b4", ms=8, zorder=5)
        reduction = 100 * (ppo_ep - gppo_ep) / ppo_ep
        ax.text(0.02, 0.04, f"GPPO ep{gppo_ep} vs PPO ep{ppo_ep}  ({reduction:+.0f}% episode difference)",
                transform=ax.transAxes, ha="left", fontsize=9,
                bbox=dict(boxstyle="round", fc="wheat", alpha=0.85))

    ax.set_xlabel("Training episodes [-]")
    ax.set_ylabel("Reward [-]")
    ax.set_title("Re=45 (regular flow) — same format as\nthe HydroGym paper's panel, 500 episodes")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim(0, 500)

    # ---------------- Panel B: Re=300 (high, our extension — the real difference) ----------------
    ax = axes[1]
    max_nan_ep = 0
    for name, color, label, marker in [
        ("ppo_high", "#d62728", "PPO", "o"),
        ("gppo_high", "#ff7f0e", "GPPO", "s"),
    ]:
        r = np.array(R[name]["episode_rewards"], dtype=float)
        eps = np.arange(1, len(r) + 1)
        ax.plot(eps, r, color=color, alpha=0.7, lw=1.3, marker=marker, ms=3, label=label)
        nan_mask = np.isnan(r)
        if nan_mask.any():
            first_nan = int(np.argmax(nan_mask)) + 1
            max_nan_ep = max(max_nan_ep, first_nan)
            ax.axvline(first_nan, color=color, ls="--", lw=1.8, alpha=0.8)
            ax.annotate(f"ep{first_nan}: diverges to NaN", xy=(first_nan, ax.get_ylim()[0]),
                        xytext=(first_nan + 3, ax.get_ylim()[0] + 0.3 + (0.4 if color == "#ff7f0e" else 0)),
                        color=color, fontsize=9, ha="left")

    ax.set_xlabel("Training episodes [-]")
    ax.set_ylabel("Reward [-]")
    ax.set_title("Re=300 (chaotic flow, near the stability limit) —\nregime NOT tested in the original paper, our contribution")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim(0, max(max_nan_ep + 20, 30))

    plt.tight_layout()
    plt.savefig("paper_comparison_500ep.png", dpi=150)
    print("Saved paper_comparison_500ep.png")

    # ---- print the honest numeric comparison for the write-up ----
    ppo_high_nan_ep = int(np.argmax(np.isnan(R["ppo_high"]["episode_rewards"]))) + 1
    gppo_high_nan_ep = int(np.argmax(np.isnan(R["gppo_high"]["episode_rewards"]))) + 1
    print(f"\nRe=300: PPO collapses at episode {ppo_high_nan_ep}, "
          f"GPPO collapses at episode {gppo_high_nan_ep} "
          f"({gppo_high_nan_ep/ppo_high_nan_ep:.1f}x later)")

    if gppo_ep:
        print(f"Re=45: GPPO reaches PPO's best smoothed reward ({target:.3f}) at ep{gppo_ep}, "
              f"PPO itself reaches it at ep{ppo_ep} "
              f"({100*(ppo_ep-gppo_ep)/ppo_ep:+.1f}% episode difference)")


if __name__ == "__main__":
    main()
