import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ppo = json.load(open("results_real.json"))
    sac = json.load(open("results_sac.json"))
    R = {**ppo, **sac}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    styles = {
        "ppo_low": dict(color="tab:blue", ls="-", marker="o", label="PPO, Re=45"),
        "gppo_low": dict(color="tab:blue", ls="--", marker="s", label="GPPO, Re=45"),
        "sac_low": dict(color="tab:green", ls="-", marker="o", label="SAC, Re=45"),
        "gsac_low": dict(color="tab:green", ls="--", marker="s", label="GSAC, Re=45"),
    }
    for name, style in styles.items():
        r = np.array(R[name]["episode_rewards"], dtype=float)
        ax.plot(np.arange(1, len(r) + 1), r, ms=3, lw=1.2, **style)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode reward")
    ax.set_title("Re=45 (low, düzenli akış)")
    ax.legend(fontsize=8)

    ax = axes[1]
    styles = {
        "ppo_high": dict(color="tab:red", ls="-", marker="o", label="PPO, Re=300"),
        "gppo_high": dict(color="tab:red", ls="--", marker="s", label="GPPO, Re=300"),
        "sac_high": dict(color="tab:orange", ls="-", marker="o", label="SAC, Re=300"),
        "gsac_high": dict(color="tab:orange", ls="--", marker="s", label="GSAC, Re=300"),
    }
    for name, style in styles.items():
        r = np.array(R[name]["episode_rewards"], dtype=float)
        ax.plot(np.arange(1, len(r) + 1), r, ms=3, lw=1.2, **style)
        if np.isnan(r).any():
            first_nan = np.argmax(np.isnan(r)) + 1
            ax.axvline(first_nan, color=style["color"], alpha=0.3, lw=1)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode reward")
    ax.set_title("Re=300 (high, kaotik akış)\nPPO'nun dikey çizgisi = NaN'a ilk düştüğü episode")
    ax.legend(fontsize=8)

    ax = axes[2]
    names = ["gppo_low", "gppo_high", "gsac_low", "gsac_high"]
    labels = ["GPPO\nRe=45", "GPPO\nRe=300", "GSAC\nRe=45", "GSAC\nRe=300"]
    colors = ["tab:blue", "tab:red", "tab:green", "tab:orange"]
    cvs, means, stds = [], [], []
    for name in names:
        g = np.array(R[name]["grad_norms"])
        cvs.append(g.std() / g.mean())
        means.append(g.mean())
        stds.append(g.std())
    bars = ax.bar(labels, cvs, color=colors, alpha=0.85)
    for b, cv in zip(bars, cvs):
        ax.text(b.get_x() + b.get_width() / 2, cv + 0.005, f"{cv:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("Gradyan normu değişim katsayısı (CV = std/mean)")
    ax.set_title("Analitik gradyan kararsızlığı:\nGPPO vs GSAC, düşük vs yüksek Re")

    plt.tight_layout()
    plt.savefig("combined_results.png", dpi=140)
    print("Saved combined_results.png")

    print("\nCV summary:")
    for name, cv in zip(names, cvs):
        print(f"  {name}: CV={cv:.4f}")
    gppo_increase = (cvs[1] - cvs[0]) / cvs[0] * 100
    gsac_increase = (cvs[3] - cvs[2]) / cvs[2] * 100
    print(f"\nGPPO CV increase low->high Re: {gppo_increase:+.1f}%")
    print(f"GSAC CV increase low->high Re: {gsac_increase:+.1f}%")


if __name__ == "__main__":
    main()
