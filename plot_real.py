import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    with open("results_real.json") as f:
        results = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    styles = {
        "ppo_low": dict(color="tab:blue", ls="-", label="PPO, Re=45 (low)"),
        "ppo_high": dict(color="tab:red", ls="-", label="PPO, Re=300 (high)"),
        "gppo_low": dict(color="tab:blue", ls="--", label="GPPO, Re=45 (low)"),
        "gppo_high": dict(color="tab:red", ls="--", label="GPPO, Re=300 (high)"),
    }
    for name, style in styles.items():
        r = np.array(results[name]["episode_rewards"], dtype=float)
        eps = np.arange(1, len(r) + 1)
        ax.plot(eps, r, **style, marker="o", ms=3)
        nan_mask = np.isnan(r)
        if nan_mask.any():
            first_nan = np.argmax(nan_mask) + 1
            ax.axvline(first_nan, color=style["color"], alpha=0.3, lw=1)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode reward")
    ax.set_title("REAL hydrogym.jax KolmogorovFlow (64x64)\nPPO vs GPPO, n=1 seed")
    ax.legend(fontsize=8)
    ax.text(0.02, 0.02,
            "PPO Re=300 diverges to NaN\n(vertical line = first NaN episode);\nGPPO Re=300 stays stable.",
            transform=ax.transAxes, fontsize=8, va="bottom",
            bbox=dict(boxstyle="round", fc="wheat", alpha=0.7))

    ax = axes[1]
    for name, color, label in [("gppo_low", "tab:blue", "GPPO grad norm, Re=45 (low)"),
                                ("gppo_high", "tab:red", "GPPO grad norm, Re=300 (high)")]:
        gn = results[name]["grad_norms"]
        eps = np.arange(1, len(gn) + 1)
        ax.plot(eps, gn, color=color, label=label, marker="o", ms=3, lw=1)
    ax.set_xlabel("Episode")
    ax.set_ylabel("||analytic gradient||_2")
    ax.set_title("Analytic gradient L2 norm during training\n(real solver, confounded by policy change)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("real_results.png", dpi=140)
    print("Saved real_results.png")

    for name in ["gppo_low", "gppo_high"]:
        g = np.array(results[name]["grad_norms"])
        print(f"{name}: grad_norm mean={g.mean():.4g} std={g.std():.4g} cv={g.std()/g.mean():.3f}")

    r_high = np.array(results["ppo_high"]["episode_rewards"], dtype=float)
    n_nan = np.isnan(r_high).sum()
    print(f"ppo_high: {n_nan}/{len(r_high)} episodes NaN "
          f"(first at ep {np.argmax(np.isnan(r_high))+1})")


if __name__ == "__main__":
    main()
