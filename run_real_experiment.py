import argparse
import json
import time

import real_rl
from safeguards import SafeguardConfig

RE_LOW = 45.0
RE_HIGH = 300.0  # library's own solver becomes numerically unstable above ~Re 350-380
                  # at this grid/dt (verified empirically); 300 is the highest safe
                  # value we found that still meaningfully contrasts with Re=45.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--short-horizon", type=int, default=6)
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--out", type=str, default="results_real.json")
    ap.add_argument("--only", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    # Reynolds override (IYILESTIRME_ONERILERI.md package A: Re sweep).
    ap.add_argument("--re", type=float, default=None,
                    help="override the Re of the selected config (package A)")
    # Strong-PPO inner loop (package B).
    ap.add_argument("--ppo-epochs", type=int, default=1)
    ap.add_argument("--ppo-minibatches", type=int, default=1)
    # Analytic-gradient safeguards, GPPO only (package C).
    ap.add_argument("--grad-clip", type=float, default=None)
    ap.add_argument("--nonfinite-fallback", action="store_true")
    ap.add_argument("--analytic-warmup", type=int, default=0)
    ap.add_argument("--adaptive-mix", action="store_true")
    ap.add_argument("--adaptive-target", type=float, default=20.0)
    args = ap.parse_args()

    configs = {
        "ppo_low": dict(algo="ppo", Re=RE_LOW),
        "ppo_high": dict(algo="ppo", Re=RE_HIGH),
        "gppo_low": dict(algo="gppo", Re=RE_LOW),
        "gppo_high": dict(algo="gppo", Re=RE_HIGH),
    }
    if args.only:
        configs = {args.only: configs[args.only]}
    if args.re is not None:
        for cfg in configs.values():
            cfg["Re"] = args.re

    sg = SafeguardConfig(
        grad_clip=args.grad_clip,
        nonfinite_fallback=args.nonfinite_fallback,
        analytic_warmup=args.analytic_warmup,
        adaptive_mix=args.adaptive_mix,
        adaptive_target=args.adaptive_target,
    )

    results = {}
    for name, cfg in configs.items():
        print(f"\n=== Running {name} (Re={cfg['Re']}, seed={args.seed}) ===", flush=True)
        t0 = time.time()
        res = real_rl.train(
            algo=cfg["algo"], Re=cfg["Re"],
            n_episodes=args.episodes, n_steps=args.steps,
            short_horizon=args.short_horizon, dt=args.dt, action_time=1.0,
            seed=args.seed, log_every=5,
            ppo_epochs=args.ppo_epochs, ppo_minibatches=args.ppo_minibatches,
            sg=sg,
        )
        res["wall_total"] = time.time() - t0
        results[name] = res
        print(f"=== {name} done in {res['wall_total']:.1f}s ===", flush=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nSaved results to {args.out}")


if __name__ == "__main__":
    main()
