import json
import time
import argparse
import os

import sac_rl
from safeguards import SafeguardConfig

RE_LOW = 45.0
RE_HIGH = 300.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--short-horizon", type=int, default=6)
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--out", type=str, default="results/results_sac.json")
    ap.add_argument("--only", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    # Reynolds override (IYILESTIRME_ONERILERI.md package A: Re sweep).
    ap.add_argument("--re", type=float, default=None,
                    help="override the Re of the selected config (package A)")
    # Analytic-gradient safeguards, GSAC only (package C).
    ap.add_argument("--grad-clip", type=float, default=None)
    ap.add_argument("--nonfinite-fallback", action="store_true")
    ap.add_argument("--analytic-warmup", type=int, default=0)
    ap.add_argument("--adaptive-mix", action="store_true")
    ap.add_argument("--adaptive-target", type=float, default=20.0)
    args = ap.parse_args()

    configs = {
        "sac_low": dict(algo="sac", Re=RE_LOW),
        "sac_high": dict(algo="sac", Re=RE_HIGH),
        "gsac_low": dict(algo="gsac", Re=RE_LOW),
        "gsac_high": dict(algo="gsac", Re=RE_HIGH),
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
        res = sac_rl.train(
            algo=cfg["algo"], Re=cfg["Re"],
            n_episodes=args.episodes, n_steps=args.steps,
            short_horizon=args.short_horizon, dt=args.dt, action_time=1.0,
            seed=args.seed, log_every=5, sg=sg,
        )
        res["wall_total"] = time.time() - t0
        results[name] = res
        print(f"=== {name} done in {res['wall_total']:.1f}s ===", flush=True)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nSaved results to {args.out}")


if __name__ == "__main__":
    main()
