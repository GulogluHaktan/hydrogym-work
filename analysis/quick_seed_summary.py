import json, math, sys, glob
import numpy as np

seeds = sys.argv[1:] or ["0", "1", "2"]

def load(seed, cfg):
    for fn in [f"results/results_500_seed{seed}_{cfg}.json",
               f"results/results_500_{cfg}.json" if seed == "0" else "___"]:
        if glob.glob(fn):
            d = json.load(open(fn))
            if "episode_rewards" in d:
                return d
            if cfg in d and isinstance(d[cfg], dict):
                return d[cfg]
            if len(d) == 1:
                v = next(iter(d.values()))
                if isinstance(v, dict) and "episode_rewards" in v:
                    return v
    return None

def clean(rs):
    return np.array([np.nan if (x is None or (isinstance(x, float) and math.isnan(x))) else x
                     for x in rs], float)

def first_nan(rs):
    a = clean(rs)
    idx = np.where(np.isnan(a))[0]
    return int(idx[0] + 1) if len(idx) else None

def smooth(a, w=20):
    fill = np.nanmin(a)
    return np.convolve(np.where(np.isnan(a), fill, a), np.ones(w) / w, "valid")

def ep_to_reach_ppo_best(ppo_rs, other_rs, w=20):
    pa, oa = clean(ppo_rs), clean(other_rs)
    if len(pa) < w or len(oa) < w:
        return None, None, None
    ps, os_ = smooth(pa, w), smooth(oa, w)
    target = np.nanmax(ps)
    ppo_ep = int(np.nanargmax(ps) + w)
    hit = np.where(os_ >= target)[0]
    other_ep = int(hit[0] + w) if len(hit) else None
    return other_ep, ppo_ep, target

for seed in seeds:
    print(f"\n================ seed={seed} ================")
    print("-- Re=45 (regular): sample efficiency --")
    ppo = load(seed, "ppo_low")
    for cfg in ["ppo_low", "gppo_low", "sac_low", "gsac_low"]:
        d = load(seed, cfg)
        if d is None:
            print(f"  {cfg:11s} (missing)"); continue
        rs = d["episode_rewards"]
        last5 = np.nanmean(clean(rs)[-5:])
        if cfg == "gppo_low" and ppo:
            oe, pe, tgt = ep_to_reach_ppo_best(ppo["episode_rewards"], rs)
            sav = f"{100*(1-oe/pe):.0f}%" if (oe and pe) else "n/a"
            print(f"  {cfg:11s} last5={last5:6.2f}   reaches PPO-best(ep{pe}) at ep {oe}  -> {sav} saving")
        else:
            print(f"  {cfg:11s} last5={last5:6.2f}")
    print("-- Re=300 (chaotic): first-NaN episode --")
    order = []
    for cfg in ["ppo_high", "sac_high", "gppo_high", "gsac_high"]:
        d = load(seed, cfg)
        if d is None:
            print(f"  {cfg:11s} (missing)"); continue
        fn = first_nan(d["episode_rewards"])
        r0 = clean(d["episode_rewards"])[:3]
        order.append((cfg.replace("_high", ""), fn))
        print(f"  {cfg:11s} first-NaN ep = {fn}    first3 rewards = {np.round(r0,2)}")
    ok = [o for o in order if o[1]]
    if ok:
        ranked = " < ".join(f"{n}({e})" for n, e in sorted(ok, key=lambda x: x[1]))
        print(f"  robustness ranking (later = better): {ranked}")
