"""n=5 significance analysis of Re=300 first-NaN episode (breaking point).
Paired sign-flip permutation test (exact, 2^5=32) + bootstrap CI + median table."""
import json, math, itertools, glob
import numpy as np

SEEDS = ["0", "1", "2", "3", "4"]
ALGOS = ["ppo", "sac", "gppo", "gsac"]

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

def first_nan(rs):
    for i, x in enumerate(rs):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return i + 1
    return None

bp = {a: [] for a in ALGOS}
for s in SEEDS:
    for a in ALGOS:
        d = load(s, f"{a}_high")
        bp[a].append(first_nan(d["episode_rewards"]))

print("=== Re=300 first-NaN episode (breaking point), n=5 ===")
print(f"{'algo':6s} " + " ".join(f"s{s}" for s in SEEDS) + "   median  mean   std")
for a in ALGOS:
    v = np.array(bp[a], float)
    print(f"{a:6s} " + " ".join(f"{x:3.0f}" for x in v) +
          f"   {np.median(v):5.0f}  {v.mean():5.1f}  {v.std(ddof=1):5.1f}")

def signflip_p(diffs):
    diffs = np.asarray(diffs, float)
    obs = abs(diffs.mean())
    n = len(diffs)
    cnt = 0
    for signs in itertools.product([1, -1], repeat=n):
        if abs((diffs * np.array(signs)).mean()) >= obs - 1e-9:
            cnt += 1
    return cnt / 2 ** n

def boot_ci(diffs, reps=20000, seed=0):
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs, float)
    n = len(diffs)
    meds = [np.median(diffs[rng.integers(0, n, n)]) for _ in range(reps)]
    return np.percentile(meds, [2.5, 97.5])

print("\n=== paired sign-flip permutation test (exact, 32 relabelings) ===")
pairs = [("gppo", "ppo"), ("gsac", "sac"), ("gsac", "gppo"),
         ("gsac", "ppo"), ("sac", "ppo"), ("gsac", "sac_and_ppo_pooled")]
for hi, lo in [("gppo", "ppo"), ("gsac", "sac"), ("gsac", "gppo"),
               ("gsac", "ppo"), ("sac", "ppo"), ("gppo", "sac")]:
    d = np.array(bp[hi], float) - np.array(bp[lo], float)
    p = signflip_p(d)
    lo_ci, hi_ci = boot_ci(d)
    star = "  *" if p < 0.05 else ""
    print(f"{hi:5s} - {lo:5s}: diffs={[int(x) for x in d]}  mean={d.mean():+6.1f}  "
          f"median={np.median(d):+5.1f}  95%CI[{lo_ci:+.0f},{hi_ci:+.0f}]  p={p:.3f}{star}")

print("\n=== most-robust (latest break) per seed ===")
for i, s in enumerate(SEEDS):
    order = sorted(ALGOS, key=lambda a: bp[a][i], reverse=True)
    print(f"  s{s}: {order[0]}({bp[order[0]][i]})  >  " +
          "  ".join(f"{a}({bp[a][i]})" for a in order[1:]))
from collections import Counter
c = Counter(sorted(ALGOS, key=lambda a: bp[a][i], reverse=True)[0] for i in range(len(SEEDS)))
print("  most-robust count:", dict(c))

print("\n=== Kruskal-Wallis (any algo differs?) ===")
try:
    from scipy.stats import kruskal, friedmanchisquare
    h, p = kruskal(*[bp[a] for a in ALGOS])
    print(f"  Kruskal-Wallis H={h:.2f}  p={p:.3f}")
    fr, pf = friedmanchisquare(*[[bp[a][i] for a in ALGOS] for i in range(len(SEEDS))])
    print(f"  Friedman (blocked by seed) chi2={fr:.2f}  p={pf:.3f}")
except Exception as e:
    print("  scipy unavailable:", e)
