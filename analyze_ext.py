#!/usr/bin/env python3
"""
HydroGym ek çalışmalar — sonuç toplayıcı (paketler A + B + C).

results_ext/*.json dosyalarını okur, her koşu için ilk-NaN epizodunu bulur,
paket başına tablo çıkarır, küçük bir kesin işaret-çevirme permütasyon testi
yapar ve EXT_SONUCLAR.md (Türkçe) yazar. matplotlib varsa paket A için
ilk-NaN vs Re grafiğini de üretir (fig_ext_re_sweep.pdf).

Kullanım:  python analyze_ext.py --dir results_ext --out EXT_SONUCLAR.md
"""
import argparse
import glob
import json
import math
import os
from itertools import product

# n=5 taban çizgisi (Re=300, seed 0..4) — IYILESTIRME_ONERILERI.md tablosu.
BASELINE = {
    "ppo":  [12, 35, 117, 14, 11],
    "sac":  [36, 6, 2, 40, 131],
    "gppo": [83, 40, 32, 6, 16],
    "gsac": [141, 63, 2, 45, 122],
}


def first_nan_episode(rewards, n_episodes):
    """1-tabanlı ilk bozulma epizodu; hiç bozulmadıysa n_episodes döndürür
    (sağdan sansürlü) ve ikinci değer olarak False (=bozulmadı) verir."""
    for i, r in enumerate(rewards):
        if r is None or not math.isfinite(r) or abs(r) > 1e6:
            return i + 1, True
    return (len(rewards) or n_episodes), False


def load_runs(d):
    runs = []
    for path in sorted(glob.glob(os.path.join(d, "*.json"))):
        base = os.path.basename(path)
        if base.endswith(".FAILED.json"):
            continue
        try:
            data = json.load(open(path))
        except Exception as e:
            print(f"  (atlandı, okunamadı: {base}: {e})")
            continue
        for cfg_name, res in data.items():
            rw = res.get("episode_rewards", [])
            n_ep = len(rw)
            fn, broke = first_nan_episode(rw, n_ep)
            runs.append({
                "file": base, "cfg": cfg_name,
                "algo": res.get("algo"), "Re": res.get("Re"),
                "n_ep": n_ep, "first_nan": fn, "broke": broke,
                "ext": res.get("ext_config", {}),
                "events": res.get("safeguard_events", []),
                "wall": res.get("wall_total"),
            })
    return runs


def _parse_seed(fname):
    import re
    m = re.search(r"seed(\d+)", fname)
    return int(m.group(1)) if m else None


def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def sign_flip_p(diffs):
    """İki-taraflı kesin işaret-çevirme permütasyon p (2^n relabelleme)."""
    n = len(diffs)
    if n == 0:
        return float("nan")
    obs = abs(mean(diffs))
    count = 0
    for signs in product([1, -1], repeat=n):
        m = abs(mean([s * d for s, d in zip(signs, diffs)]))
        if m >= obs - 1e-12:
            count += 1
    return count / (2 ** n)


def fmt_list(xs):
    return ", ".join(str(x) for x in xs)


def section_A(runs, lines):
    a = [r for r in runs if r["file"].startswith("A_")]
    lines.append("\n## Paket A — Reynolds taraması\n")
    if not a:
        lines.append("_Henüz sonuç yok._\n")
        return None
    res_grid = {}   # (algo, Re) -> {seed: first_nan}
    for r in a:
        seed = _parse_seed(r["file"])
        res_grid.setdefault((r["algo"], float(r["Re"])), {})[seed] = r["first_nan"]
    algos = ["ppo", "gppo", "sac", "gsac"]
    res_set = sorted({re for (_, re) in res_grid})
    lines.append("İlk-NaN epizodu (config tamamlandıysa = epizot sayısı, sansürlü).\n")
    lines.append("| Re | " + " | ".join(a.upper() for a in algos) + " |")
    lines.append("|" + "---|" * (len(algos) + 1))
    for re in res_set:
        cells = []
        for al in algos:
            g = res_grid.get((al, re), {})
            vals = [g[s] for s in sorted(g)]
            cells.append(f"{fmt_list(vals)} (md {median(vals):.0f})" if vals else "—")
        lines.append(f"| {re:.0f} | " + " | ".join(cells) + " |")
    lines.append("\nRe=300 taban çizgisi (n=5): "
                 + "; ".join(f"{k.upper()} md {median(v):.0f}" for k, v in BASELINE.items()))
    return res_grid


def section_B(runs, lines):
    b = [r for r in runs if r["file"].startswith("B_")]
    lines.append("\n## Paket B — güçlü PPO baseline (Re=300)\n")
    if not b:
        lines.append("_Henüz sonuç yok._\n")
        return
    by = {}
    for r in b:
        seed = _parse_seed(r["file"])
        kind = "ppo" if "_ppo_" in r["file"] else "gppo"
        by.setdefault(kind, {})[seed] = r["first_nan"]
        cfg = r["ext"]
        ep = cfg.get("ppo_epochs"); mb = cfg.get("ppo_minibatches")
    lines.append(f"PPO iç-döngü: epochs={ep}, minibatches={mb}.\n")
    lines.append("| Algoritma | güçlü PPO ilk-NaN (seed sırası) | md | ort | taban md | taban ort |")
    lines.append("|---|---|---|---|---|---|")
    for kind in ["ppo", "gppo"]:
        g = by.get(kind, {})
        vals = [g[s] for s in sorted(g)]
        if not vals:
            continue
        base = BASELINE[kind]
        lines.append(f"| {kind.upper()} | {fmt_list(vals)} | {median(vals):.0f} | "
                     f"{mean(vals):.1f} | {median(base):.0f} | {mean(base):.1f} |")
    # eşleşen seed'lerde güçlü-GPPO − güçlü-PPO
    common = sorted(set(by.get("ppo", {})) & set(by.get("gppo", {})))
    if common:
        diffs = [by["gppo"][s] - by["ppo"][s] for s in common]
        lines.append(f"\nGüçlü GPPO − güçlü PPO (eşleşen seed {fmt_list(common)}): "
                     f"{fmt_list(diffs)}, ort {mean(diffs):+.1f}, işaret-çevirme p = {sign_flip_p(diffs):.3f}")


def section_C(runs, lines):
    c = [r for r in runs if r["file"].startswith("C_")]
    lines.append("\n## Paket C — GSAC-taban korumaları (gsac_high, Re=300)\n")
    if not c:
        lines.append("_Henüz sonuç yok._\n")
        return
    variants = ["clip", "fallback", "warmup", "adaptive", "combo"]
    by = {}
    for r in c:
        seed = _parse_seed(r["file"])
        for v in variants:
            if f"_{v}_" in r["file"]:
                by.setdefault(v, {})[seed] = r
                break
    base = BASELINE["gsac"]
    lines.append(f"Taban GSAC (n=5): {fmt_list(base)}, md {median(base):.0f}, "
                 f"ort {mean(base):.1f}, std {pstdev(base):.1f}\n")
    lines.append("| Varyant | ilk-NaN (seed sırası) | md | ort | std | Δstd vs taban | işaret-çevirme p (varyant−taban) |")
    lines.append("|---|---|---|---|---|---|---|")
    for v in variants:
        g = by.get(v, {})
        seeds = sorted(g)
        vals = [g[s]["first_nan"] for s in seeds]
        if not vals:
            continue
        # eşleşen seed'lerde farkı al (taban seed sırası 0..4)
        diffs = [g[s]["first_nan"] - base[s] for s in seeds if s is not None and s < len(base)]
        p = sign_flip_p(diffs) if diffs else float("nan")
        lines.append(f"| {v} | {fmt_list(vals)} | {median(vals):.0f} | {mean(vals):.1f} | "
                     f"{pstdev(vals):.1f} | {pstdev(vals) - pstdev(base):+.1f} | {p:.3f} |")
    # koruma olayları özeti
    lines.append("\n### Koruma tetiklenme sayıları (varyant başına, seed toplamı)")
    for v in variants:
        g = by.get(v, {})
        if not g:
            continue
        from collections import Counter
        cnt = Counter()
        for r in g.values():
            for ev in r["events"]:
                cnt[ev.get("reason", "?")] += 1
        if cnt:
            lines.append(f"- **{v}**: " + ", ".join(f"{k}×{n}" for k, n in cnt.most_common()))


def pstdev(xs):
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def make_fig_A(res_grid, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    algos = ["ppo", "gppo", "sac", "gsac"]
    res_set = sorted({re for (_, re) in res_grid})
    all_re = res_set + [300.0]
    fig, ax = plt.subplots(figsize=(6, 4))
    for al in algos:
        ys, es = [], []
        for re in all_re:
            if re == 300.0:
                vals = BASELINE[al]
            else:
                g = res_grid.get((al, re), {})
                vals = [g[s] for s in sorted(g)]
            if vals:
                ys.append(median(vals)); es.append((min(vals), max(vals)))
            else:
                ys.append(float("nan")); es.append((float("nan"), float("nan")))
        xs = all_re
        ax.plot(xs, ys, "o-", label=al.upper())
    ax.set_xlabel("Re"); ax.set_ylabel("ilk-NaN epizodu (medyan)")
    ax.set_title("Paket A — kırılma epizodu vs Reynolds")
    ax.legend(); fig.tight_layout()
    path = os.path.join(out_dir, "fig_ext_re_sweep.pdf")
    fig.savefig(path); plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results_ext")
    ap.add_argument("--out", default="EXT_SONUCLAR.md")
    args = ap.parse_args()

    runs = load_runs(args.dir)
    lines = ["# HydroGym ek çalışmalar — sonuçlar (paketler A + B + C)\n",
             f"_{len(runs)} koşu · otomatik üretildi: analyze_ext.py_\n"]
    fails = glob.glob(os.path.join(args.dir, "*.FAILED.json"))
    if fails:
        lines.append(f"\n> ⚠ {len(fails)} koşu HATA verdi: "
                     + ", ".join(os.path.basename(f) for f in fails) + "\n")

    res_grid = section_A(runs, lines)
    section_B(runs, lines)
    section_C(runs, lines)

    if res_grid:
        fig = make_fig_A(res_grid, args.dir)
        if fig:
            lines.append(f"\n![Re taraması]({os.path.relpath(fig, os.path.dirname(os.path.abspath(args.out)) or '.')})\n")

    lines.append("\n---\n### Ham koşu listesi\n")
    lines.append("| dosya | config | ilk-NaN | bozuldu? | duvar (s) |")
    lines.append("|---|---|---|---|---|")
    for r in sorted(runs, key=lambda x: x["file"]):
        w = f"{r['wall']:.0f}" if r["wall"] else "—"
        lines.append(f"| {r['file']} | {r['cfg']} | {r['first_nan']} | "
                     f"{'evet' if r['broke'] else 'hayır'} | {w} |")

    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"yazıldı: {args.out}  ({len(runs)} koşu)")


if __name__ == "__main__":
    main()
