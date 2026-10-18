"""
Analytic-gradient safeguards shared by real_rl.py (GPPO) and sac_rl.py (GSAC).

Package C of IYILESTIRME_ONERILERI.md: attack the *seed-variance of the
mid-training breaking point* of the analytic-gradient methods without
pretending to remove the NaN failure altogether. Four mechanisms, each
independently switchable, all OFF by default so a run with no flags is
bit-identical to the original seed=0..4 sweep:

  1. grad_clip           - global-L2-norm clip of the analytic gradient
                           before it is mixed in.
  2. nonfinite_fallback  - if the analytic gradient is non-finite this
                           episode, drop it and take the pure model-free
                           actor/policy step instead.
  3. analytic_warmup     - first N episodes: pure model-free (no analytic
                           injection) so the critic / value head settles
                           before the through-simulator gradient is trusted.
  4. adaptive_mix        - GI-PPO alpha-policy style: shrink the analytic
                           weight when its (post-clip) norm exceeds
                           `adaptive_target`, so a spiking gradient
                           contributes less, not more.

`process_analytic` is called once per episode (not inside a jitted hot
loop), so plain Python + jax.tree_util is fine.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import jax
import jax.numpy as jnp


BASE_WEIGHT = 0.5  # the original 50/50 analytic / model-free mix


@dataclass
class SafeguardConfig:
    grad_clip: float | None = None
    nonfinite_fallback: bool = False
    analytic_warmup: int = 0
    adaptive_mix: bool = False
    adaptive_target: float = 20.0

    @property
    def active(self) -> bool:
        return (
            self.grad_clip is not None
            or self.nonfinite_fallback
            or self.analytic_warmup > 0
            or self.adaptive_mix
        )

    def as_dict(self):
        return asdict(self)


def _global_norm(pytree) -> float:
    leaves = jax.tree_util.tree_leaves(pytree)
    return float(jnp.sqrt(sum(jnp.sum(x ** 2) for x in leaves)))


def _scale(pytree, factor: float):
    return jax.tree_util.tree_map(lambda x: x * factor, pytree)


def process_analytic(g_analytic, gnorm: float, episode: int, cfg: SafeguardConfig):
    """Return (g_analytic_out, w_analytic, w_modelfree, info_dict).

    w_analytic + w_modelfree always sum to 1.0. When the analytic term is
    dropped, w_analytic == 0.0 and the caller still takes a (pure model-free)
    step, so the per-episode update count is unchanged.
    """
    info = {"episode": episode, "gnorm": gnorm, "reason": "mix",
            "clip_scale": 1.0, "w_analytic": BASE_WEIGHT}

    finite = math.isfinite(gnorm)

    if cfg.analytic_warmup and episode < cfg.analytic_warmup:
        info["reason"] = "warmup"
        info["w_analytic"] = 0.0
        return g_analytic, 0.0, 1.0, info

    if cfg.nonfinite_fallback and not finite:
        info["reason"] = "nonfinite_fallback"
        info["w_analytic"] = 0.0
        return g_analytic, 0.0, 1.0, info

    clip_scale = 1.0
    if cfg.grad_clip is not None and finite and gnorm > cfg.grad_clip:
        clip_scale = cfg.grad_clip / (gnorm + 1e-12)
        g_analytic = _scale(g_analytic, clip_scale)
        info["reason"] = "clipped"
    info["clip_scale"] = clip_scale
    eff_norm = gnorm * clip_scale

    w_an = BASE_WEIGHT
    if cfg.adaptive_mix and finite and eff_norm > cfg.adaptive_target:
        w_an = BASE_WEIGHT * (cfg.adaptive_target / (eff_norm + 1e-12))
        info["reason"] = "adaptive" if info["reason"] == "mix" else info["reason"] + "+adaptive"

    w_mf = 1.0 - w_an
    info["w_analytic"] = w_an
    return g_analytic, w_an, w_mf, info
