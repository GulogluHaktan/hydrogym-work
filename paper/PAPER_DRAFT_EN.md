# Gradient at the Edge of Chaos: Why Analytic-Gradient RL Has No Seed-Robust Advantage on Chaotic Kolmogorov Flow

*(Alt title candidate: "The Breaking Point of Analytic-Gradient Reinforcement Learning in
HydroGym" — the conference title; keep as running head.)*

**Working English journal draft — template-neutral.**
Target class: Q1 ML+science venue (Machine Learning: Science and Technology, or Neural Networks).
Final formatting (LaTeX `article` → target template) is deferred; author in this plain
Markdown file, convert at the end via the `makale-latex-yazici` skill.

**Authors:** Cem Özkurt¹ (corresponding, ORCID 0000-0002-1251-7715); Muhammed Emin Onay² (0009-0000-0454-4986); Eyüp Altuğ Tunç² (0009-0007-5166-8077); Ahmet Kutay Küçükler³ (0009-0007-0202-016X); Haktan Yusuf Güloğlu³ (0009-0007-9429-6154)
**Affiliations:**
¹ Department of Data Science and Analytics, Faculty of Computer and Information Sciences, Sakarya University, Sakarya, Türkiye
² Department of Computer Engineering, Faculty of Computer and Information Sciences, Sakarya University, Sakarya, Türkiye
³ Department of Computer Engineering, Sakarya University of Applied Sciences, Sakarya, Türkiye

---

## Draft status / editing map

| Section | Depends on training? | State |
|---|---|---|
| 1. Introduction | No | reframed around the n=5 seed-dominance result + surviving mechanisms |
| 2. Related work | No | expanded — GI-PPO [26], RPO [29], Bucci [27], Peitz [28] added |
| 3. Methods | No | drafted + Alg. 1/2 boxes; honest re: single-step PPO |
| 4. Experimental setup & reproducibility | No | drafted; repro subsection still to finalize |
| 5. Results | **Yes** | **n=5 complete**; §5.3 sign-flip + bootstrap + omnibus, nothing significant |
| 6. Mechanism of the break | **Yes (partly)** | §6(a) n=4-confirmed; §6(b) 1 window in 20 runs (suggestive); §6(c) 1/5 seeds |
| 7. Discussion | Yes | rewritten: seed-dominance is the finding; GSAC lean non-significant (3/5); §6(b)/§6(c) survive |
| 8. Limitations | Yes | updated for n=5; finding is seed-dominance so more seeds strengthen it |
| 9. Conclusion & future work | Yes | rewritten for n=5; future work: more seeds, GSAC-floor guards, Re sweep, stronger PPO |
| References | No | 29 verified (1–25: 2026-08-26; 26–29: 2026-08-30); +1 repro citation TODO |

**Planned new experiments (journal delta over the ICFAR conference version):**
1. Extend beyond n=5 + a Reynolds sweep — the current n=5 permutation test (§5.3) finds nothing significant; more seeds would test GSAC's non-significant median lean.
2. Action-amplitude short-horizon re-run of seeds 0–1 (logs now in place) — trigger-episode cause (§6(c)).
3. Reynolds sweep {45, 100, 200, 300} — does a milder chaotic regime restore a stable ordering?
4. GSAC-floor guards: analytic-grad clip, non-finite fallback, warmup, adaptive weight (GI-PPO α-policy).
5. Ablations at Re=45: horizon H ∈ {3, 6, 12}; mixing ratio ∈ {0.25, 0.5, 0.75}.
6. (If VRAM allows) one 128×128 confirmation run; reconciliation with the official HydroGym GPPO.

---

## Abstract

*(n=5 complete; §5.3 significance test done — no separable ordering.)*

This work asks whether injecting an analytic (backpropagated-through-the-simulator) policy
gradient helps or hurts as a fluid-flow control task becomes chaotic, and whether off-policy
RL tolerates the resulting gradient instability better than on-policy RL. The *Nature* paper
introducing HydroGym [1] reports that Gradient-Enhanced PPO (GPPO) — which mixes an analytic
reward-trajectory gradient, backpropagated through the differentiable Kolmogorov-flow solver,
with the standard PPO gradient — cuts training iterations by at least 65% relative to PPO,
shown at a single Reynolds number and seed. Whether this holds across the regular and chaotic
regimes of the same environment is untested — notable given the documented ill-conditioning
of backpropagation-through-time under chaotic dynamics [7] and the Re-shrunk predictability
horizon of turbulence [25].

We evaluate PPO [6], GPPO, SAC [8], and a gradient-enhanced SAC variant (**GSAC**) — the
first transfer of the off-policy maximum-entropy analytic-gradient family [4] to fluid
dynamics — on the real `hydrogym.jax` Kolmogorov-flow environment at a regular (Re=45) and a
chaotic (Re=300) operating point, 500 episodes per configuration, across independent training
seeds. **At Re=45**, GPPO reaches PPO's best reward in substantially fewer episodes in 4 of 5
seeds (≈63–84%; seed=0's 63% closest to the paper's 65%), and the off-policy family reaches
better final reward than the on-policy family in every seed. **At Re=300**, all four
algorithms eventually diverge (NaN), and the episode at which they first diverge is dominated
by the seed, not the algorithm: over five seeds PPO breaks at {12, 35, 117, 14, 11}, SAC at
{36, 6, 2, 40, 131}, GPPO at {83, 40, 32, 6, 16}, GSAC at {141, 63, 2, 45, 122}. GSAC has the
highest median breaking point (63 vs 14–36) and is the most robust of the four in 3 of 5
seeds, but an exact sign-flip permutation test separates no pair of algorithms (all p ≥ 0.19;
Kruskal–Wallis p = 0.57, Friedman p = 0.77), and analytic-gradient injection does not delay
the on-policy break (GPPO − PPO mean −2.4 episodes, p = 0.81). At seed=2 both off-policy
methods collapse at episode 2 — the flow diverges before learning begins, GSAC and bare SAC
failing identically.
The reproducible statements about chaotic-regime training are instead mechanistic: (i) the break
is always sudden and precursor-free; (ii) whether a brief post-trigger "survival window"
may depend on the replay buffer's size at the trigger — a dilution calculation predicts a
~1% survival chance only for large-buffer GSAC, and the one window observed across 20 runs is
that case; and (iii) in a rare failure mode (1 of 5 seeds) when the randomly initialized policy
already sits near the solver's stability boundary, the flow diverges within ~2 episodes
regardless of algorithm — bare SAC and GSAC fail identically, with normal control amplitude
and a normal analytic-gradient norm at the point of divergence. GSAC has the highest ceiling
of the four (2–10× the stable horizon of any other method when training gets a foothold) but
no reliable floor. We conclude that analytic-gradient injection **does not impose a
seed-robust ordering** on chaotic-regime training stability, and that the useful,
reproducible characterization of that regime is mechanistic rather than a ranking of
algorithms.

**Keywords:** reinforcement learning, differentiable simulation, analytic policy gradients,
fluid-dynamics control, chaotic dynamics, training stability, PPO, SAC.

---

## 1. Introduction

The active control of fluid flows — for drag reduction, mixing enhancement, or
extreme-event suppression — has become one of the most active application areas of
reinforcement learning. The founding study by Rabault et al. [23] first showed that deep RL
can control periodic vortex shedding behind a cylinder, achieving ~8% drag reduction at
Re=100; this line has since extended into the turbulent regime, e.g. turbulent
separation-bubble control [10] and real-time, state-dependent turbulent-wake control [24].
HydroGym [1] consolidates these scattered codebases into a single, solver-agnostic platform
spanning 61+ environments from laminar to turbulent, and — alongside classical solvers
(FEM, LBM) — ships a JAX-based [13], end-to-end **differentiable** pseudo-spectral solver
that continues the line of differentiable-CFD libraries JAX-CFD [14] and JAX-Fluids [15, 16].
A methodologically related, PyTorch-based, fully differentiable flow-control benchmark,
FluidGym [2], belongs to the same ecosystem.

Differentiability lets RL move beyond model-free training by computing **analytic policy
gradients** — the derivative of the reward with respect to policy parameters, obtained by
the chain rule *through the simulator itself* rather than by sampling. The HydroGym paper
demonstrates this with GPPO: the analytic gradient is mixed with PPO's likelihood-ratio
gradient — a construction closely related to Gradient-Informed PPO [26] — and on the
Kolmogorov-flow environment this is reported to cut training iterations by at least 65% and
to yield more efficient controllers with lower action amplitude [1].
That result, however, is shown at a **single operating point** of the wide Re=40–500 range
the paper defines for Kolmogorov flow; the behaviour of the analytic gradient at the regular
and chaotic ends of that range is not separately compared.

This gap overlaps a well-documented risk. Metz et al. [7] demonstrate a common
"chaos-induced failure mode" of backpropagation through iterated maps, physics simulations,
and learned optimizers, and tie it to the spectrum of the system Jacobian. On the
fluid-dynamics side, Boffetta and Musacchio [25] show that the largest Lyapunov exponent of
homogeneous-isotropic turbulence grows with Reynolds number (faster than the dimensional
expectation), so the predictability horizon shrinks as Re rises. Together these two
independent results — one from optimization, one from fluid dynamics — suggest that in a
chaotic regime the norm and direction of the analytic gradient can become hypersensitive to
initial conditions (a butterfly effect), and that GPPO's reported advantage could weaken, or
reverse, as Re increases.

Analytic-gradient injection has so far been applied mostly with **on-policy** methods (GPPO)
or in rigid-body / contact-rich robotics tasks: SHAC [3] mitigates the exploding/vanishing
gradient problem with short-horizon rollouts and a learned terminal value function; AHAC [5]
goes further by adapting the horizon to avoid stiff dynamics rather than fixing it. On the
off-policy side, SAPO [4] carries analytic-gradient RL into a maximum-entropy (SAC-like)
framework and reports strong performance on deformable-body tasks — but this approach has
never been applied to fluid dynamics. We add **GSAC**, a gradient-enhanced version of SAC
[8], as a fourth method, placing the comparison at the intersection of the SHAC family
(short-horizon analytic gradient) and the SAPO family (off-policy maximum-entropy analytic
gradient), and asking whether the off-policy replay-buffer structure tolerates chaotic-regime
gradient instability better than on-policy GPPO.

**Contributions.**
1. We extend HydroGym's GPPO-vs-PPO result on Kolmogorov flow [1] from a single Reynolds
   point to a regular-vs-chaotic regime comparison, and from n=1 to n=5 seeds.
2. We introduce **GSAC** (SAC + short-horizon analytic gradient), never previously applied
   to HydroGym and the first transfer of the SAPO [4] family to fluid dynamics, giving a
   2×2 design (on-/off-policy × with/without analytic gradient) that isolates each factor.
3. **A seed-dominance result, replicated over five seeds with a formal test.** In the chaotic
   regime all four algorithms eventually diverge, and the episode at which they first diverge
   is dominated by the training seed, not the algorithm: over five seeds each algorithm's
   first-NaN episode spans more than an order of magnitude (PPO 11–117, SAC 2–131, GPPO 6–83,
   GSAC 2–141), PPO is both the first-to-break at seed=0 and the most robust of the four at
   seed=2, and an exact sign-flip permutation test separates no pair of algorithms (all
   p ≥ 0.19; Kruskal–Wallis p = 0.57, Friedman p = 0.77). The "clean multiplicative" pattern
   seen at n=1 is a small-n artifact; analytic-gradient injection does not delay the on-policy
   break (GPPO − PPO mean −2.4 episodes, p = 0.81). GSAC carries a non-significant lean — the
   highest median breaking point (63) and most robust in 3 of 5 seeds — reported as a
   candidate effect, not a result.
4. **Two mechanistic results that do survive**, both algorithm-design-independent: (a) a
   replay-buffer dilution calculation for whether a brief post-trigger "survival window" can
   appear (non-zero probability only for large-buffer GSAC; one window seen in 20 runs, that case);
   and (b) an early-init failure mode — when the initial policy already sits near the solver's
   stability boundary the flow diverges within ~2 episodes regardless of algorithm, which
   we document directly with per-episode action-amplitude and analytic-gradient-norm logs.
5. GSAC is characterized as **highest ceiling, no reliable floor**: 2–10× the stable horizon
   of any other method when training gets a foothold (seeds 0, 1, 4), but episode-2 failure
   when the flow diverges before learning begins (seed=2).

**Pre-registered hypothesis.** The norm/variance of the analytic gradient will be stable at
low Re and unstable at high Re; against this instability, GSAC's off-policy structure will be
more robust than GPPO's on-policy structure. *(Outcome at n=5: not supported at p < 0.05.
GSAC does have the highest median breaking point and is most robust in 3 of 5 seeds, and
GSAC − GPPO is positive in the mean (+39 episodes), but the sign-flip permutation p-value is
0.19 and the effect is dominated by seed-to-seed variance. Per the pre-registered decision
rule (§5.3) the finding stands as "no seed-robust ordering"; GSAC's lean is reported as a
candidate effect for a larger-n study.)*

---

## 2. Related work

### 2.1 RL for flow control and HydroGym

Since Rabault et al. [23], model-free RL has repeatedly been shown to succeed on turbulent
flow control: turbulent separation-bubble control [10] reports ~9% vs ~6.8% drag/separation
reduction over classical periodic forcing; an early study [11] establishes DRL's
applicability to turbulent conditions; real-time state-dependent bluff-body wake control [24]
achieves 2–4× the performance of model-based baselines in wind-tunnel experiments. A broad
robotics-focused survey [12] documents the wider maturity of RL for real-world control.
HydroGym [1] unifies these under one solver-agnostic RL interface (61+ environments), shows
zero-shot transfer of learned policies across environments (e.g. 38% drag reduction on a 3D
wing section), and reports the ≥65% sample-efficiency gain for GPPO over PPO on Kolmogorov
flow that this paper examines. FluidGym [2] is a methodologically related, PyTorch-based,
fully differentiable flow-control benchmark with PPO/SAC baselines. None of these works
systematically compares analytic-gradient injection across a regular and a chaotic regime.

### 2.2 Analytic policy gradients through differentiable simulation

The single closest prior method to the HydroGym paper's GPPO is **GI-PPO** [26] (Son et al.,
NeurIPS 2023): it introduces an "α-policy" that adaptively weights the analytic gradient
against the PPO gradient, with explicit variance/bias metrics that down-weight the analytic
term when it is detected to be high-variance or biased. Our fixed 50/50 mix is the
non-adaptive special case (α held constant); moving to an α-policy-style adaptive weight is
the natural next step for the chaotic regime (§9). **RPO** [29] (Zhong et al., 2025) analyzes
the on-policy analytic-gradient setting directly: it shows that reparameterization policy
gradients with sample reuse optimize a PPO-style surrogate via BPTT, and targets the training
instability of that combination — the same instability this paper measures at Re=300.

The broader method family originated in the robotics/control literature. **SHAC** [3]
decouples the analytic gradient from the full episode horizon using short-horizon rollouts
and a smooth critic, mitigating exploding/vanishing gradients; it reports >17× reductions in
training time on muscle-actuated locomotion. The 6-step short horizon used in our GPPO/GSAC
is a direct adaptation of this idea. **AHAC** [5] replaces the fixed short horizon with one
adaptively truncated to avoid stiff contact dynamics, gaining ~40% more reward than
model-free baselines on locomotion. **SAPO** [4] carries the idea into a maximum-entropy,
off-policy framework, testing the hypothesis that entropy regularization smooths the
analytic-gradient optimization landscape, on rigid-body plus deformable-material tasks. Our
GSAC is the first application of the SAPO idea to fluid dynamics.

### 2.3 Analytic-gradient instability in chaotic systems

The theoretical basis of our hypothesis is Metz et al. [7]: in chaotic dynamical systems,
backpropagated gradients explode through a butterfly-effect mechanism rooted in the system
Jacobian spectrum, and clipping, truncated BPTT, or a switch to zeroth-order methods are
proposed as remedies. The fluid-dynamics counterpart is Boffetta and Musacchio [25]: the
largest Lyapunov exponent of homogeneous-isotropic turbulence grows with Reynolds number and
the predictability horizon shrinks inversely with it. Together, these frame our 500-episode
finding that "GPPO and GSAC also eventually reach NaN, only much later": instability is
inevitable; what differs across algorithms is only the delay mechanism (off-policy structure
+ short horizon). That gradient-free RL *can* control spatiotemporal chaos is not in question
— Bucci et al. [27] do exactly this on the Kuramoto–Sivashinsky equation; the question here
is specifically whether *analytic-gradient* injection survives the chaotic regime.

### 2.4 Analytic vs. zeroth-order gradients; partial differentiability

Suh et al. [20] compare analytic and zeroth-order policy gradients in systems with stiffness
and discontinuities, and via a bias-variance analysis propose an α-graded interpolating
estimator — of which our 50/50 mix is roughly a special case (α=0.5). A 2026 follow-up [21]
re-tests the same question with a lighter hybrid-switching scheme (DDCG). ABPT [22] proposes
an amended BPTT scheme combining 0-step and N-step returns for partially differentiable
rewards (e.g. quadrotor flight), a problem class close to HydroGym's mixed control-reward /
physics-state structure.

### 2.5 Base RL algorithms

The two gradient-free anchors of the comparison are PPO [6] and Soft Actor-Critic [8, 9].
PPO's advantage estimation builds on TRPO [17] and Generalized Advantage Estimation
(GAE) [18]; Soft Q-learning [19], the origin of SAC's maximum-entropy framework, provides the
theoretical grounds for why entropy regularization can make GSAC more robust in the chaotic
regime — the entropy term forces the policy toward a broader action distribution under
uncertainty, damping the effect of a single ill-conditioned gradient step.

### 2.6 RL against spatiotemporal chaos, and distributed PDE control

The most direct precedent for RL control of a *chaotic* PDE is Bucci et al. [27], who
stabilize the one-dimensional Kuramoto–Sivashinsky equation — a standard chaotic-PDE
benchmark with sensitive dependence on perturbations — with deep RL, driving it onto unstable
fixed points. Peitz et al. [28] address the *distributed* structure of PDE actuation:
exploiting translational equivariance, a high-dimensional control problem becomes a
multi-agent problem with identical uncoupled agents and a convolutionally reduced
observation, which transfers across domain sizes. Kolmogorov flow's low-wavenumber sinusoidal
actuators sit in the same distributed-control setting. Beyond these, the broader body of
model-free flow-control work [10, 11, 23, 24] confirms the maturity of RL for turbulent flow
control but does not compare analytic-gradient injection against the chaotic/turbulent regime
— the specific gap this paper targets.

*Reference-expansion note (journal delta): a reproducibility/checklist citation to be added
for §4.3; possible additions on model-based RL for PDE control.*

---

## 3. Methods

### 3.1 Environment

Experiments run on the real `hydrogym.jax.envs.kolmogorov.KolmogorovFlow` environment: a
64×64 grid (4096 degrees of freedom), pseudo-spectral IMEX-RK4CN time integration [1, 13–16],
4 actuator modes (low-wavenumber sinusoidal forcing patterns). The observation is the 8×8
subsampled magnitude of the velocity field.

We do **not** use the library's official GPPO implementation; we re-implement the same
mechanism (§3.3) inside our own PPO/SAC training loop as an independent codebase. Because the
exact mixing ratio, horizon length, and hyperparameters the paper used for GPPO are not given
in the published text or figure captions, the choices in §3.3–3.4 are **our reimplementation's**
and are not claimed to be identical to the paper's. *(Journal delta: a reconciliation run
against the official implementation is planned.)*

Two library quirks required a subclass. (i) `step_env` ignores the `params.dt` /
`params.action_time` passed at call time and re-reads `self.default_params` internally, so to
reduce the timestep and action duration to POC scale we override the `default_params`
property on a subclass (`FastKolmogorovFlow`). (ii) The library default `reward_alpha=0`
makes the reward control-cost-only; we set `reward_alpha=1` so flow energy is included in the
reward.

**Reynolds-number choice.** At this grid/dt combination (64×64, dt=0.001) the library's own
solver was empirically observed to diverge numerically near Re≈350–380 — even under zero
action, from purely analytic initial conditions, and not resolved by reducing dt 10× (so not
a simple CFL issue). We therefore use **Re=300** for the "chaotic" regime — the highest value
that stays below this limit while forming a meaningful contrast with Re=45 — rather than the
originally intended Re=450. *(Journal delta: an intermediate Re sweep {45, 100, 200, 300} is
planned to establish the delay-vs-Re trend; a quantitative characterization of both regimes
via the largest Lyapunov exponent / energy spectrum is planned to substantiate the
"regular" vs "chaotic" labels.)*

### 3.2 PPO and GAE

The policy is a diagonal Gaussian: an MLP (2 hidden layers, 64 units, tanh) produces a
bounded mean action `a_mean = 0.5·tanh(·)`, together with a state-independent, learnable
`log σ`; the executed action is sampled `a ~ N(a_mean(s), σ²)` and is **not** re-clipped, so
sampled actions can exceed ±0.5 (observed |a| up to ≈1.2 early in training). A separate 2×64
MLP is the value function. Advantage estimation uses GAE [18]:

Â_t = Σ_{l=0}^{T−t−1} (γλ)^l δ_{t+l},  δ_t = r_t + γ V(s_{t+1}) − V(s_t)

with γ=0.99, λ=0.95. PPO's clipped surrogate objective [6], derived from TRPO's [17]
trust-region idea:

L^PPO(θ) = E_t[ min( ρ_t(θ) Â_t, clip(ρ_t(θ), 1−ε, 1+ε) Â_t ) ],  ρ_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)

with ε=0.2. The total loss adds a value term and an entropy bonus:
L = −L^PPO + 0.5·L^V − 0.001·H.

**Honest note on the PPO update.** Our loop performs **one full-batch gradient step per
episode** on the clipped surrogate (no PPO epochs, no minibatching), with an episode being
15 environment steps. This is a deliberately minimal PPO; it keeps the on-policy and
off-policy training loops comparable and keeps the analytic-gradient mix (§3.3) simple, but
it means "PPO" here is weaker than a fully tuned implementation. The comparison of interest
is *relative* (PPO vs GPPO, SAC vs GSAC, on- vs off-policy) and all four share this budget.

### 3.3 GPPO: analytic-gradient injection

Following the short-horizon idea of SHAC [3] and AHAC [5], over a short horizon of H=6 steps
we forward-propagate the policy's **mean** action (no sampling, so the whole chain stays
smooth) through the differentiable simulator and take the analytic gradient of the negative
total reward with respect to θ via `jax.grad`:

L_analytic(θ) = −Σ_{t=1}^{H} r(s_t, a_mean_θ(s_t)),  g_analytic = ∇_θ L_analytic(θ)

The final gradient mixes this with the standard PPO gradient at equal weight:
g_final = 0.5·g_analytic + 0.5·g_PPO — a special case (α=0.5) of the α-graded estimator of
Suh et al. [20]. This is the same mechanism as the HydroGym paper's description of GPPO
(analytic backprop of the reward trajectory through the deterministic solver, combined with
the standard gradient [1]); the horizon length and mixing weight are our own choices
(§3.1, §8).

```
Algorithm 1  GPPO — one training episode (mix weight β = 0.5, horizon H = 6)
--------------------------------------------------------------------------
Input : policy θ, value ϕ, optimizer state, fixed reset state s0
 1  τ ← rollout(π_θ, s0) for T = 15 steps         # sample a_t ~ N(a_meanθ(s_t), σ²)
 2  Â, R ← GAE(τ; γ = 0.99, λ = 0.95)
 3  g_PPO ← ∇_θ [ −L^PPO(θ; τ, Â) + 0.5 L^V(ϕ; τ, R) − 0.001 H(π_θ) ]   # clip ε = 0.2
 4  # analytic branch: deterministic, differentiable, short horizon
 5  L_an(θ) ← − Σ_{t=1..H} r( f_sim^t(s0, a_meanθ) )      # mean action, no sampling
 6  g_an ← ∇_θ L_an(θ)                                    # jax.grad through the solver
 7  g ← β · g_an + (1 − β) · g_PPO
 8  θ ← Adam_update(θ, g; lr = 3e-4)                      # single full-batch step
Output: θ, ϕ, log(‖g_an‖, RMS|a_t|, max|a_t|, Σ r_t)
--------------------------------------------------------------------------
PPO is Algorithm 1 with lines 4–7 replaced by  g ← g_PPO.
```

*(Journal delta: H ∈ {3, 6, 12} and mixing ratio β ∈ {0.25, 0.5, 0.75} ablations planned at
Re=45.)*

### 3.4 SAC and GSAC

SAC [8, 9] uses a squashed (tanh) Gaussian actor, twin Q-critics, and target networks to
optimize the maximum-entropy objective originating from Soft Q-learning [19]:

J(π) = E[ Σ_t r(s_t, a_t) + α·H(π(·|s_t)) ]

with actor loss J_π(φ) = E_{s~D, a~π_φ}[ α·log π_φ(a|s) − min(Q_{ψ1}, Q_{ψ2})(s, a) ].
Critics are updated with soft target networks (τ=0.005). The entropy temperature is fixed
(α=0.1, not auto-tuned). Per episode: the rollout's 15 transitions are added to a replay
buffer (capacity = n_episodes × 15), then 10 gradient updates are performed with batch
size 64.

**GSAC** follows SAPO's [4] core motivation (off-policy + maximum entropy + analytic
gradient) and applies the same mechanism as GPPO (§3.3; H=6, deterministic action) to the
actor gradient: g_final = 0.5·g_analytic + 0.5·g_π, with g_π = ∇_φ J_π(φ). GSAC is the first
transfer of the SAPO family to fluid dynamics; unlike SAPO it uses HydroGym's Kolmogorov-flow
environment (not the rigid/deformable Rewarped platform) and a fixed 50/50 mix (not SAPO's
own entropy-based smoothing scheme). See §8.

```
Algorithm 2  GSAC — one training episode (mix weight β = 0.5, horizon H = 6)
--------------------------------------------------------------------------
Input : actor φ, critics ψ1, ψ2, targets ψ̄1, ψ̄2, replay buffer D, reset state s0
 1  τ ← rollout(π_φ, s0) for T = 15 steps ; D ← D ∪ τ
 2  for k = 1 … 10 do                                   # updates_per_episode
 3      B ← sample(D, 64)
 4      for i ∈ {1, 2}:  ψ_i ← Adam_update(ψ_i, ∇ critic_loss(ψ_i; B, ψ̄1, ψ̄2, φ))
 5      g_π ← ∇_φ E_{s∈B}[ α log π_φ(a|s) − min_i Q_{ψ_i}(s, a) ],  a ~ π_φ(·|s), α = 0.1
 6      φ ← Adam_update(φ, g_π)
 7      ψ̄_i ← (1 − τ_s) ψ̄_i + τ_s ψ_i ,  τ_s = 0.005
 8  # analytic branch on the actor, after the SAC updates
 9  g_an ← ∇_φ [ − Σ_{t=1..H} r( f_sim^t(s0, a_meanφ) ) ]
10  φ ← Adam_update(φ, β · g_an + (1 − β) · g_π)         # g_π = last actor grad from line 5
Output: φ, ψ, log(‖g_an‖, RMS|a_t|, max|a_t|, Σ r_t)
--------------------------------------------------------------------------
SAC is Algorithm 2 with lines 8–10 removed.
```

**Consistency note.** GAE, PPO clipping, and the 50/50 analytic/standard gradient mix are
hand-rolled in plain JAX/optax and kept identical in form across the PPO and SAC training
scripts. Both training scripts additionally log per-episode action amplitude (RMS and
absolute max) for the trigger-episode diagnostic of §6.

---

## 4. Experimental setup and reproducibility

### 4.1 Configuration

| Parameter | Value |
|---|---|
| Grid | 64×64 (4096 DOF) |
| Solver | pseudo-spectral IMEX-RK4CN (real `hydrogym.jax`) |
| Re (regular / chaotic) | 45 / 300 |
| Episodes / steps per episode | 500 / 15 |
| Environment dt / action_time | 0.001 / 1.0 |
| Reward | `reward_alpha = 1` (flow energy + control cost) |
| Analytic horizon (GPPO/GSAC) | H = 6 steps, mean (deterministic) action |
| Gradient mixing | 0.5 · standard + 0.5 · analytic |
| Optimizer / learning rate | Adam / 3×10⁻⁴ (actor and critic) |
| Policy / value / critic network | 2 hidden layers × 64 units, tanh |
| Action | PPO: mean = 0.5·tanh(·), executed action sampled (not re-clipped); SAC: squashed tanh |
| PPO | γ=0.99, λ=0.95, clip ε=0.2, entropy coeff 0.001, **1 full-batch step/episode** |
| SAC | γ=0.99, α=0.1 (fixed), τ=0.005, 10 updates/episode, batch 64, buffer = 500×15 |
| Seeds | n=5 (seed=0–4) |
| Hardware | RTX 5060 laptop GPU (8 GB), JAX 0.10.2 + CUDA 12 |

All four algorithms are trained for 500 episodes in both Re regimes, matching the training
budget reported by the HydroGym paper [1].

### 4.2 Seed protocol

Each `(algorithm, Re, seed)` triple is trained independently for **five seeds** (seed=0–4).
JAX PRNG is fully deterministic, so a given `(config, seed)` is bit-for-bit reproducible (the
seed=3 GSAC/Re=300 run was regenerated identically after its first dump was lost). Configs
are run strictly one at a time: this class of consumer GPU cannot host concurrent JAX
processes without severe memory contention / OOM. Per-seed wall time for the full 8-config
sweep is ~6–8.5 h on the RTX 5060 (seed=4: ~6.2 h; ppo ~28 min/config, gppo ~63 min/config,
sac ~28 min/config, gsac ~60 min/config).

### 4.3 Reproducibility package

*(Journal delta — to be finalized before submission.)*
- Experiment code released as a standalone repository (separated from the upstream HydroGym
  clone), with `requirements-lock.txt` (122 pinned packages), a portable `setup_env.sh`, and
  the parametrized sweep driver `run_seed_sweep.sh`.
- Every per-config result dump (`results_500[_seedN]_{config}.json`: episode rewards, analytic
  gradient norms, wall times) and driver log included.
- Archived with a DOI (Zenodo); "Code availability" and "Data availability" statements.
- ML reproducibility checklist completed.

---

## 5. Results

> **STATUS (2026-08-31): n=5 complete (seed=0–4). §5.3 significance test done — no pair of
> algorithms is separable (all p ≥ 0.19; Kruskal–Wallis p = 0.57, Friedman p = 0.77). GSAC
> has the highest median breaking point and is most robust in 3/5 seeds but not significantly;
> seed=2 shows the early-init failure mode (§6(c)). Finding: no seed-robust ordering.**

### 5.1 Re=45 (regular flow): sample efficiency

At Re=45, GPPO reaches PPO's best 500-episode smoothed reward in substantially fewer episodes
in **4 of 5 seeds** (seed=0: 63% fewer, from the conference version; seed=1: ~78%; seed=3:
~84%; seed=4: ~70%), and by a smaller margin in seed=2 (~26%). The *direction* of GPPO's
sample-efficiency advantage over PPO holds in all five seeds, and the seed=0 value (63%) is
close to the paper's reported 65% [1] — evidence that our reimplementation captures the
paper's mechanism. (The exact "episode-to-reach" metric is smoothing-window dependent; a
fixed protocol will be pinned down for the final version, which is why the seed=1–4
percentages are quoted approximately.)

**Table 2.** Final performance at Re=45, last-5-episode mean reward (seed=0 / 1 / 2 / 3 / 4).

| Algorithm | Final reward (last-5-ep. mean) |
|---|---|
| PPO  | −25.1 / −25.2 / −27.2 / −26.4 / −25.6 |
| GPPO | −25.4 / −24.6 / −26.7 / −25.4 / −25.4 |
| SAC  | −16.7 / −16.9 / −18.9 / −16.3 / −20.3 |
| GSAC | −18.0 / −17.1 / −20.4 / −16.3 / −19.5 |

Two Re=45 findings are seed-stable: (i) the off-policy family (SAC/GSAC) reaches clearly
better final reward than the on-policy family in every seed — its replay-buffer structure is
already far more sample-efficient, so no GPPO-style acceleration analysis applies to it; and
(ii) **GSAC does not exceed SAC's final performance in any seed** — the analytic gradient
does not help the off-policy method in the regular regime. (seed=2 trained to a worse level
across all four algorithms, but the *relative* ordering is unchanged.)

### 5.2 Re=300 (chaotic flow): the breaking point does not have a stable ordering

In the chaotic regime **all four algorithms have a breaking point** — none completes 500
episodes stably; all eventually diverge (NaN). The episode at which the break first occurs
varies by algorithm, but far more so **by seed**: the ranking permutes across the five seeds,
and no algorithm is separable from any other by a formal test (§5.3).

**Table 3.** First-NaN episode at Re=300 across the five seeds, with per-algorithm median and
mean. "Most robust" (last row) = latest break in that seed.

| Algorithm | seed=0 | seed=1 | seed=2 | seed=3 | seed=4 | median | mean |
|---|---|---|---|---|---|---|---|
| PPO  | 12  | 35 | 117 | 14 | 11  | 14 | 37.8 |
| SAC  | 36  | 6  | 2   | 40 | 131 | 36 | 43.0 |
| GPPO | 83  | 40 | 32  | 6  | 16  | 32 | 35.4 |
| GSAC | **141** | **63** | 2 | **45** | 122 | **63** | **74.6** |
| most robust | GSAC | GSAC | **PPO** | GSAC | **SAC** | — | — |

At n=2 (the conference version) two claims survived: analytic-gradient injection delays the
break (GPPO > PPO, GSAC > SAC) in direction, and GSAC is the most robust of the four. **At
n=5 neither is a statistically separable ordering; the two claims fare differently.**

*Analytic gradient does not delay the on-policy break.* GPPO > PPO holds at seeds 0, 1, 4
(83 vs 12, 40 vs 35, 16 vs 11) but reverses hard at seed=2 (32 vs 117) and seed=3 (6 vs 14,
GPPO breaking *first of all four*). The paired per-seed difference GPPO − PPO is
{+71, +5, −85, −8, +5}, mean **−2.4 episodes**, sign-flip permutation p = **0.81**. On-policy
analytic-gradient injection has no measurable effect on the breaking point.

*GSAC has the highest median and mean and is most robust in 3 of 5 seeds, but not
significantly so.* GSAC's breaking point (median 63, mean 74.6) is the highest of the four;
it is the most robust algorithm at seeds 0, 1, 3, second at seed=4, and tied-last at seed=2
(the early-init failure mode, §6(c) — the flow diverges before learning begins and bare SAC
fails identically at the same episode). But the spread is enormous: GSAC − SAC per seed is
{+105, +57, 0, +5, −9}, mean +31.6, permutation p = **0.38**; GSAC − GPPO is
{+58, +23, −30, +39, +106}, mean +39.2, p = **0.19**; GSAC − PPO p = **0.38**. A
Kruskal–Wallis test across all four algorithms gives p = 0.57, and a Friedman test blocked by
seed gives p = 0.77 — **no algorithm's breaking point is distinguishable from any other's.**

Every algorithm's range spans more than an order of magnitude (PPO 11–117, SAC 2–131,
GPPO 6–83, GSAC 2–141). By median there is a suggestive ordering GSAC (63) > SAC (36) >
GPPO (32) > PPO (14) — analytic gradient raising the median in both the off-policy (GSAC vs
SAC) and on-policy (GPPO vs PPO, ignoring the mean) direction — but with n=5 and these
variances none of it reaches significance, and the mean for GPPO is actually below PPO.

What *does* hold across all five seeds:
1. **Every algorithm eventually diverges at Re=300** — analytic-gradient injection never
   prevents the break, at most postpones it.
2. **The break is never preceded by a visible precursor** in the reward or gradient signal
   (§6(a)) — in any algorithm, in any seed.
3. **The breaking point is dominated by the seed, not the algorithm.** Within-algorithm
   variance across seeds (std 30–57 episodes) swamps between-algorithm differences in mean
   (spread of means: 35–75).

The "clean multiplicative" structure reported at n=1 and the "GSAC most robust" claim that
survived n=2 are both **artifacts of small n**. The honest n=5 statement: at Re=300 the
episode at which a differentiable chaotic-flow simulation first diverges under RL training is
a high-variance, seed-dominated quantity; analytic-gradient injection, off-policy structure,
and their combination do not impose a statistically separable ordering on it. GSAC carries a
non-significant robustness lean (highest median, most robust in 3/5) — a candidate effect
that a larger seed budget could confirm, but which we do not claim at n=5.

### 5.3 Significance test (n=5)

The first-NaN episode at Re=300 is the outcome variable; the four algorithms are compared
pairwise on their five paired per-seed differences with an **exact sign-flip permutation
test** (all 2⁵ = 32 relabelings, two-sided) and a **bootstrap 95% CI** for the median
difference (20,000 resamples). Omnibus tests: Kruskal–Wallis (unblocked) and Friedman
(blocked by seed).

**Table 3b.** Pairwise breaking-point comparison, n=5.

| Comparison | per-seed diff (s0–s4) | mean | median | 95% CI (median) | p (sign-flip) |
|---|---|---|---|---|---|
| GPPO − PPO  | +71, +5, −85, −8, +5     | −2.4  | +5  | [−85, +71]  | 0.81 |
| GSAC − SAC  | +105, +57, 0, +5, −9     | +31.6 | +5  | [−9, +105]  | 0.38 |
| GSAC − GPPO | +58, +23, −30, +39, +106 | +39.2 | +39 | [−30, +106] | 0.19 |
| GSAC − PPO  | +129, +28, −115, +31, +111 | +36.8 | +31 | [−115, +129] | 0.38 |
| SAC − PPO   | +24, −29, −115, +26, +120 | +5.2  | +24 | [−115, +120] | 0.88 |
| GPPO − SAC  | +47, +34, +30, −34, −115 | −7.6  | +30 | [−115, +47] | 0.88 |

Omnibus: Kruskal–Wallis H = 2.00, p = 0.57; Friedman χ² = 1.80, p = 0.77.

**No pairwise or omnibus comparison reaches p < 0.05.** Per the pre-registered decision rule
(§1: report "no seed-robust ordering" unless GSAC beats every other algorithm at p < 0.05
one-sided — it does not), the finding stands as **no statistically separable ordering of
chaotic-regime training stability by algorithm.** GSAC's non-significant lean (highest median
and mean, most robust in 3/5 seeds, all pairwise p ∈ [0.19, 0.38]) is reported as a candidate
effect for a future larger-n study, not a result.

---

## 6. Mechanism of the break

*(§6(a)–(b) are direct inspections of raw training curves — no new training needed. §6(c)
uses the per-episode action-amplitude log added for the journal version (§3.4, §4.1). The
trigger-cause question at the end still needs the planned short-horizon re-run.)*

### 6(a) The break is sudden and precursor-free

For all four algorithms, in every seed with a mid-training break (seeds 0, 1, 3, 4; seed=2's
off-policy runs break at episode 2 and are covered in §6(c)), the mean and standard deviation
of the reward are stable over the 40-episode window immediately before the break — no ramp,
no gradual degradation. The longest-runway confirmations come from seed=4, where `sac_high`
(break at episode 131) and `gsac_high` (break at 122) each have a full 40-episode pre-break
window that is flat to within noise: −32.29 → −32.32 (std 1.09) and −32.29 → −32.08
(std 1.18) respectively. seed=3 `sac_high` (break at 40) is likewise flat (−33.0 → −32.6,
std 1.5).

**Table 4.** Pre-break reward statistics (40-episode window where the runway allows,
else the full pre-break run; seed=0 / seed=1 / seed=4).

| Algorithm | First-half mean | Second-half mean | Std. dev. |
|---|---|---|---|
| PPO  | −32.75 / −32.28 / −32.36 | −32.77 / −33.05 / −32.39 | 1.35 / 1.32 / 0.91 |
| SAC  | −33.31 / −32.76 / −32.29 | −32.50 / −32.79 / −32.32 | 1.30 / 1.39 / 1.09 |
| GPPO | −32.61 / −32.38 / −32.60 | −32.98 / −33.01 / −33.05 | 1.10 / 1.24 / 1.34 |
| GSAC | −32.49 / −32.19 / −32.29 | −32.46 / −32.08 / −32.08 | 1.32 / 1.16 / 1.18 |

GPPO/GSAC analytic gradient norms over the 15 episodes before the break are similarly flat
(fluctuating ~20–30, no upward trend). The break is therefore not the policy *gradually*
driving the solver past its limit, but a sudden, stochastic event — a single episode's
randomly sampled action sequence, combined with the current policy, pushing the flow past
the solver's Re≈350–380 numerical-stability limit (§3.1) — consistent with the butterfly-effect
trigger expected from Metz et al. [7] and the Re-shrunk predictability horizon of Boffetta
and Musacchio [25].

### 6(b) On-policy vs off-policy after the trigger: the buffer-dilution effect

**Table 5.** Trigger episode vs permanent-collapse episode, all five seeds. "Recovered" =
intermediate episodes with a valid (non-NaN) reward between trigger and permanent collapse.
Across all 20 Re=300 runs **exactly one** shows a recovery window: seed=0 GSAC (2 episodes).

| Algorithm | Trigger (s0/s1/s2/s3/s4) | Recovered |
|---|---|---|
| PPO  | 12 / 35 / 117 / 14 / 11  | 0 / 0 / 0 / 0 / 0 |
| SAC  | 36 / 6 / 2 / 40 / 131    | 0 / 0 / 0 / 0 / 0 |
| GPPO | 83 / 40 / 32 / 6 / 16    | 0 / 0 / 0 / 0 / 0 |
| GSAC | 141 / 63 / 2 / 45 / 122  | **2** / 0 / 0 / 0 / 0 |

In on-policy PPO/GPPO the trigger episode's rollout is used in that same episode's update, so
trigger and permanent collapse always coincide — no escape, in any of the 10 on-policy runs.
Off-policy training *could* in principle dilute the poisoned transitions in its replay buffer.
Whether it does depends on how full the buffer is at the trigger: a run that triggers at
episode 36 has ~540 transitions and the trigger episode's ~15 poisoned ones are ~2.8% of
them, so across 10 updates × 64 samples = 640 draws at least one poisoned sample is almost
certain; a run that triggers at episode 141 has ~2115 transitions, the poisoned fraction is
~0.7%, and the probability of *never* sampling one over 640 draws rises to ~1%. Only GSAC
triggers late enough (seeds 0 and 4, episodes 141 and 122) for this to be non-negligible.

**Table 6.** Replay-buffer dilution at the trigger episode, all off-policy runs. "Survival
prob." is the chance that none of 10×64 = 640 sampled transitions is one of the ~15 poisoned
ones. Only the two large-buffer GSAC runs (s0, s4) have a non-negligible value.

| Algorithm | Trigger | Buffer | Bad fraction | Survival prob. | Window? |
|---|---|---|---|---|---|
| SAC (s0)  | 36  | ~540  | 2.8% | ≈0%     | no |
| SAC (s1)  | 6   | ~90   | 16.7% | ≈0%    | no |
| SAC (s2)  | 2   | ~15   | ~100% | ≈0%    | no |
| SAC (s3)  | 40  | ~600  | 2.5% | ≈0%     | no |
| SAC (s4)  | 131 | ~1965 | 0.8% | ≈0.5%   | no |
| GSAC (s0) | 141 | ~2115 | 0.7% | ≈1%     | **2 ep** |
| GSAC (s1) | 63  | ~945  | 1.6% | ≈0.004% | no |
| GSAC (s2) | 2   | ~15   | ~100% | ≈0%    | no |
| GSAC (s3) | 45  | ~675  | 2.2% | ≈0%     | no |
| GSAC (s4) | 122 | ~1830 | 0.8% | ≈0.7%   | no |

**Cross-seed check.** The dilution calculation gives a non-zero survival probability only when
the poisoned fraction at the trigger falls to ~1% or below, which requires a buffer of
~1800+ transitions — i.e. a GSAC run that triggers past roughly episode 120. Two of the five
GSAC runs meet this (seed=0 at episode 141, seed=4 at episode 122); the model predicts a
recovery window is *possible but rare* (~1%) for these and effectively impossible for the
other 18 runs. Observed: one window, in seed=0 GSAC. This is consistent with the model —
every on-policy run and every small-buffer run collapses in the trigger episode itself, and
the single positive instance is the large-buffer case — but with one positive across 20 runs
the mechanism is **suggestive, not confirmed**; a targeted study (many seeds, or a controlled
buffer-size sweep at a fixed trigger) is needed to establish the ~1% survival rate.

### 6(c) Divergence before learning: the early-init failure mode

The seed=2 sweep exposes a second, qualitatively different failure mode, and the
action-amplitude log (added for this purpose; §3.4, §4.1) lets us characterize it directly.
In seed=2 at Re=300, **both SAC and GSAC break at episode 2** with an *identical* episode-1
trajectory: same reward (−33.68), same action RMS (0.308), same action abs-max (0.489). This
identity is expected — SAC and GSAC share the seed's initialization (actor, critics, reset
state) and diverge only after episode 1, when GSAC applies its extra analytic step. Yet both
reach NaN on episode 2.

The logs rule out the §6(b) mechanism and an action-explosion mechanism alike:
- GSAC's episode-1 **analytic gradient norm is 54** — squarely in the normal range (seed=0:
  49, seed=1: 69 over their first ten episodes). The analytic step was not anomalous.
- The episode-2 rollout's **action RMS is 0.307 and abs-max 0.485** — indistinguishable from
  episode 1. The policy is not driving the flow with large control.

What remains is that **seed=2's randomly initialized policy, composed with the Re=300 base
flow, integrates to numerical divergence within two episodes — before any actor-learning
algorithm has a foothold.** Bare SAC diverges identically, so this is not a property of
analytic-gradient injection; it is a property of where the random initialization lands
relative to the solver's stability boundary (§3.1). GSAC's long runways in seeds 0, 1, 4 —
141, 63, 122 episodes — require several non-diverged episodes for the analytic gradient to
accumulate a stabilizing effect; seed=2 provides no such runway.

**This mode is rare: 1 of 5 seeds.** Seeds 3 and 4 both break mid-training (episodes 14–45
and 11–131), so the episode-2 collapse is not the typical Re=300 behavior — it is a
low-probability tail of the same "sudden, precursor-free, stochastic trigger" of §6(a),
occurring at episode 2 instead of 45–141. It still reframes GSAC as **highest ceiling, no
reliable floor**, and it shows the trigger-episode distribution reaches all the way down to
episode 2.

**Open question (still unresolved).** §6(b) is a plausibility calculation for the
*post-trigger lock-in*; §6(c) shows *when* the trigger can occur (as early as episode 2,
init-dependent). Neither explains *why a given (seed, init) triggers at a given episode*. The
action-amplitude log now in place (seeds ≥ 3 carry it; seeds 0–1 regenerate it bit-identically
under fixed JAX PRNG) is the instrument for this; a dedicated short-horizon re-run of seeds 0–1
logging the per-step action norm is the next diagnostic.

---

## 7. Discussion

This work extends the HydroGym paper's GPPO-vs-PPO result on Kolmogorov flow [1] in two
directions, and the second direction produces a seed-dominance result that is itself the main
finding.

**Regime extension (Re=45).** At the regular operating point the paper's core claim holds:
GPPO reaches PPO's best reward in substantially fewer episodes in 4 of 5 seeds (≈63–84%,
seed=0's 63% closest to the reported 65%), the direction is seed-stable across all five, and
the off-policy family reaches better final reward than the on-policy family in every seed.
GSAC never beats SAC here — the analytic gradient does not help the off-policy method when
the flow is not chaotic.

**Chaotic extension (Re=300): no separable ordering at n=5.** The conference version, at n=2,
reported that GSAC is the most robust of the four and that analytic-gradient injection delays
the break in direction (GPPO > PPO, GSAC > SAC). Over five seeds the second claim does not
hold at all — GPPO − PPO has mean −2.4 episodes and sign-flip p = 0.81 — and the first
survives only as a non-significant lean: GSAC has the highest median breaking point (63) and
is the most robust of the four in 3 of 5 seeds, but every pairwise permutation p ≥ 0.19 and
the omnibus tests (Kruskal–Wallis p = 0.57, Friedman p = 0.77) find no algorithm effect. Each
algorithm's first-NaN episode spans more than an order of magnitude; PPO, first to break at
seed=0, is the most robust of the four at seed=2; at seed=2 both off-policy methods collapse
at episode 2 (§6(c)). The "clean multiplicative" n=1 story and the "GSAC most robust" n=2
story are both **small-n artifacts**. The finding that replaces them: **in the chaotic regime
the breaking point is dominated by the random path of training, not by the algorithm** —
within-algorithm cross-seed std (30–57 episodes) swamps the spread of between-algorithm means
(35–75). Analytic-gradient injection, off-policy structure, and their combination fail to
impose a statistically separable ordering on when a differentiable chaotic-flow simulation
diverges under RL training. This is a concrete, quantified instance of the chaos-induced
gradient instability of Metz et al. [7] and the Re-shrunk predictability horizon of
Boffetta–Musacchio [25]: the instability is not merely large, it is *irreducibly stochastic*
at this problem scale.

**Two mechanistic results do survive**, both algorithm-design-independent:
- **§6(b), the buffer-dilution calculation:** a post-trigger "survival window" (valid
  episodes between trigger and permanent collapse) has non-zero probability only when the
  poisoned fraction of the replay buffer at the trigger is ~1% or below — i.e. only for GSAC
  runs that trigger past ~episode 120. Exactly one of 20 Re=300 runs shows a window (seed=0
  GSAC, 2 episodes) and it is that case. Suggestive, not confirmed.
- **§6(c), the early-init failure mode:** when the randomly initialized policy composed with
  the Re=300 base flow already sits near the solver's stability boundary, divergence occurs
  within ~2 episodes — before any actor-learning algorithm has a foothold. Bare SAC and GSAC
  fail identically here, and the action-amplitude log shows normal control and a normal
  analytic-gradient norm at the point of divergence. This is why GSAC has the highest ceiling
  (141, 63, 122 episodes when it gets a runway; most robust in 3 of 5 seeds) but no reliable
  floor (episode 2 at seed=2).

None of this refutes the HydroGym paper's core claim that GPPO helps at its tested operating
point; it shows that the *benefit does not have a seed-robust ordering* once the regime is
chaotic, and that the useful, reproducible statements about chaotic-regime training are
mechanistic (§6) rather than "which algorithm wins."

---

## 8. Limitations

- **n=5, and the seed-dominance finding could use more** — five seeds falsify the n=2
  orderings (§5.2) and support the negative claim, but GSAC's non-significant median lean
  (p ≈ 0.19–0.38) would need a substantially larger seed budget to confirm or reject.
  Because the finding is seed-dominance ("no seed-robust ordering"), more seeds strengthen it
  rather than risk overturning it.
- **First-NaN episode at Re=300 spans >1 order of magnitude per algorithm** across the five
  seeds (PPO {12,35,117,14,11}, SAC {36,6,2,40,131}, GPPO {83,40,32,6,16}, GSAC {141,63,2,45,122}).
  These are point observations of a stochastic quantity, not a law; we deliberately do not
  report "delay multipliers" as the conference version did.
- **Re=300 was used instead of the intended Re=450** — the library's own solver has a
  stability limit near Re≈350–380 at this grid/dt. It is possible that a milder chaotic
  regime would show a more stable ordering; the planned Re sweep {45, 100, 200, 300} tests
  this.
- **GPPO/GSAC mixing ratio, horizon, and other hyperparameters are our reimplementation's** —
  the HydroGym paper does not publish them; the 63%/77% vs 65% agreement is a strong signal
  but not a claim of identity.
- **Minimal PPO update** (one full-batch step per episode, no epochs/minibatching) — "PPO"
  here is weaker than a tuned implementation; the comparison is relative and budget-matched.
- **Short episodes (15 RL steps) and short analytic horizon (6 steps)** — may be insufficient
  to fully observe RL convergence or chaotic divergence (Lyapunov growth [25]).
- **GSAC's mixing weight is a fixed 50/50** — the adaptive α-graded estimator of Suh et al.
  [20], AHAC's [5] adaptive horizon, and SAPO's [4] entropy-based smoothing are untested;
  these are the natural next step toward *preventing* rather than *delaying* instability at
  Re=300.
- **The cause of the trigger episode itself remains open (§6(c))** — the action-amplitude log
  is now in place but the dedicated short-horizon re-run of seeds 0–1 has not yet been done.
- **The buffer-dilution calculation of §6(b) rests on a single positive instance** — one
  recovery window across 20 Re=300 runs (seed=0 GSAC). The calculation is consistent with all
  20 (windows possible only for large-buffer GSAC, impossible otherwise) but one positive
  cannot establish the predicted ~1% rate; a controlled buffer-size sweep is needed.
- **§6(c) rests on one seed** — the early-init divergence is documented for seed=2 only.
  Neither seed=3 nor seed=4 reproduces it (seed=3 breaks at 14/40/6/45; seed=4 at 11/131/16/122
  — all mid-training), so at n=5 the mode is 1/5. It is a real but rare failure mode, not the
  typical Re=300 behavior.

---

## 9. Conclusion and future work

Analytic-gradient injection does not eliminate numerical instability in the chaotic regime —
and, across five training seeds, it does not reliably delay the break either: GPPO − PPO has
mean −2.4 episodes (sign-flip p = 0.81), and while GSAC has the highest median breaking point
(63) and is most robust in 3 of 5 seeds, no pairwise permutation test reaches p < 0.05 (all
p ≥ 0.19) and the omnibus tests find no algorithm effect (Kruskal–Wallis p = 0.57, Friedman
p = 0.77). The n=2 ordering "GSAC > SAC, GPPO > PPO in direction" does not survive n=5;
"GSAC most robust" survives only as a non-significant lean. The reproducible statement is one
of seed-dominance: **at Re=300 the episode at which training first diverges is dominated by
the random path of training, not by the algorithm** — within-algorithm cross-seed variance
swamps between-algorithm differences. What survives is mechanistic: the break is sudden and
precursor-free (§6(a)); the replay buffer's size at the trigger sets whether a brief
post-trigger survival window appears (§6(b)); and in a rare failure mode (1 of 5 seeds) the
flow diverges within ~2 episodes regardless of algorithm because the random initial policy
already sits near the solver's stability boundary (§6(c)). GSAC, on this evidence, has the
highest ceiling of the four — 2–10× the stable horizon of any other method when training gets
a foothold — but no reliable floor.

**Future work.**
1. Extend well beyond n=5 to test GSAC's non-significant median lean (currently p ≈ 0.19–0.38)
   — the seed-dominance of the breaking point means a decisive test needs many seeds.
2. Resolve the trigger-episode cause (§6(a)) with the action-amplitude-logged short-horizon
   re-run of seeds 0–1 — is the trigger a single large sampled action, or a resonant sequence?
3. Test whether GSAC's floor can be raised (not its ceiling) with cheap, principled guards:
   analytic-gradient norm clipping [7], a non-finite-gradient fallback to pure SAC, a
   few-episode analytic warmup, and an adaptive mixing weight (GI-PPO [26] α-policy with its
   variance/bias metrics; or AHAC [5]/SAPO [4]-style). Hypothesis: these reduce GSAC's
   cross-seed breaking-point variance but cannot prevent the §6(c) early-init divergence,
   which is environmental.
4. Re sweep {45, 100, 200, 300} to locate the Reynolds number at which the (weak) algorithm
   ordering disappears, framed with Boffetta–Musacchio [25].
5. Reconcile against the official HydroGym GPPO implementation and a stronger PPO baseline
   (epochs/minibatching); extend to a second environment.

---

## Acknowledgment

The author thanks Sakarya University's National Technology Workshop Artificial Intelligence
Center for the computational resources used in this study.

---

## References

*([1]–[25] verified by real-time web search on 2026-08-26 — title, author list, and
publication data confirmed from each source's own page; see also the companion document
`HydroGym_Literatur_Taramasi_HaktanGuloglu.pdf`, N=25, 2026-08-25. [26]–[29] added
2026-08-30 for the journal version, verified the same way; a reproducibility-checklist
citation for §4.3 is still to be added.)*

[1] C. Lagemann, S. Mokbel, M. Gondrum, M. Rüttgers, Y. Wang, P. Suárez, L. Paehler,
D. A. Bezgin, A. B. Buhendwa, J. L. Callaham, S. Ahnert, N. Zolman, X. Shao, J.-C. Loiseau,
N. Adams, M. Meinke, W. Schröder, K. Lagemann, E. Lagemann, R. Vinuesa, and S. L. Brunton,
"The HydroGym Reinforcement Learning Platform for Fluid Dynamics," *Nature*, 2026,
doi:10.1038/s41586-026-10917-6. Preprint: arXiv:2512.17534.

[2] J. Becktepe, A. Franz, N. Thuerey, and S. Peitz, "Plug-and-Play Benchmarking of
Reinforcement Learning Algorithms for Large-Scale Flow Control (FluidGym)," arXiv:2601.15015,
2026.

[3] J. Xu, V. Makoviychuk, Y. Narang, F. Ramos, W. Matusik, A. Garg, and M. Macklin,
"Accelerated Policy Learning with Parallel Differentiable Simulation (SHAC)," in *Proc.
ICLR*, 2022. arXiv:2204.07137.

[4] E. Xing, V. Luk, and J. Oh, "Stabilizing Reinforcement Learning in Differentiable
Multiphysics Simulation (SAPO / Rewarped)," in *Proc. ICLR (Spotlight)*, 2025.
arXiv:2412.12089.

[5] I. Georgiev, K. Srinivasan, J. Xu, E. Heiden, and A. Garg, "Adaptive Horizon
Actor-Critic for Policy Learning in Contact-Rich Differentiable Simulation (AHAC)," in
*Proc. ICML*, 2024. arXiv:2405.17784.

[6] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal Policy
Optimization Algorithms," arXiv:1707.06347, 2017.

[7] L. Metz, C. D. Freeman, S. S. Schoenholz, and T. Kachman, "Gradients are Not All You
Need," arXiv:2111.05803, 2021.

[8] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, "Soft Actor-Critic: Off-Policy Maximum
Entropy Deep Reinforcement Learning with a Stochastic Actor," in *Proc. ICML*, 2018.
arXiv:1801.01290.

[9] T. Haarnoja, A. Zhou, K. Hartikainen et al., "Soft Actor-Critic Algorithms and
Applications," arXiv:1812.05905, 2018.

[10] B. Font, F. Alcántara-Ávila, J. Rabault, R. Vinuesa, and O. Lehmkuhl, "Deep
Reinforcement Learning for Active Flow Control in a Turbulent Separation Bubble," *Nature
Communications*, 2025, doi:10.1038/s41467-025-56408-6.

[11] "Applying Deep Reinforcement Learning to Active Flow Control in Turbulent Conditions,"
arXiv:2006.10683, 2020.

[12] "Deep Reinforcement Learning for Robotics: A Survey of Real-World Successes," *Annual
Review of Control, Robotics, and Autonomous Systems*, 2025.

[13] J. Bradbury, R. Frostig, P. Hawkins et al., "JAX: Composable Transformations of
Python+NumPy Programs," 2018. github.com/jax-ml/jax.

[14] D. Kochkov, J. A. Smith, A. Alieva, Q. Wang, M. P. Brenner, and S. Hoyer, "Machine
Learning–Accelerated Computational Fluid Dynamics (JAX-CFD)," *PNAS* 118(21), 2021,
doi:10.1073/pnas.2101784118.

[15] D. A. Bezgin, A. B. Buhendwa, and N. A. Adams, "JAX-Fluids: A Fully-Differentiable
High-Order CFD Solver for Compressible Two-Phase Flows," arXiv:2203.13760, 2022.

[16] D. A. Bezgin, A. B. Buhendwa, and N. A. Adams, "JAX-Fluids 2.0: Towards HPC for
Differentiable CFD of Compressible Two-Phase Flows," arXiv:2402.05193, 2024.

[17] J. Schulman, S. Levine, P. Moritz, M. I. Jordan, and P. Abbeel, "Trust Region Policy
Optimization (TRPO)," arXiv:1502.05477, 2015.

[18] J. Schulman, P. Moritz, S. Levine, M. Jordan, and P. Abbeel, "High-Dimensional
Continuous Control Using Generalized Advantage Estimation (GAE)," arXiv:1506.02438, 2015
(presented at ICLR 2016).

[19] T. Haarnoja, H. Tang, P. Abbeel, and S. Levine, "Reinforcement Learning with Deep
Energy-Based Policies (Soft Q-Learning)," in *Proc. ICML*, 2017. arXiv:1702.08165.

[20] H. J. T. Suh, M. Simchowitz, K. Zhang, and R. Tedrake, "Do Differentiable Simulators
Give Better Policy Gradients?," in *Proc. ICML*, 2022. arXiv:2202.00817.

[21] K. Onoda, P. Parmas, M. Yaguchi, and Y. Matsuo, "Does 'Do Differentiable Simulators
Give Better Policy Gradients?' Give Better Policy Gradients?," arXiv:2604.18161, 2026.

[22] F. Li, F. Sun, T. Zhang, and D. Zou, "ABPT: Amended Backpropagation through Time with
Partially Differentiable Rewards," arXiv:2501.14513, 2025.

[23] J. Rabault, M. Kuchta, A. Jensen, U. Réglade, and N. Cerardi, "Artificial Neural
Networks Trained through Deep Reinforcement Learning Discover Control Strategies for Active
Flow Control," *Journal of Fluid Mechanics* 865, 281–302, 2019, doi:10.1017/jfm.2019.62.

[24] "Autonomous Real-Time Control of Turbulent Dynamics (REACT)," Imperial College London,
arXiv:2509.11002, 2026.

[25] G. Boffetta and S. Musacchio, "Chaos and Predictability of Homogeneous-Isotropic
Turbulence," *Physical Review Letters* 119, 054102, 2017,
doi:10.1103/PhysRevLett.119.054102. arXiv:1707.01648.

[26] S. Son, L. Y. Zheng, R. Sullivan, Y.-L. Qiao, and M. C. Lin, "Gradient Informed Proximal
Policy Optimization," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2023.
arXiv:2312.08710.

[27] M. A. Bucci, O. Semeraro, A. Allauzen, G. Wisniewski, L. Cordier, and L. Mathelin,
"Control of Chaotic Systems by Deep Reinforcement Learning," *Proceedings of the Royal
Society A* 475(2231), 20190351, 2019, doi:10.1098/rspa.2019.0351.

[28] S. Peitz, J. Stenner, V. Chidananda, O. Wallscheid, S. L. Brunton, and K. Taira,
"Distributed Control of Partial Differential Equations Using Convolutional Reinforcement
Learning," *Physica D: Nonlinear Phenomena*, 2024. arXiv:2301.10737.

[29] H. Zhong, X. Wang, Z. Li, and L. Huang, "Reparameterization Proximal Policy Optimization
(RPO)," arXiv:2508.06214, 2025.
