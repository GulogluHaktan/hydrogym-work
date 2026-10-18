"""
PPO and GPPO on the REAL hydrogym.jax KolmogorovFlow environment
(hydrogym/jax/envs/kolmogorov.py), not a custom reimplementation.

Uses FastKolmogorovFlow (fast_env.py) which only overrides action_time/dt/
max_episode_steps/reward_alpha to POC-appropriate scales (the base class
hardcodes action_time=10.0, dt=0.001, reward_alpha=0.0, which is far too
slow for a same-night run).
"""
import time
import jax
import jax.numpy as jnp
import optax

from fast_env import FastKolmogorovFlow
from safeguards import SafeguardConfig, process_analytic

HIDDEN = 64


def init_params(key, obs_dim, act_dim):
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    def dense(k, din, dout, scale=1.0):
        kw, _ = jax.random.split(k)
        w = jax.random.normal(kw, (din, dout)) * jnp.sqrt(2.0 / din) * scale
        b = jnp.zeros((dout,))
        return {"w": w, "b": b}

    return {
        "pi1": dense(k1, obs_dim, HIDDEN),
        "pi2": dense(k2, HIDDEN, HIDDEN),
        "pi_mean": dense(k3, HIDDEN, act_dim, scale=0.1),
        "log_std": jnp.zeros((act_dim,)) - 1.0,
        "v1": dense(k4, obs_dim, HIDDEN),
        "v2": dense(k5, HIDDEN, 1),
    }


def _mlp(p1, p2, x):
    x = jax.nn.tanh(x @ p1["w"] + p1["b"])
    x = jax.nn.tanh(x @ p2["w"] + p2["b"])
    return x


def policy_mean(params, obs):
    h = _mlp(params["pi1"], params["pi2"], obs)
    return 0.5 * jax.nn.tanh(h @ params["pi_mean"]["w"] + params["pi_mean"]["b"])


def value_fn(params, obs):
    h = jax.nn.tanh(obs @ params["v1"]["w"] + params["v1"]["b"])
    return (h @ params["v2"]["w"] + params["v2"]["b"])[0]


def sample_action(params, obs, key):
    mean = policy_mean(params, obs)
    std = jnp.exp(params["log_std"])
    eps = jax.random.normal(key, mean.shape)
    action = mean + std * eps
    logp = -0.5 * jnp.sum(((action - mean) / std) ** 2 + 2 * jnp.log(std) + jnp.log(2 * jnp.pi))
    return action, logp


def log_prob(params, obs, action):
    mean = policy_mean(params, obs)
    std = jnp.exp(params["log_std"])
    return -0.5 * jnp.sum(((action - mean) / std) ** 2 + 2 * jnp.log(std) + jnp.log(2 * jnp.pi))


def make_rollout_fns(env, params_env, n_steps, short_horizon):
    dummy_key = jax.random.PRNGKey(0)

    def rollout_episode(policy_params, init_obs, init_state, key):
        def body(carry, key_t):
            obs, state = carry
            action, logp = sample_action(policy_params, obs, key_t)
            value = value_fn(policy_params, obs)
            next_obs, next_state, reward, done, info = env.step_env(dummy_key, state, action, params_env)
            return (next_obs, next_state), (obs, action, logp, value, reward)

        keys = jax.random.split(key, n_steps)
        (final_obs, final_state), (obs, actions, logps, values, rewards) = jax.lax.scan(
            body, (init_obs, init_state), keys
        )
        return obs, actions, logps, values, rewards

    def analytic_short_rollout_loss(policy_params, init_obs, init_state):
        def body(carry, _):
            obs, state = carry
            action = policy_mean(policy_params, obs)
            next_obs, next_state, reward, done, info = env.step_env(dummy_key, state, action, params_env)
            return (next_obs, next_state), reward

        (_, _), rewards = jax.lax.scan(body, (init_obs, init_state), xs=None, length=short_horizon)
        return -jnp.sum(rewards)

    return rollout_episode, analytic_short_rollout_loss


def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    last_value = values[-1]
    values_ext = jnp.concatenate([values, last_value[None]])

    def body(carry, t):
        gae = carry
        delta = rewards[t] + gamma * values_ext[t + 1] - values_ext[t]
        gae = delta + gamma * lam * gae
        return gae, gae

    ts = jnp.arange(len(rewards))[::-1]
    _, advs = jax.lax.scan(body, 0.0, ts)
    advs = advs[::-1]
    returns = advs + values
    return advs, returns


def ppo_loss(params, obs, actions, old_logps, advs, returns, clip_eps=0.2):
    new_logps = jax.vmap(lambda o, a: log_prob(params, o, a))(obs, actions)
    ratio = jnp.exp(new_logps - old_logps)
    advs_n = (advs - advs.mean()) / (advs.std() + 1e-8)
    surr1 = ratio * advs_n
    surr2 = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * advs_n
    policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))
    values = jax.vmap(lambda o: value_fn(params, o))(obs)
    value_loss = jnp.mean((values - returns) ** 2)
    entropy = jnp.mean(0.5 * jnp.sum(1.0 + jnp.log(2 * jnp.pi) + 2 * params["log_std"]))
    return policy_loss + 0.5 * value_loss - 0.001 * entropy


def grad_l2_norm(grad_pytree):
    leaves = jax.tree_util.tree_leaves(grad_pytree)
    return jnp.sqrt(sum(jnp.sum(x ** 2) for x in leaves))


def add_grads(g1, g2, w1=0.5, w2=0.5):
    return jax.tree_util.tree_map(lambda a, b: w1 * a + w2 * b, g1, g2)


def train(algo, Re, n_episodes, n_steps=15, short_horizon=6, seed=0, lr=3e-4,
          dt=0.005, action_time=1.0, log_every=5,
          ppo_epochs=1, ppo_minibatches=1, sg=None):
    """PPO / GPPO on the real Kolmogorov env.

    With the default arguments (ppo_epochs=1, ppo_minibatches=1, sg=None) this
    is bit-identical to the original seed=0..4 sweep. The extra knobs drive
    IYILESTIRME_ONERILERI.md packages B and C:

      * ppo_epochs / ppo_minibatches > 1  -> package B (a real PPO inner loop
        instead of one full-batch step per episode).
      * sg (SafeguardConfig)              -> package C (analytic-gradient
        clip / non-finite fallback / warmup / adaptive mix; GPPO only).
    """
    sg = sg or SafeguardConfig()
    strong = (ppo_epochs > 1) or (ppo_minibatches > 1)

    env = FastKolmogorovFlow(flow_config={"Re": Re}, dt=dt, action_time=action_time,
                              max_episode_steps=n_steps, reward_alpha=1.0)
    params_env = env.default_params
    obs_dim, act_dim = params_env.obs_dim, params_env.action_dim

    rollout_episode, analytic_loss_fn = make_rollout_fns(env, params_env, n_steps, short_horizon)
    rollout_jit = jax.jit(rollout_episode)
    ppo_grad_fn = jax.jit(jax.grad(ppo_loss))
    analytic_grad_fn = jax.jit(jax.grad(analytic_loss_fn))

    key = jax.random.PRNGKey(seed)
    key, pkey = jax.random.split(key)
    policy_params = init_params(pkey, obs_dim, act_dim)
    opt = optax.adam(lr)
    opt_state = opt.init(policy_params)

    reset_key = jax.random.PRNGKey(seed + 1000)
    init_obs0, init_state0 = env.reset_env(reset_key, params_env)

    episode_rewards, grad_norms, wall_times = [], [], []
    action_rms, action_absmax = [], []
    mix_weights, safeguard_events = [], []
    t_start = time.time()

    for ep in range(n_episodes):
        key, akey, gkey = jax.random.split(key, 3)

        obs, actions, logps, values, rewards = rollout_jit(policy_params, init_obs0, init_state0, akey)
        # per-episode action-amplitude log (diagnostic for the trigger episode, §6)
        action_rms.append(float(jnp.sqrt(jnp.mean(actions ** 2))))
        action_absmax.append(float(jnp.max(jnp.abs(actions))))
        advs, returns = compute_gae(rewards, values)

        if not strong:
            # ---- original path (one full-batch PPO gradient step per episode) ----
            g_ppo = ppo_grad_fn(policy_params, obs, actions, logps, advs, returns)
            if algo == "gppo":
                g_analytic = analytic_grad_fn(policy_params, init_obs0, init_state0)
                gnorm = float(grad_l2_norm(g_analytic))
                grad_norms.append(gnorm)
                if sg.active:
                    g_analytic, w_an, w_mf, info = process_analytic(g_analytic, gnorm, ep, sg)
                    mix_weights.append(w_an)
                    if info["reason"] != "mix":
                        safeguard_events.append(info)
                    g_final = add_grads(g_analytic, g_ppo, w_an, w_mf)
                else:
                    g_final = add_grads(g_analytic, g_ppo, 0.5, 0.5)
            else:
                g_final = g_ppo
            updates, opt_state = opt.update(g_final, opt_state, policy_params)
            policy_params = optax.apply_updates(policy_params, updates)
        else:
            # ---- package B: PPO inner loop (epochs x minibatches) ----
            B = obs.shape[0]
            mb = max(1, B // ppo_minibatches)
            g_ppo_last = None
            for _ in range(ppo_epochs):
                key, perm_key = jax.random.split(key)
                perm = jax.random.permutation(perm_key, B)
                for s in range(0, B, mb):
                    idx = perm[s:s + mb]
                    g_ppo_last = ppo_grad_fn(policy_params, obs[idx], actions[idx],
                                             logps[idx], advs[idx], returns[idx])
                    updates, opt_state = opt.update(g_ppo_last, opt_state, policy_params)
                    policy_params = optax.apply_updates(policy_params, updates)
            if algo == "gppo":
                # one analytic step after the PPO inner loop, mirroring GSAC's
                # extra actor step (mixed with the final minibatch PPO grad).
                g_analytic = analytic_grad_fn(policy_params, init_obs0, init_state0)
                gnorm = float(grad_l2_norm(g_analytic))
                grad_norms.append(gnorm)
                g_analytic, w_an, w_mf, info = process_analytic(g_analytic, gnorm, ep, sg)
                mix_weights.append(w_an)
                if info["reason"] != "mix":
                    safeguard_events.append(info)
                g_final = add_grads(g_analytic, g_ppo_last, w_an, w_mf)
                updates, opt_state = opt.update(g_final, opt_state, policy_params)
                policy_params = optax.apply_updates(policy_params, updates)

        ep_reward = float(jnp.sum(rewards))
        episode_rewards.append(ep_reward)
        wall_times.append(time.time() - t_start)

        if (ep + 1) % log_every == 0 or ep == 0:
            extra = f" | grad_norm={grad_norms[-1]:.4g}" if grad_norms else ""
            print(f"[{algo} Re={Re}] ep {ep+1}/{n_episodes} reward={ep_reward:.4f}{extra} "
                  f"t={wall_times[-1]:.1f}s")

    return {
        "algo": algo, "Re": Re,
        "episode_rewards": episode_rewards,
        "grad_norms": grad_norms,
        "wall_times": wall_times,
        "action_rms": action_rms,
        "action_absmax": action_absmax,
        "ext_config": {
            "ppo_epochs": ppo_epochs, "ppo_minibatches": ppo_minibatches,
            "strong_ppo": strong, "safeguards": sg.as_dict(),
        },
        "mix_weights": mix_weights,
        "safeguard_events": safeguard_events,
    }
