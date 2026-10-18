"""
SAC and GSAC (gradient-enhanced SAC) on the REAL hydrogym.jax KolmogorovFlow
environment. Mirrors real_rl.py's PPO/GPPO structure but off-policy:
squashed-Gaussian actor, twin Q critics with target networks, a simple
replay buffer, and (for GSAC) a 50/50 mix of the actor's SAC gradient with
an analytic short-horizon reward gradient backpropagated through the
differentiable simulator (same mechanism as GPPO, applied to the actor).
"""
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax

from fast_env import FastKolmogorovFlow
from safeguards import SafeguardConfig, process_analytic

HIDDEN = 64
LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0
ACTION_SCALE = 0.5  # matches env's min/max_action = -0.5/0.5


def _dense(k, din, dout, scale=1.0):
    kw, _ = jax.random.split(k)
    w = jax.random.normal(kw, (din, dout)) * jnp.sqrt(2.0 / din) * scale
    b = jnp.zeros((dout,))
    return {"w": w, "b": b}


def init_actor(key, obs_dim, act_dim):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    return {
        "h1": _dense(k1, obs_dim, HIDDEN),
        "h2": _dense(k2, HIDDEN, HIDDEN),
        "mean": _dense(k3, HIDDEN, act_dim, scale=0.1),
        "log_std": _dense(k4, HIDDEN, act_dim, scale=0.1),
    }


def init_critic(key, obs_dim, act_dim):
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "h1": _dense(k1, obs_dim + act_dim, HIDDEN),
        "h2": _dense(k2, HIDDEN, HIDDEN),
        "out": _dense(k3, HIDDEN, 1),
    }


def _mlp2(p1, p2, x):
    x = jax.nn.relu(x @ p1["w"] + p1["b"])
    x = jax.nn.relu(x @ p2["w"] + p2["b"])
    return x


def actor_dist(params, obs):
    h = _mlp2(params["h1"], params["h2"], obs)
    mean = h @ params["mean"]["w"] + params["mean"]["b"]
    log_std = h @ params["log_std"]["w"] + params["log_std"]["b"]
    log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
    return mean, log_std


def sample_squashed(params, obs, key):
    mean, log_std = actor_dist(params, obs)
    std = jnp.exp(log_std)
    eps = jax.random.normal(key, mean.shape)
    pre_tanh = mean + std * eps
    squashed = jnp.tanh(pre_tanh)
    action = ACTION_SCALE * squashed
    logp_gauss = -0.5 * jnp.sum(eps ** 2 + 2 * log_std + jnp.log(2 * jnp.pi))
    logp = logp_gauss - jnp.sum(jnp.log(1 - squashed ** 2 + 1e-6)) - mean.shape[-1] * jnp.log(ACTION_SCALE)
    return action, logp


def deterministic_action(params, obs):
    mean, _ = actor_dist(params, obs)
    return ACTION_SCALE * jnp.tanh(mean)


def critic_q(params, obs, action):
    x = jnp.concatenate([obs, action], axis=-1)
    h = _mlp2(params["h1"], params["h2"], x)
    return (h @ params["out"]["w"] + params["out"]["b"])[0]


def grad_l2_norm(grad_pytree):
    leaves = jax.tree_util.tree_leaves(grad_pytree)
    return jnp.sqrt(sum(jnp.sum(x ** 2) for x in leaves))


def add_grads(g1, g2, w1=0.5, w2=0.5):
    return jax.tree_util.tree_map(lambda a, b: w1 * a + w2 * b, g1, g2)


def soft_update(target, online, tau):
    return jax.tree_util.tree_map(lambda t, o: (1 - tau) * t + tau * o, target, online)


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, act_dim):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros((capacity,), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.ptr, self.size = 0, 0

    def add_batch(self, obs, act, rew, next_obs):
        n = obs.shape[0]
        idx = (self.ptr + np.arange(n)) % self.capacity
        self.obs[idx] = obs
        self.act[idx] = act
        self.rew[idx] = rew
        self.next_obs[idx] = next_obs
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size, rng):
        idx = rng.integers(0, self.size, size=min(batch_size, self.size))
        return (jnp.array(self.obs[idx]), jnp.array(self.act[idx]),
                jnp.array(self.rew[idx]), jnp.array(self.next_obs[idx]))


def make_rollout_and_losses(env, params_env, n_steps, short_horizon, gamma=0.99, alpha=0.1):
    dummy_key = jax.random.PRNGKey(0)

    def rollout_episode(actor_params, init_obs, init_state, key):
        def body(carry, key_t):
            obs, state = carry
            action, _ = sample_squashed(actor_params, obs, key_t)
            next_obs, next_state, reward, done, info = env.step_env(dummy_key, state, action, params_env)
            return (next_obs, next_state), (obs, action, reward, next_obs)

        keys = jax.random.split(key, n_steps)
        (final_obs, final_state), (obs, actions, rewards, next_obs) = jax.lax.scan(
            body, (init_obs, init_state), keys
        )
        return obs, actions, rewards, next_obs

    def critic_loss_fn(critic_params, target_critic1, target_critic2, actor_params,
                        obs, act, rew, next_obs, key):
        next_action, next_logp = jax.vmap(lambda o, k: sample_squashed(actor_params, o, k))(
            next_obs, jax.random.split(key, obs.shape[0])
        )
        q1_t = jax.vmap(lambda o, a: critic_q(target_critic1, o, a))(next_obs, next_action)
        q2_t = jax.vmap(lambda o, a: critic_q(target_critic2, o, a))(next_obs, next_action)
        target_q = jnp.minimum(q1_t, q2_t) - alpha * next_logp
        y = rew + gamma * target_q
        y = jax.lax.stop_gradient(y)
        q_pred = jax.vmap(lambda o, a: critic_q(critic_params, o, a))(obs, act)
        return jnp.mean((q_pred - y) ** 2)

    def actor_loss_fn(actor_params, critic1, critic2, obs, key):
        action, logp = jax.vmap(lambda o, k: sample_squashed(actor_params, o, k))(
            obs, jax.random.split(key, obs.shape[0])
        )
        q1 = jax.vmap(lambda o, a: critic_q(critic1, o, a))(obs, action)
        q2 = jax.vmap(lambda o, a: critic_q(critic2, o, a))(obs, action)
        q = jnp.minimum(q1, q2)
        return jnp.mean(alpha * logp - q)

    def analytic_short_rollout_loss(actor_params, init_obs, init_state):
        def body(carry, _):
            obs, state = carry
            action = deterministic_action(actor_params, obs)
            next_obs, next_state, reward, done, info = env.step_env(dummy_key, state, action, params_env)
            return (next_obs, next_state), reward

        (_, _), rewards = jax.lax.scan(body, (init_obs, init_state), xs=None, length=short_horizon)
        return -jnp.sum(rewards)

    return rollout_episode, critic_loss_fn, actor_loss_fn, analytic_short_rollout_loss


def train(algo, Re, n_episodes, n_steps=15, short_horizon=6, seed=0, lr=3e-4,
          dt=0.001, action_time=1.0, updates_per_episode=10, batch_size=64,
          tau=0.005, log_every=5, sg=None):
    """SAC / GSAC on the real Kolmogorov env.

    With sg=None this is bit-identical to the original seed=0..4 sweep. Passing
    a SafeguardConfig drives IYILESTIRME_ONERILERI.md package C for GSAC:
    analytic-gradient clip / non-finite fallback / warmup / adaptive mix. The
    per-episode update count is unchanged in every branch (a dropped analytic
    term still takes the pure-SAC extra actor step).
    """
    assert algo in ("sac", "gsac")
    sg = sg or SafeguardConfig()
    env = FastKolmogorovFlow(flow_config={"Re": Re}, dt=dt, action_time=action_time,
                              max_episode_steps=n_steps, reward_alpha=1.0)
    params_env = env.default_params
    obs_dim, act_dim = params_env.obs_dim, params_env.action_dim

    rollout_episode, critic_loss_fn, actor_loss_fn, analytic_loss_fn = make_rollout_and_losses(
        env, params_env, n_steps, short_horizon
    )
    rollout_jit = jax.jit(rollout_episode)
    critic_grad_fn = jax.jit(jax.grad(critic_loss_fn))
    actor_grad_fn = jax.jit(jax.grad(actor_loss_fn))
    analytic_grad_fn = jax.jit(jax.grad(analytic_loss_fn))

    key = jax.random.PRNGKey(seed)
    key, ak, c1k, c2k = jax.random.split(key, 4)
    actor_params = init_actor(ak, obs_dim, act_dim)
    critic1 = init_critic(c1k, obs_dim, act_dim)
    critic2 = init_critic(c2k, obs_dim, act_dim)
    target_critic1, target_critic2 = critic1, critic2

    actor_opt = optax.adam(lr)
    critic_opt = optax.adam(lr)
    actor_opt_state = actor_opt.init(actor_params)
    critic1_opt_state = critic_opt.init(critic1)
    critic2_opt_state = critic_opt.init(critic2)

    reset_key = jax.random.PRNGKey(seed + 1000)
    init_obs0, init_state0 = env.reset_env(reset_key, params_env)

    buffer = ReplayBuffer(capacity=n_episodes * n_steps, obs_dim=obs_dim, act_dim=act_dim)
    rng = np.random.default_rng(seed)

    episode_rewards, grad_norms, wall_times = [], [], []
    action_rms, action_absmax = [], []
    mix_weights, safeguard_events = [], []
    t_start = time.time()

    for ep in range(n_episodes):
        key, akey, gkey = jax.random.split(key, 3)

        obs, actions, rewards, next_obs = rollout_jit(actor_params, init_obs0, init_state0, akey)
        # per-episode action-amplitude log (diagnostic for the trigger episode, §6)
        _a = np.asarray(actions)
        action_rms.append(float(np.sqrt(np.mean(_a ** 2))))
        action_absmax.append(float(np.max(np.abs(_a))))
        buffer.add_batch(np.array(obs), np.array(actions), np.array(rewards), np.array(next_obs))

        for _ in range(updates_per_episode):
            b_obs, b_act, b_rew, b_next_obs = buffer.sample(batch_size, rng)
            key, ck, ak2 = jax.random.split(key, 3)

            c_grad1 = critic_grad_fn(critic1, target_critic1, target_critic2, actor_params,
                                      b_obs, b_act, b_rew, b_next_obs, ck)
            c_grad2 = critic_grad_fn(critic2, target_critic1, target_critic2, actor_params,
                                      b_obs, b_act, b_rew, b_next_obs, ck)
            c1_upd, critic1_opt_state = critic_opt.update(c_grad1, critic1_opt_state, critic1)
            c2_upd, critic2_opt_state = critic_opt.update(c_grad2, critic2_opt_state, critic2)
            critic1 = optax.apply_updates(critic1, c1_upd)
            critic2 = optax.apply_updates(critic2, c2_upd)

            a_grad = actor_grad_fn(actor_params, critic1, critic2, b_obs, ak2)
            a_upd, actor_opt_state = actor_opt.update(a_grad, actor_opt_state, actor_params)
            actor_params = optax.apply_updates(actor_params, a_upd)

            target_critic1 = soft_update(target_critic1, critic1, tau)
            target_critic2 = soft_update(target_critic2, critic2, tau)

        if algo == "gsac":
            g_analytic = analytic_grad_fn(actor_params, init_obs0, init_state0)
            gnorm = float(grad_l2_norm(g_analytic))
            grad_norms.append(gnorm)
            if sg.active:
                g_analytic, w_an, w_mf, info = process_analytic(g_analytic, gnorm, ep, sg)
                mix_weights.append(w_an)
                if info["reason"] != "mix":
                    safeguard_events.append(info)
                g_final = add_grads(g_analytic, a_grad, w_an, w_mf)
            else:
                g_final = add_grads(g_analytic, a_grad, 0.5, 0.5)
            a_upd2, actor_opt_state = actor_opt.update(g_final, actor_opt_state, actor_params)
            actor_params = optax.apply_updates(actor_params, a_upd2)

        ep_reward = float(jnp.sum(rewards))
        episode_rewards.append(ep_reward)
        wall_times.append(time.time() - t_start)

        if (ep + 1) % log_every == 0 or ep == 0:
            extra = f" | grad_norm={grad_norms[-1]:.4g}" if algo == "gsac" else ""
            print(f"[{algo} Re={Re}] ep {ep+1}/{n_episodes} reward={ep_reward:.4f}{extra} "
                  f"t={wall_times[-1]:.1f}s")

    return {
        "algo": algo, "Re": Re,
        "episode_rewards": episode_rewards,
        "grad_norms": grad_norms,
        "wall_times": wall_times,
        "action_rms": action_rms,
        "action_absmax": action_absmax,
        "ext_config": {"safeguards": sg.as_dict()},
        "mix_weights": mix_weights,
        "safeguard_events": safeguard_events,
    }
