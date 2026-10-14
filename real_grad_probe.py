"""Controlled probe on the REAL hydrogym.jax env: fixed untrained policy,
analytic gradient norm sampled from several points near each Re's reset
state (perturbed slightly), isolating dynamics-driven gradient sensitivity
from PPO training-progress confounds (mirrors grad_sensitivity_probe.py from
the custom fallback sim)."""
import json
import time
import jax
import jax.numpy as jnp
import numpy as np

import real_rl
from fast_env import FastKolmogorovFlow

N_SAMPLES = 15
SHORT_HORIZON = 6


def main():
    key = jax.random.PRNGKey(0)
    key, pkey = jax.random.split(key)

    out = {}
    for Re, label in [(45.0, "low"), (300.0, "high")]:
        env = FastKolmogorovFlow(flow_config={"Re": Re}, dt=0.001, action_time=1.0,
                                  max_episode_steps=SHORT_HORIZON, reward_alpha=1.0)
        params_env = env.default_params
        policy_params = real_rl.init_params(pkey, params_env.obs_dim, params_env.action_dim)

        reset_key = jax.random.PRNGKey(1000)
        init_obs, init_state = env.reset_env(reset_key, params_env)

        _, analytic_loss_fn = real_rl.make_rollout_fns(env, params_env, SHORT_HORIZON, SHORT_HORIZON)
        analytic_grad_fn = jax.jit(jax.grad(analytic_loss_fn))

        t0 = time.time()
        norms = []
        k = key
        for i in range(N_SAMPLES):
            k, pk = jax.random.split(k)
            perturb_omega = init_state.omega_hat * (1.0 + 0.01 * jax.random.normal(pk, init_state.omega_hat.shape))
            state_i = init_state.replace(omega_hat=perturb_omega)
            obs_i = env.get_obs(state_i, params_env)
            g = analytic_grad_fn(policy_params, obs_i, state_i)
            gnorm = float(real_rl.grad_l2_norm(g))
            if not np.isnan(gnorm):
                norms.append(gnorm)
        norms = np.array(norms)
        out[label] = norms.tolist()
        print(f"Re={Re} ({label}): n_valid={len(norms)}/{N_SAMPLES} mean={norms.mean():.4g} "
              f"std={norms.std():.4g} cv={norms.std()/norms.mean():.3f} time={time.time()-t0:.1f}s")

    with open("real_grad_probe.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
