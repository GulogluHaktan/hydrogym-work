"""
Thin subclass of the real hydrogym.jax KolmogorovFlow env that overrides the
physical time held per RL step (`action_time`) and substep size (`dt`) down
to POC-scale values. The base class's `_rollout` reads dt/action_time from
`self.default_params` regardless of what params are passed to step_env, so
this has to be done via subclassing rather than env_config.
"""
import jax.numpy as jnp
from hydrogym.jax.envs.kolmogorov import KolmogorovFlow, KolmogorovFlowParams


class FastKolmogorovFlow(KolmogorovFlow):
    def __init__(self, env_config=None, flow_config=None, dt=0.005, action_time=1.0,
                 max_episode_steps=20, reward_alpha=1.0):
        self._fast_dt = dt
        self._fast_action_time = action_time
        self._fast_max_steps = max_episode_steps
        self._fast_reward_alpha = reward_alpha
        super().__init__(env_config=env_config, flow_config=flow_config)

    @property
    def default_params(self) -> KolmogorovFlowParams:
        return KolmogorovFlowParams(
            action_dim=4,
            obs_dim=self.flow.obs_size ** 2,
            dt=self._fast_dt,
            action_time=self._fast_action_time,
            save_time=1,
            max_episode_steps=self._fast_max_steps,
            reward_alpha=self._fast_reward_alpha,
        )
