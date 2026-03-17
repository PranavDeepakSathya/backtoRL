import numpy as np
import gymnasium
import pufferlib

from turan_env_c import CEnv, CHECKER_C3, CHECKER_C4, CHECKER_K4


class TuranPufferEnv(pufferlib.PufferEnv):
    """
    PufferLib 3.0 native wrapper around the vectorised C environment.

    num_agents = num_envs (each parallel env is one "agent" from PufferLib's
    perspective).  The C env already handles auto-reset, so PufferEnv.done
    stays False and we never need to call reset() externally mid-training.
    """

    def __init__(self, n=20, num_envs=1024, checker_id=CHECKER_C3, buf=None, seed=0):
        na = n * (n - 1) // 2

        # Must be set BEFORE super().__init__()
        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=1, shape=(na,), dtype=np.uint8)
        self.single_action_space = gymnasium.spaces.Discrete(na)
        self.num_agents = num_envs

        super().__init__(buf)

        self._env = CEnv(n=n, num_envs=num_envs, checker_id=checker_id)
        self.n = n
        self._ep_return = np.zeros(num_envs, dtype=np.float32)

    def reset(self, seed=None):
        obs = self._env.reset()          # (num_envs, na)  np.bool_
        self.observations[:] = obs
        self._ep_return[:] = 0.0
        return self.observations, {}

    def step(self, actions):
        obs, rewards, dones = self._env.step(actions.astype(np.int32))
        self.observations[:] = obs
        self.rewards[:]      = rewards
        self.terminals[:]    = dones
        self.truncations[:]  = False

        self._ep_return += rewards
        info = {}
        if dones.any():
            # PufferLib reads list values with .extend(), scalar with .append()
            info['episode_return'] = self._ep_return[dones].tolist()
            self._ep_return[dones] = 0.0

        return self.observations, self.rewards, self.terminals, self.truncations, info

    def close(self):
        self._env.close()
