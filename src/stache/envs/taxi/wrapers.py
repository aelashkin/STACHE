import numpy as np
import gymnasium as gym
from gymnasium import spaces

# 1. One‑hot wrapper for Discrete(500) → Box(500)
class OneHotObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Discrete), \
            "OneHotObs only supports Discrete spaces"
        n = env.observation_space.n
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(n,), dtype=np.float32
        )

    def observation(self, obs):
        vec = np.zeros(self.observation_space.shape, dtype=np.float32)
        vec[obs] = 1.0
        return vec