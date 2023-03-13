import gymnasium as gym
import numpy as np


class NormalizeObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=np.array([-1, -1, -1]),
            high=np.array(
                [1, 1, 1],
            ),
        )

    def observation(self, obs):
        return (obs - self.env.observation_space.low) / (
            self.env.observation_space.high - self.env.observation_space.low
        )
