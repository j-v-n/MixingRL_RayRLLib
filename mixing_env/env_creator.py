from mixing_env.mixing_env import MixEnv
from mixing_env.wrappers import NormalizeObs

from gymnasium.wrappers.normalize import NormalizeReward


def mixing_env_creator(config):
    obs_filter = config.pop("obs_filter", None)
    reward_filter = config.pop("reward_filter", None)
    env = MixEnv()
    if obs_filter is not None:
        if obs_filter == "normalize_obs":
            env = NormalizeObs(env)

    if reward_filter is not None:
        if reward_filter == "normalize_rewards":
            env = NormalizeReward(env)

    return env
