import ray
from ray import tune
from ray.tune.registry import register_env
from mixing_env.env_creator import mixing_env_creator

register_env("mixing_environment", mixing_env_creator)


if __name__ == "__main__":
    ray.init()

    tune.run(
        "PPO",
        config={
            "env": "mixing_environment",
            "env_config": {
                "obs_filter": "normalize_obs",
                # search spaces
                "reward_filter": tune.grid_search(["normalize_rewards", None]),
            },
            # different wrappers for evaluation
            # no reward wrapper
            # if we are using search space use a conditional search space: tune.sample_from()
            "evaluation_config": {"obs_filter": "normalize_obs", "reward_filter": None},
        },
        local_dir="./logfiles/",
    )
