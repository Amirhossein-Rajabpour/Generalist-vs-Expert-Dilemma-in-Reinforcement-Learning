from ray import tune, air
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.impala import ImpalaConfig
import gymnasium as gym
from ray.rllib.env.vector_env import VectorEnv
import numpy as np
from minigrid.wrappers import ImgObsWrapper, FlatObsWrapper, FullyObsWrapper



def main():

    # Register the desired MiniGrid environment
    register_env('MiniGrid-FlatObs-v0', lambda config: FlatObsWrapper(gym.make('MiniGrid-Empty-8x8-v0')))


    config = ImpalaConfig()

    # Set the config object's env.
    config = config.environment(env="MiniGrid-FlatObs-v0")

    # Update the config object.
    config = config.training(
        lr=tune.grid_search([0.0001, ]), grad_clip=20.0
    )
    config = config.resources(num_gpus=0)
    config = config.rollouts(num_rollout_workers=3)
    
    
    # Use to_dict() to get the old-style python config dict when running with tune.
    tune.Tuner(
        "IMPALA",
        run_config=air.RunConfig(stop={"training_iteration": 3}),
        param_space=config.to_dict(),
    ).fit()


if __name__ == "__main__":
    main()