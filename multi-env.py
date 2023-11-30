from ray import tune, air
from ray.tune.registry import register_env
from ray.rllib.algorithms.impala import ImpalaConfig
import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper


# Define a function that returns different environments based on the worker index
def env_creator(config):
    worker_index = config.worker_index
    if worker_index % 2 == 0:
        return FlatObsWrapper(gym.make('MiniGrid-DoorKey-5x5-v0'))
    elif worker_index % 2 == 1:
        return FlatObsWrapper(gym.make('MiniGrid-Empty-5x5-v0'))
    else:
        # Default environment or you can raise an exception if you prefer
        return FlatObsWrapper(gym.make('MiniGrid-Empty-8x8-v0'))


def main():
    # Register the custom environment creator
    register_env('custom_env', env_creator)

    config = ImpalaConfig()

    # Set the config object's env to the custom environment
    config = config.environment(env="custom_env")

    # Update the config object.
    config = config.training(
        lr=tune.grid_search([0.0001, ]), grad_clip=20.0
    )
    config = config.resources(num_gpus=0)

    # Specify the number of parallel workers (environments)
    config = config.rollouts(num_rollout_workers=4)  # Adjust the number of workers as needed

    # Use to_dict() to get the old-style python config dict when running with tune.
    tune.Tuner(
        "IMPALA",
        run_config=air.RunConfig(stop={"training_iteration": 3}),
        param_space=config.to_dict(),
    ).fit()


if __name__ == "__main__":
    main()
