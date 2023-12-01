import argparse
import numpy as np
import random

import ray
from ray import tune, air
from ray.tune.registry import register_env
from ray.rllib.algorithms import impala, ppo, sac, a2c, a3c, dqn

import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper



class MultiTaskMiniGridEnv(gym.Env):
    def __init__(self, env_names, seed=None):
        self.envs = [FlatObsWrapper(gym.make(env_name)) for env_name in env_names]
        self.current_env = self.envs[0]
        self.action_space = self.current_env.action_space
        self.observation_space = self.current_env.observation_space
        self.turn = 0


    def reset(self, *, seed=None, options=None):
        self.current_env = self.envs[self.turn] 
        self.turn = (self.turn + 1) % len(self.envs) 
        return self.current_env.reset()

    def step(self, action):
        return self.current_env.step(action)


def main(args):        
    env_names = ['MiniGrid-Empty-8x8-v0', 'MiniGrid-DoorKey-5x5-v0', 'MiniGrid-FourRooms-v0']
    baselines = {
        "IMPALA": impala.ImpalaConfig,
        "PPO": ppo.PPOConfig,
        "SAC": sac.SACConfig,
        "A2C": a2c.A2CConfig,
        "A3C": a3c.A3CConfig,
        "DQN": dqn.DQNConfig
    }
    
    ray.init() # initializing ray
    
    # setting random seeds
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    if args.mode == "SingleTask":
        env_names = [args.env]
    register_env(args.mode, lambda _: MultiTaskMiniGridEnv(env_names, seed=args.seed))

    config = baselines[args.algorithm]() \
        .environment(env=args.mode) \
        .resources(num_gpus=args.n_gpus) \
        .rollouts(num_rollout_workers=args.n_workers) \
        .framework("torch") \
        .training(lr=tune.grid_search(args.lr), grad_clip=20.0)
        
    tune.Tuner(
        args.algorithm,
        run_config=air.RunConfig(stop={"timesteps_total": 1e6}),
        param_space=config.to_dict(),
    ).fit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Expert-Generalist Dilemma in Reinforcement Learning")
    
    parser.add_argument("--mode", type=str, default="MultiTask", help="Single or MultiTask")
    parser.add_argument("--env", type=str, default="MiniGrid-Empty-8x8-v0", help="environment to use (just for the SingleTask mode)")
    parser.add_argument("--algorithm", type=str, default="IMPALA", help="algorithm to use: options[IMPALA, PPO, SAC, A2C, A3C, DQN]")
    parser.add_argument("--train_iters", type=int, default=20, help="number of training iterations")
    parser.add_argument('--lr', metavar='N', type=float, nargs='+', default=[0.0001], help='a float for the learning rate')
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    
    parser.add_argument("--n_gpus", type=int, default=1, help="number of gpus")
    parser.add_argument("--n_workers", type=int, default=15, help="number of workers")
    
    args = parser.parse_args()
    
    main(args)