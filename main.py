import os
import csv
import random
import argparse
import numpy as np
from typing import Dict
from datetime import datetime


import ray
from ray import tune, air
from ray.tune.experiment.trial import Trial
from ray.tune.registry import register_env
from ray.tune.utils import flatten_dict
from ray.tune.logger import CSVLoggerCallback
from ray.rllib.algorithms import impala, ppo, sac, a2c, a3c, dqn

from minigrid.wrappers import FlatObsWrapper

from envs import BaseEnv, all_maps


@ray.remote(num_cpus=4)
class SharedList:
    def __init__(self):
        self.num_envs = len(all_maps)
        self.rewards = [0] * self.num_envs
        self.eposides = [0] * self.num_envs
        self.time_steps = [0] * self.num_envs
        self.total_time_steps = [0] * self.num_envs

    def rewards_add_item(self, index, item):
        self.rewards[index] += item

    def rewards_get_list(self):
        return self.rewards
    
    def eposides_increment(self, index):
        self.eposides[index] += 1
        
    def eposides_get_list(self):
        return self.eposides
    
    def time_steps_increment(self, index):
        self.time_steps[index] += 1
        self.total_time_steps[index] += 1
        
    def time_steps_get_list(self):
        return self.time_steps
    
    def total_time_steps_get_list(self):
        return self.total_time_steps
    
    def reset(self):
        for i in range(self.num_envs):
            self.rewards[i] = 0
            self.eposides[i] = 0
            self.time_steps[i] = 0
        

class CustomLoggerCallback(CSVLoggerCallback):
    def __init__(self, shared_list_actor):
        super().__init__()
        self.shared_list_actor = shared_list_actor
        
    def log_trial_result(self, iteration: int, trial: "Trial", result: Dict):
        rewards = ray.get(self.shared_list_actor.rewards_get_list.remote())
        eposides = ray.get(self.shared_list_actor.eposides_get_list.remote())
        time_steps = ray.get(self.shared_list_actor.time_steps_get_list.remote())
        total_time_steps = ray.get(self.shared_list_actor.total_time_steps_get_list.remote())
        
        new_cols = []
        for i in range(len(all_maps)):
            if eposides[i] != 0:
                col_mean_reward = f'env{i}_mean_reward'
                col_mean_length = f'env{i}_mean_episode_length'
                col_total_time_steps = f'env{i}_tts'
                result[col_mean_reward] = rewards[i]/eposides[i]
                result[col_mean_length] = time_steps[i]/eposides[i]
                result[col_total_time_steps] = total_time_steps[i]
                new_cols.append(col_mean_reward)
                new_cols.append(col_mean_length)
                new_cols.append(col_total_time_steps)
            
        if trial not in self._trial_files:
            self._setup_trial(trial)

        tmp = result.copy()
        tmp.pop("config", None)
        result = flatten_dict(tmp, delimiter="/")

        if not self._trial_csv[trial]:
            self._trial_csv[trial] = csv.DictWriter(
                self._trial_files[trial], result.keys()
            )
            if not self._trial_continue[trial]:
                self._trial_csv[trial].writeheader()
        
        fieldnames = list(self._trial_csv[trial].fieldnames) + new_cols
        self._trial_csv[trial].writerow(
            {k: v for k, v in result.items() if k in fieldnames}
        )
        self._trial_files[trial].flush()
        self.shared_list_actor.reset.remote()
        
        
        

def main(args):        
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
        
    
    shared_list_actor = SharedList.remote()
    if args.map_i == -1: # MultiTask
        mode = "MultiTask"
        register_env(mode, lambda config: FlatObsWrapper(BaseEnv(config.worker_index % len(all_maps), shared_list_actor)))
    else:
        mode = "SingleTask"
        register_env(mode, lambda _: FlatObsWrapper(BaseEnv(args.map_i, shared_list_actor)))
    
    

    config = baselines[args.algorithm]() \
        .environment(env=mode) \
        .resources(num_gpus=args.n_gpus) \
        .rollouts(num_rollout_workers=args.n_workers) \
        .framework("torch") \
        .training(lr=tune.grid_search(args.lr))
        
    # Ensure the log directory exists
    log_dir = os.path.abspath(args.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    
    exp_name = f'{args.algorithm}_SingleTask_{args.map_i}' if mode == "SingleTask" else f'{args.algorithm}_MultiTask'
    exp_name += f'_{datetime.now().strftime("%m-%d %H:%M")}'
    
    tune.Tuner(
        args.algorithm,
        run_config=air.RunConfig(
            callbacks=[CustomLoggerCallback(shared_list_actor)],
            stop={"timesteps_total": args.total_timesteps}, # stops when total timesteps is reached
            name=exp_name, # name of the experiment
            storage_path=log_dir # path to store results
            ),
        param_space=config.to_dict(),
    ).fit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Expert-Generalist Dilemma in Reinforcement Learning")
    
    parser.add_argument("--map_i", type=int, default=1, help="map to use. Options are [0, 1, 2, 3]. -1 for all maps")
    parser.add_argument("--algorithm", type=str, default="IMPALA", help="algorithm to use: options[IMPALA, PPO, SAC, A2C, A3C, DQN]")
    
    # training specs
    parser.add_argument("--train_iters", type=int, default=20, help="number of training iterations")
    parser.add_argument("--total_timesteps", type=int, default=1e6, help="total interaction with the environment")
    parser.add_argument('--lr', metavar='N', type=float, nargs='+', default=[0.0001], help='a float for the learning rate')
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    
    # hardware specs
    parser.add_argument("--n_gpus", type=int, default=1, help="number of gpus")
    parser.add_argument("--n_workers", type=int, default=11, help="number of workers")
    
    # logging
    parser.add_argument("--log_dir", type=str, default="./results", help="path to store results")
    
    args = parser.parse_args()
    
    main(args)