import os
import random
import argparse
import numpy as np
from datetime import datetime

import ray
from ray import tune, air
from ray.tune.registry import register_env
from ray.rllib.algorithms import impala, ppo, sac, a2c, a3c, dqn, appo
from ray.rllib.utils.torch_utils import set_torch_seed

from minigrid.wrappers import FlatObsWrapper

from envs import BaseEnv, all_maps
from utils.shared_list import SharedList
from utils.callbacks import CustomLoggerCallback
        
        
        
def main(args):        
    baselines = {
        "IMPALA": impala.ImpalaConfig,
        "PPO": ppo.PPOConfig,
        "SAC": sac.SACConfig,
        "A2C": a2c.A2CConfig,
        "A3C": a3c.A3CConfig,
        "DQN": dqn.DQNConfig,
        "APPO": appo.APPOConfig
    }
    
    ray.init() # initializing ray
    
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
        .training(lr=tune.grid_search(args.lr)) \
            
    # setting random seeds
    np.random.seed(args.seed)
    random.seed(args.seed)
    set_torch_seed(args.seed)
    config['seed'] = args.seed
        
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
    
    parser.add_argument("--map_i", type=int, default=-1, help="map to use. Options are [0, 1, 2, 3]. -1 for all maps")
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