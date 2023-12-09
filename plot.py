import os
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from envs import BaseEnv
from utils.shared_list import SharedList


def get_subdirectories(dir):
    return [entry.name for entry in Path(dir).iterdir() if entry.is_dir()]

def plot_metric(dir, metric, map_i, y_axis='time steps'):
    '''
    mean_reward and mean_episode_length have only been implemented so far.
    y_axis can be either 'time steps' or 'training steps'
    
    exmaple: plot_metric('./results', 'mean_reward', 1, "training steps")
    '''
    for subdir in get_subdirectories(dir):
        subdir_path = os.path.join(dir, subdir)
        csv_file = os.path.join(subdir_path, get_subdirectories(subdir_path)[0], 'progress.csv')
            
        if os.path.isfile(csv_file):
            df = pd.read_csv(csv_file)
                
            col_reward = f'env{map_i}_{metric}'
            col_time_steps = f'env{map_i}_tts'
            if col_reward in df.columns:
                if y_axis == 'time steps':
                    plt.plot(df[col_time_steps], df[col_reward], label='_'.join(subdir.split('_')[:2]))
                elif y_axis == 'training steps':
                    plt.plot(df[col_reward], label='_'.join(subdir.split('_')[:2]))

    plt.xlabel(y_axis)
    plt.ylabel(metric)
    plt.title(f'env{map_i}')
    plt.legend()
    plt.show()
    
def plot_map(map_index):
    env = BaseEnv(map_index=map_index, shared_list_actor=SharedList.remote(), render_mode="rgb_array")

    # plot initial state
    env.reset(seed=42, options=None)
    img = env.render()
    plt.title(f'env{map_index}')
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)