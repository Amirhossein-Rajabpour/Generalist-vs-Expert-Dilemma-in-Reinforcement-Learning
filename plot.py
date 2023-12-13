import os
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from envs import BaseEnv
from utils.shared_list import SharedList

NUM_SEEDS = 5


def get_subdirectories(dir):
    return [entry.name for entry in Path(dir).iterdir() if entry.is_dir()]


def plot_metric(dir, metric, map_i, y_axis='interactions'):
    '''
    mean_reward and mean_episode_length have only been implemented so far.
    y_axis can be either 'interactions' or 'training steps'
    
    exmaple: plot_metric('./results', 'mean_reward', 1, "training steps")
    '''
    for subdir in get_subdirectories(dir):
        subdir_path = os.path.join(dir, subdir)
        csv_file = os.path.join(subdir_path, get_subdirectories(subdir_path)[0], 'progress.csv')
        singleTask = True if subdir_path.split('/')[-1].split('_')[1] == 'SingleTask' else False
        color = 'red' if singleTask else 'green'

        if os.path.isfile(csv_file):
            df = pd.read_csv(csv_file)

            col_reward = f'env{map_i}_{metric}'
            col_time_steps = f'env{map_i}_tts'
            if col_reward in df.columns:
                if y_axis == 'interactions':
                    plt.plot(df[col_time_steps], df[col_reward], label='_'.join(subdir.split('_')[:2]), color=color)
                elif y_axis == 'training steps':
                    plt.plot(df[col_reward], label='_'.join(subdir.split('_')[:2]), color=color)

            display(df)
            print("=====================================================")

    plt.xlabel(y_axis)
    plt.ylabel(metric)
    plt.title(f'env{map_i}')
    plt.legend()
    plt.show()


def plot_algorithms(dir, metric, map_i, y_axis='interactions'):
    dfs = pd.DataFrame()  # list to store dataframes
    df_PPO = pd.DataFrame({})
    df_IMPALA = pd.DataFrame({})
    df_APPO = pd.DataFrame({})

    for subdir in get_subdirectories(dir):
        subdir_path = os.path.join(dir, subdir)
        csv_file = os.path.join(subdir_path, get_subdirectories(subdir_path)[0], 'progress.csv')
        current_algo = subdir_path.split('/')[-1].split('_')[0]

        if os.path.isfile(csv_file):
            df = pd.read_csv(csv_file)

            col_reward = f'env{map_i}_{metric}'
            if col_reward in df.columns:
                if current_algo == 'PPO':
                    df_PPO = pd.concat([df_PPO, df[[col_reward]]], axis=1)
                elif current_algo == 'APPO':
                    df_APPO = pd.concat([df_APPO, df[[col_reward]]], axis=1)
                elif current_algo == 'IMPALA':
                    df_IMPALA = pd.concat([df_IMPALA, df[[col_reward]]], axis=1)

            if df_PPO.columns.size == NUM_SEEDS:
                df_PPO['PPO'] = df_PPO.mean(axis=1)
                dfs = pd.concat([dfs, df_PPO[['PPO']]], axis=1)
            elif df_APPO.columns.size == NUM_SEEDS:
                df_APPO['APPO'] = df_APPO.mean(axis=1)
                dfs = pd.concat([dfs, df_APPO[['APPO']]], axis=1)
            elif df_IMPALA.columns.size == NUM_SEEDS:
                df_IMPALA['IMPALA'] = df_IMPALA.mean(axis=1)
                dfs = pd.concat([dfs, df_IMPALA[['IMPALA']]], axis=1)

    display(dfs)

    plt.plot(dfs.index, dfs)
    plt.legend(list(dfs.columns))
    plt.xlabel(y_axis)
    plt.ylabel(metric)
    plt.title(f'env{map_i}')
    plt.show()


def plot_interactions(dir, metric, map_i, y_axis='interactions'):
    '''
    mean_reward and mean_episode_length have only been implemented so far.
    y_axis can be either 'interactions' or 'training steps'

    exmaple: plot_metric('./results', 'mean_reward', 1, "training steps")
    '''

    dfs = pd.DataFrame()  # list to store dataframes
    df_Multi = pd.DataFrame({})
    df_Single = pd.DataFrame({})

    for subdir in get_subdirectories(dir):
        subdir_path = os.path.join(dir, subdir)
        csv_file = os.path.join(subdir_path, get_subdirectories(subdir_path)[0], 'progress.csv')
        current_multi = False if subdir_path.split('/')[-1].split('_')[1] == 'SingleTask' else True

        if os.path.isfile(csv_file):
            df = pd.read_csv(csv_file)

            col_reward = f'env{map_i}_{metric}'
            if col_reward in df.columns:
                if current_multi:
                    df_Multi = pd.concat([df_Multi, df[[col_reward]]], axis=1)
                else:
                    df_Single = pd.concat([df_Single, df[[col_reward]]], axis=1)

            print("========================df before===========================")
            display(dfs)

            if df_Multi.columns.size == NUM_SEEDS:
                df_Multi['multi'] = df_Multi.mean(axis=1)
                dfs = pd.concat([dfs, df_Multi[['multi']]], axis=1)
            elif df_Single.columns.size == NUM_SEEDS:
                df_Single['single'] = df_Single.mean(axis=1)
                dfs = pd.concat([dfs, df_Single[['single']]], axis=1)

        # concatenate dataframes and calculate mean
    print("========================df===========================")
    display(dfs)

    # df_concat = pd.concat(dfs)
    # df_mean = df_concat.groupby(df_concat[col_time_steps]).mean()
    # dfs['avg'] = dfs.mean(axis=1)
    # df_average = dfs[['avg']]

    print("=========================df mean============================")
    # display(df_average)

    # plot the average dataframe
    # plt.plot(df_mean.index, df_mean[col_reward], color=color)
    plt.plot(dfs.index, dfs)
    plt.legend(list(dfs.columns))
    plt.xlabel(y_axis)
    plt.ylabel(metric)
    plt.title(f'env{map_i}')
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
