import os
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


def get_subdirectories(dir):
    return [entry.name for entry in Path(dir).iterdir() if entry.is_dir()]

def plot_metric(dir, metric, map_i):
    for subdir in get_subdirectories(dir):
        subdir_path = os.path.join(dir, subdir)
        csv_file = os.path.join(subdir_path, get_subdirectories(subdir_path)[0], 'progress.csv')
            
        if os.path.isfile(csv_file):
            df = pd.read_csv(csv_file)
                
            col_name = f'env{map_i}_{metric}'
            if col_name in df.columns:
                plt.plot(df[col_name], label='_'.join(subdir.split('_')[:2]))

    plt.xlabel('training iteration')
    plt.ylabel(metric)
    plt.title(f'map{map_i}')
    plt.legend()
    plt.show()