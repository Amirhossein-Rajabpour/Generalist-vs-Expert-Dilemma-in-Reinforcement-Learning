import os
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


def get_subdirectories(dir):
    return [entry.name for entry in Path(dir).iterdir() if entry.is_dir()]

def plot_metric(dir, metric):
    for subdir in get_subdirectories(dir):
        subdir_path = os.path.join(dir, subdir)
        csv_file = os.path.join(subdir_path, get_subdirectories(subdir_path)[0], 'progress.csv')
            
        if os.path.isfile(csv_file):
            df = pd.read_csv(csv_file)
                
            if metric in df.columns:
                plt.plot(df[metric], label=subdir)

    plt.xlabel('training iteration')
    plt.ylabel(metric)
    plt.title(f'{metric} over different runs')
    plt.legend()
    plt.show()