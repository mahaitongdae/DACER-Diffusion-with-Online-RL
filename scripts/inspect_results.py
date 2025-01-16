from typing import Dict

import relax
from pathlib import Path
import re
import csv
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
sns.set_context(font_scale=1.2)

def plot_mean(patterns_dict: Dict, env_name, fig_name = None):
    fig = plt.figure(figsize=(4, 3))
    package_path = Path(relax.__file__)
    logdir = package_path.parent.parent / 'logs' / env_name
    dfs = []
    for alg, pattern in patterns_dict.items():
        matching_dir = [s for s in logdir.iterdir() if re.match(pattern, str(s))]
        for dir in matching_dir:
            csv_path = dir / 'log.csv'
            df = pd.read_csv(str(csv_path))
            df = df[df['step'] <= 1000000]
            df.loc[:, ('seed')] = str(dir).split('_s')[1].split('_')[0]
            df.loc[:, ('Algorithm')] = alg
            df['Iteration'] = df['step'] / 5
            dfs.append(df)

    total_df = pd.concat(dfs, ignore_index=True)
    total_df.rename(columns={'avg_ret': 'Returns'}, inplace=True)
    sns.lineplot(data=total_df, x='Iteration', y='Returns', hue='Algorithm')
    if fig_name is not None:
        plt.savefig(fig_name)
    else:
        plt.show()
    


def load_best_results(pattern, env_name, show_df = False):
    package_path = Path(relax.__file__)
    logdir = package_path.parent.parent / 'logs' / env_name
    # pattern = r".*diffv2.*noise_scale_0\.0\d$"
    # pattern = r".*diffv2.*noise_scale_0\.09"
    # pattern = r".*qsm.*01-07.*qsm_lr_schedule$"
    
    matching_dir = [s for s in logdir.iterdir() if re.match(pattern, str(s))]
    dfs = []
    for dir in matching_dir:
        csv_path = dir / 'log.csv'
        df = pd.read_csv(str(csv_path))
        df = df[df['step'] <= 1000000]
        sliced_df = df.loc[df['avg_ret'].idxmax()]
        sliced_df.loc['seed'] = str(dir).split('_s')[1].split('_')[0]
        dfs.append(sliced_df)
    total_df = pd.concat(dfs, ignore_index=True, axis=1).T
    if show_df:
        print(total_df.to_markdown())
    print(f"${total_df['avg_ret'].mean():.2f} \pm {total_df['avg_ret'].std():.2f} $", )
    return total_df

if __name__ == "__main__":
    patterns_dict = {
            'ours': r".*diffv2.*01-.*smaller_par$",
            'QSM': r".*qsm.*01-0.*new_seed_set$",
            'DIPO': r".*dipo.*01-09.*new_seed_set$",
            'DACER': r".*dacer.*01-09.*new_seed_set$",
            'SAC': r".*sac.*01.*large_scale_run$"
        }
    for key, value in patterns_dict.items():
        _ = load_best_results(value, env_name, show_df=False)
    plot_mean(patterns_dict, 'HalfCheetah-v4')
