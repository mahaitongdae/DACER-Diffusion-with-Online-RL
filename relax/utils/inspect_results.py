from typing import Dict

import relax
from pathlib import Path
import re
import csv
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
sns.set_context(font_scale=1.2)


def plot_mean(patterns_dict: Dict, env_name, path=None, fig_name = None,
              max_steps=None, title=None, ylim=None):
    plt.figure(figsize=(4, 3))
    if path is None:
        package_path = Path(relax.__file__)
        logdir = package_path.parent.parent / 'logs' / env_name
    else:
        package_path = Path(path)
        logdir = package_path / env_name
    dfs = []
    for alg, pattern in patterns_dict.items():
        matching_dir = [s for s in logdir.iterdir() if re.match(pattern, str(s))]
        for dir in matching_dir:
            csv_path = dir / 'log.csv'
            df = pd.read_csv(str(csv_path))
            df.loc[:, ('seed')] = str(dir).split('_s')[1].split('_')[0]
            df.loc[:, ('Algorithm')] = alg
            dfs.append(df)
            df.rename(columns={'avg_ret': 'Return'}, inplace=True)
            df['Iterations'] = df['step'] / 5

    total_df = pd.concat(dfs, ignore_index=True)
    if max_steps is not None:
        total_df = total_df[total_df['step'] < max_steps]
    sns.lineplot(data=total_df, x='Iterations', y='Return', hue='Algorithm')
    if title is not None:
        plt.title(title)
    else:
        plt.title(env_name)
    if ylim is not None:
        plt.ylim(ylim)
    if fig_name is not None:
        plt.tight_layout()
        plt.savefig(fig_name)
    else:
        plt.show()
    


def load_results(pattern, env_name, show_df=False,
              max_steps=None, display_digits=0, path=None):
    if path is None:
        package_path = Path(relax.__file__)
        logdir = package_path.parent.parent / 'logs' / env_name
    else:
        package_path = Path(path)
        logdir = package_path / env_name
    
    matching_dir = [s for s in logdir.iterdir() if re.match(pattern, str(s))]
    dfs = []
    for dir in matching_dir:
        csv_path = dir / 'log.csv'
        df = pd.read_csv(str(csv_path))
        if max_steps is not None:
            df = df[df['step'] < max_steps]
        sliced_df = df.loc[df['avg_ret'].idxmax()]
        sliced_df.loc['seed'] = str(dir).split('_s')[1].split('_')[0]
        dfs.append(sliced_df)
    total_df = pd.concat(dfs, ignore_index=True, axis=1).T
    if show_df:
        print(total_df.to_markdown())
    print(f"${total_df['avg_ret'].mean():.{display_digits}f} \pm {total_df['avg_ret'].std():.{display_digits}f}$")
    return total_df
