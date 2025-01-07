import relax
from pathlib import Path
import re
import csv
import pandas as pd

def main():
    package_path = Path(relax.__file__)
    logdir = package_path.parent.parent / 'logs' / 'HalfCheetah-v4'
    # pattern = r".*diffv2.*noise_scale_0\.0\d$"
    # pattern = r".*diffv2.*noise_scale_0\.09"
    pattern = r".*qsm.*01-07.*qsm_lr_schedule$"
    matching_dir = [s for s in logdir.iterdir() if re.match(pattern, str(s))]
    dfs = []
    for dir in matching_dir:
        csv_path = dir / 'log.csv'
        df = pd.read_csv(str(csv_path))
        sliced_df = df.loc[df['avg_ret'].idxmax()]
        sliced_df.loc['seed'] = str(dir).split('_s')[1].split('_')[0]
        dfs.append(sliced_df)
    total_df = pd.concat(dfs, ignore_index=True, axis=1).T
    print(total_df.to_markdown())
    print(total_df['avg_ret'].mean())



main()
