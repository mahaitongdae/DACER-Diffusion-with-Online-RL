import os
import gymnasium as gym
import sys
import time
from pathlib import Path
import argparse
import pickle
import csv

import numpy as np
import matplotlib.pyplot as plt
import jax
from tensorboardX import SummaryWriter
from tqdm import tqdm

from relax.env import create_env
from relax.utils.persistence import PersistFunction
import imageio
import cv2
import jax.numpy as jnp

from relax.network.sdac import create_sdac_net
from relax.network.sac import create_sac_net
from relax.algorithm.sdac import SDAC
from relax.algorithm.sac import SAC

from relax.trainer.evaluator import LoggerWithSuccessRate

if __name__ == "__main__":

    # policy_root = Path(
    #     '/home/naliseas-workstation/Documents/haitong/DACER-Diffusion-with-Online-RL/logs/pushtcurriculum-v1/sac_2025-04-24_15-26-11_s100_test_use_atp1')

    # policy_root = Path(
    #   '/home/naliseas-workstation/Documents/haitong/DACER-Diffusion-with-Online-RL/logs/pushtcurriculum-v1/sdac_2025-04-23_17-55-35_s100_test_use_atp1')

    policy_root = Path(
        '/home/naliseas-workstation/Documents/haitong/DACER-Diffusion-with-Online-RL/logs/pushtcurriculum-v2/sac_2025-04-30_20-19-36_s100_ablation_rela_obs')

    env_name = str(policy_root).split('/')[-2]
    if env_name.startswith('dm_control'):
        from relax.env.dmc.register import register_dm_control_envs
        register_dm_control_envs()
    if env_name.startswith('pusht'):
        from relax.env.pusht.pusht_env import PushTEnv
        # from relax.env.pusht_orig.pusht_env import PushTEnv

    master_rng = np.random.default_rng(0)
    env_seed, env_action_seed, policy_seed = map(
        int, master_rng.integers(0, 2**32 - 1, 3))


    logger = LoggerWithSuccessRate(policy_root, 'log2.csv')


    policy = PersistFunction.load(policy_root / "deterministic.pkl")

    @jax.jit
    def policy_fn(params, obs):
        return policy(params, obs).clip(-1, 1)
    
    for i in range(10):
        step = int(400000 * (i+1))
        policy_param_name = f"policy-{step}-{int(80000*(i+1))}.pkl"
        with open(policy_root / policy_param_name, "rb") as f:
            policy_params = pickle.load(f)
            

        env = gym.make('pushtcurriculum-v2')
        suc = 0
        suc = []
        for i in tqdm(range(100)):
            done = False
            obs, _ = env.reset(options={'curriculum_level': 1.0})
            state = np.concatenate(
                [env.unwrapped.agent.position, env.unwrapped.block.position, [env.unwrapped.block.angle]])
            
            while not done:
                action = policy_fn(policy_params, obs)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                # rewards.append(reward)
                state = np.concatenate(
                    [env.unwrapped.agent.position, env.unwrapped.block.position, [env.unwrapped.block.angle]])
            suc.append(terminated)            

        logger.log(step, 0.0, 0.0, np.mean(suc), np.std(suc))

# tag = 'sdac' if 'sdac' in str(policy_root) else 'sac'
# with open(f'data/sucrate-{tag}.pkl', 'wb') as f:
#     pickle.dump(res, f)