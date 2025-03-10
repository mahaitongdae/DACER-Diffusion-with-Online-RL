import os

import sys
from pathlib import Path
import argparse
import pickle
import csv

import numpy as np
import jax
from tensorboardX import SummaryWriter

from relax.env import create_env
from relax.utils.persistence import PersistFunction
import imageio
import cv2

def evaluate(env, policy_fn, policy_params, num_episodes):
    ep_len_list = []
    ep_ret_list = []
    frames = []
    
    for s in range(num_episodes):
        obs, _ = env.reset(seed=s)
        ep_len = 0
        ep_ret = 0.0
        obses_list = []
        while True:
            act = policy_fn(policy_params, obs)
            obs, reward, terminated, truncated, _ = env.step(act)
            ep_len += 1
            ep_ret += reward
            frame = env.render()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
            cv2.imshow("DM Control Real-Time Render", frame)
    
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
                break
            
            obses_list.append(obs)
            if terminated or truncated:
                break
            
        ep_len_list.append(ep_len)
        ep_ret_list.append(ep_ret)
        cv2.destroyAllWindows()
        
    return ep_len_list, ep_ret_list, obses_list

if __name__ == "__main__":
    policy_root = Path('/home/haitong/PycharmProjects/DACER-Diffusion-with-Online-RL/logs/dm_control_walker_walk-v0/sdac_2025-02-19_22-15-17_s100_test_use_atp1')
    env_name = str(policy_root).split('/')[-2]
    from relax.env.dmc.register import register_dm_control_envs
    register_dm_control_envs()

    master_rng = np.random.default_rng(0)
    env_seed, env_action_seed, policy_seed = map(int, master_rng.integers(0, 2**32 - 1, 3))
    env, _, _ = create_env(env_name, env_seed, env_action_seed)

    policy = PersistFunction.load(policy_root / "deterministic.pkl")
    @jax.jit
    def policy_fn(policy_params, obs):
        return policy(policy_params, obs).clip(-1, 1)

    step = int(1e6)
    policy_path = "policy-500000-100000.pkl"
    with open(policy_root / policy_path, "rb") as f:
        policy_params = pickle.load(f)

    ep_len_list, ep_ret_list, _ = evaluate(env, policy_fn, policy_params, 1)
    print(ep_ret_list)
