import os

import sys
import time
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

OFF_SCREEN=True
if OFF_SCREEN:
    os.environ["MUJOCO_GL"] = "osmesa"

def evaluate(env, policy_fn, policy_params, num_episodes, video_name=None, seed=0):
    ep_len_list = []
    ep_ret_list = []


    for s in range(num_episodes):
        frames = []
        obs, _ = env.reset(seed=s+seed)
        ep_len = 0
        ep_ret = 0.0
        obses_list = []
        while True:
            act = policy_fn(policy_params, obs)
            obs, reward, terminated, truncated, _ = env.step(act)
            ep_len += 1
            ep_ret += reward
            frame = env.render()
            if video_name is not None:
                resized_frame = cv2.resize(frame, (200, 160), interpolation=cv2.INTER_AREA)
                frames.append(resized_frame)
            if not OFF_SCREEN:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
                cv2.imshow("DM Control Real-Time Render", frame)
                time.sleep(0.05)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
                    break

            obses_list.append(obs)
            if terminated or truncated:
                break

        ep_len_list.append(ep_len)
        ep_ret_list.append(ep_ret)
        cv2.destroyAllWindows()
        if video_name is not None:
            imageio.mimsave(video_name + str(s + 1) + '.gif', frames)
            print(f"Video saved as {video_name}")

    return ep_len_list, ep_ret_list, obses_list

if __name__ == "__main__":
    import time
    policy_root = Path('/n/home05/haitongma/src/DACER-Diffusion-with-Online-RL/logs/Ant-v4/diffv2_2025-01-16_15-27-10_s100_large_scale_run')
    env_name = str(policy_root).split('/')[-2]
    if env_name.startswith('dm_control'):
        from relax.env.dmc.register import register_dm_control_envs
        register_dm_control_envs()
    if env_name.startswith('pusht'):
        from relax.env.pusht.pusht_env import PushTEnv

    master_rng = np.random.default_rng(0)
    env_seed, env_action_seed, policy_seed = map(int, master_rng.integers(0, 2**32 - 1, 3))
    env, _, _ = create_env(env_name, env_seed, env_action_seed, render_mode='rgb_array') #

    policy = PersistFunction.load(policy_root / "deterministic.pkl")
    @jax.jit
    def policy_fn(policy_params, obs):
        return policy(policy_params, obs).clip(-1, 1)

    step = int(1e6)
    policy_path = "policy-1000000-200000.pkl"
    with open(policy_root / policy_path, "rb") as f:
        policy_params = pickle.load(f)

    ep_len_list, ep_ret_list, _ = evaluate(env, policy_fn, policy_params, 1, video_name=str(policy_root / 'visu'), seed=1)
    print(ep_ret_list)
