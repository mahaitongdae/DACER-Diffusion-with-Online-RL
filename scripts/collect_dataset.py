import numpy as np

from pathlib import Path
import pickle

import numpy as np
import jax

from relax.env import create_env
from relax.utils.persistence import PersistFunction
import relax
import os
from tqdm import tqdm

def collect_data(env, policy_fn, policy_params, num_episodes, dataset_name):
    ep_len_list = []
    ep_ret_list = []

    states = []
    actions = []
    next_states = []
    dones = []

    for s in tqdm(range(num_episodes), desc="Episodes"):
        obs, _ = env.reset(seed=s)
        ep_len = 0
        ep_ret = 0.0
        obses_list = []
        while True:
            states.append(obs)
            act = policy_fn(policy_params, obs)
            actions.append(act)
            obs, reward, terminated, truncated, _ = env.step(act)
            next_states.append(obs)
            dones.append(terminated)
            ep_len += 1
            ep_ret += reward

            obses_list.append(obs)
            if terminated or truncated:
                break

        ep_len_list.append(ep_len)
        ep_ret_list.append(ep_ret)
        np.savez(dataset_name, states=states, actions=actions, next_states=next_states, dones=dones)

    return ep_len_list, ep_ret_list, obses_list

if __name__ == "__main__":
    # policy_root = Path('/home/haitong/PycharmProjects/DACER-Diffusion-with-Online-RL/logs/dm_control_walker_run-v0/sdac_2025-02-19_22-15-17_s100_test_use_atp1')
    # policy_path = "policy-1000000-200000.pkl"
    # policy_root = Path('/home/haitong/PycharmProjects/DACER-Diffusion-with-Online-RL/logs/dm_control_walker_stand-v0/sdac_2025-02-19_21-55-57_s100_test_use_atp1')
    # policy_path = "policy-700000-140000.pkl"
    policy_root = Path('/home/haitong/PycharmProjects/DACER-Diffusion-with-Online-RL/logs/dm_control_walker_walk-v0/sdac_2025-02-19_22-15-17_s100_test_use_atp1')
    policy_path ="policy-500000-100000.pkl"
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

    with open(policy_root / policy_path, "rb") as f:
        policy_params = pickle.load(f)
    dataset_dir = policy_root.parent / 'data'
    os.makedirs(str(dataset_dir), exist_ok=True)
    episodes = 50
    ep_len_list, ep_ret_list, _ = collect_data(env, policy_fn, policy_params, episodes,
                                               str(dataset_dir / policy_path.split('-')[1]) + '_{}'.format(episodes))
