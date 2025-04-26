import os.path
from pathlib import Path
from functools import partial
import yaml
import hydra
from omegaconf import DictConfig, OmegaConf
from types import SimpleNamespace
from hydra.core.hydra_config import HydraConfig

import jax, jax.numpy as jnp

from relax.algorithm.sac import SAC
from relax.algorithm.dacer import DACER
from relax.algorithm.qsm import QSM
from relax.algorithm.dipo import DIPO
from relax.algorithm.qvpo import QVPO
from relax.algorithm.sdac import SDAC
from relax.algorithm.sac_ctrl import CTRLSAC
from relax.algorithm.sac_rand import RANDSAC
from relax.buffer import TreeBuffer
from relax.network.sac import create_sac_net
from relax.network.sac_ctrl import create_ctrl_sac_net
from relax.network.dacer import create_dacer_net
from relax.network.qsm import create_qsm_net
from relax.network.dipo import create_dipo_net
from relax.network.sdac import create_sdac_net
from relax.network.qvpo import create_qvpo_net
from relax.network.sac_rand import create_rand_sac_net
from relax.trainer.off_policy import OffPolicyTrainer
from relax.env import create_env, create_vector_env
from relax.utils.experience import Experience, ObsActionPair
from relax.utils.fs import PROJECT_ROOT
from relax.utils.random_utils import seeding
from relax.utils.log_diff import log_git_details

@hydra.main(version_base=None, config_path='../config', config_name='relax')
def run(args: DictConfig):
    alg_args = args.alg
    if args.debug:
        from jax import config
        config.update("jax_disable_jit", True)

    master_seed = args.seed
    master_rng, _ = seeding(master_seed)
    env_seed, env_action_seed, eval_env_seed, buffer_seed, init_network_seed, train_seed = map(
        int, master_rng.integers(0, 2**32 - 1, 6)
    )
    init_network_key = jax.random.key(init_network_seed)
    train_key = jax.random.key(train_seed)
    del init_network_seed, train_seed

    if 'dm_control' in args.env:
        from relax.env.dmc.register import register_dm_control_envs
        register_dm_control_envs()
    if 'pusht' in args.env:
        from relax.env.pusht.pusht_env import PushTEnv

    if args.num_vec_envs > 0:
        env, obs_dim, act_dim = create_vector_env(args.env, args.num_vec_envs, env_seed, env_action_seed, mode="futex")
    else:
        env, obs_dim, act_dim = create_env(args.env, env_seed, env_action_seed)
    eval_env = None

    hidden_sizes = [alg_args.hidden_dim] * alg_args.hidden_num
    diffusion_hidden_sizes = [alg_args.diffusion_hidden_dim] * alg_args.hidden_num


    buffer = TreeBuffer.from_experience(obs_dim, act_dim, size=int(1e6), seed=buffer_seed)

    gelu = partial(jax.nn.gelu, approximate=False)

    if alg_args.alg_name == 'sdac':
        total_grad_steps = args.total_step if args.num_vec_envs == 0 else int(args.total_step / args.num_vec_envs)
        def mish(x: jax.Array):
            return x * jnp.tanh(jax.nn.softplus(x))
        agent, params = create_sdac_net(init_network_key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, mish,
                                          num_timesteps=alg_args.diffusion_steps,
                                          num_particles=alg_args.num_particles,
                                          noise_scale=alg_args.noise_scale,
                                          target_entropy_scale=alg_args.target_entropy_scale)
        algorithm = SDAC(agent, params, lr=alg_args.lr, alpha_lr=alg_args.alpha_lr,
                           delay_alpha_update=alg_args.delay_alpha_update,
                             lr_schedule_end=alg_args.lr_schedule_end,
                             use_ema=alg_args.use_ema_policy, total_grad_steps=total_grad_steps)
    elif alg_args.alg_name == "qsm":
        agent, params = create_qsm_net(init_network_key, obs_dim, act_dim, hidden_sizes, num_timesteps=20, num_particles=alg_args.num_particles)
        algorithm = QSM(agent, params, lr=alg_args.lr, lr_schedule_end=alg_args.lr_schedule_end)
    elif alg_args.alg_name == "sac":
        agent, params = create_sac_net(init_network_key, obs_dim, act_dim, hidden_sizes, gelu)
        algorithm = SAC(agent, params, lr=alg_args.lr)
    elif alg_args.alg_name == "ctrlsac":
        repr_dim = alg_args.repr_dim
        w_hidden_sizes = [alg_args.hidden_dim] * alg_args.w_hidden_num
        policy_hidden_sizes = [alg_args.policy_hidden_dim] * alg_args.hidden_num
        agent, params = create_ctrl_sac_net(init_network_key, obs_dim, act_dim, repr_dim,
                                            hidden_sizes, w_hidden_sizes, policy_hidden_sizes=policy_hidden_sizes,
                                            activation=gelu, w_activation=gelu)
        algorithm = CTRLSAC(agent, params, obs_dim, repr_dim, lr=alg_args.lr, alpha_lr=alg_args.alpha_lr)
    elif alg_args.alg_name == "randsac":
        repr_dim = alg_args.repr_dim
        sigma = alg_args.sigma
        random_feature_dim = alg_args.random_feature_dim
        w_hidden_sizes = [alg_args.hidden_dim] * alg_args.w_hidden_num
        policy_hidden_sizes = [alg_args.policy_hidden_dim] * alg_args.hidden_num
        agent, params = create_rand_sac_net(init_network_key, obs_dim, act_dim, repr_dim,
                                            hidden_sizes, w_hidden_sizes,
                                            policy_hidden_sizes=policy_hidden_sizes,
                                            activation=gelu, w_activation=gelu,
                                            mu_random_feature_sigma=sigma,
                                            mu_random_feature_dim=random_feature_dim
                                            )
        algorithm = RANDSAC(agent, params, obs_dim, repr_dim, lr=alg_args.lr, alpha_lr=alg_args.alpha_lr)
    elif alg_args.alg_name == "dacer":
        def mish(x: jax.Array):
            return x * jnp.tanh(jax.nn.softplus(x))
        agent, params = create_dacer_net(init_network_key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, mish,
                                         num_timesteps=alg_args.diffusion_steps)
        algorithm = DACER(agent, params, lr=alg_args.lr, lr_schedule_end=alg_args.lr_schedule_end)
    elif alg_args.alg_name == "dipo":
        diffusion_buffer = TreeBuffer.from_example(
            ObsActionPair.create_example(obs_dim, act_dim),
            args.total_step,
            int(master_rng.integers(0, 2**32 - 1)),
            remove_batch_dim=False
        )
        TreeBuffer.connect(buffer, diffusion_buffer, lambda exp: ObsActionPair(exp.obs, exp.action))

        def mish(x: jax.Array):
            return x * jnp.tanh(jax.nn.softplus(x))

        agent, params = create_dipo_net(init_network_key, obs_dim, act_dim, hidden_sizes, num_timesteps=100)
        algorithm = DIPO(agent, params, diffusion_buffer, lr=alg_args.lr, action_gradient_steps=30, policy_target_delay=2, action_grad_norm=0.16)
    elif alg_args.alg_name == "qvpo":
        def mish(x: jax.Array):
            return x * jnp.tanh(jax.nn.softplus(x))
        agent, params = create_qvpo_net(init_network_key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, mish,
                                          num_timesteps=alg_args.diffusion_steps,
                                          num_particles=alg_args.num_particles,
                                          noise_scale=alg_args.noise_scale)
        algorithm = QVPO(agent, params, lr=alg_args.lr, alpha_lr=alg_args.alpha_lr, delay_alpha_update=alg_args.delay_alpha_update)
    else:
        raise ValueError(f"Invalid algorithm {alg_args.alg_name}!")

    exp_dir = Path(HydraConfig.get().run.dir) # PROJECT_ROOT / "logs" / args.env / (alg_args.alg_name + '_' + time.strftime("%Y-%m-%d_%H-%M-%S") + f'_s{args.seed}_{args.suffix}')
    trainer = OffPolicyTrainer(
        env=env,
        algorithm=algorithm,
        buffer=buffer,
        start_step=args.start_step,
        total_step=args.total_step,
        sample_per_iteration=1,
        evaluate_env=eval_env,
        save_policy_every=int(args.total_step / args.total_evals),
        warmup_with="random",
        log_path=exp_dir,
    )

    trainer.setup(Experience.create_example(obs_dim, act_dim, trainer.batch_size))
    log_git_details(log_file=os.path.join(exp_dir, 'git.diff'))

    # Save the arguments to a YAML file
    # args_dict = args
    # with open(os.path.join(exp_dir, 'config.yaml'), 'w') as yaml_file:
    #     yaml.dump(args_dict, yaml_file)
    trainer.run(train_key)


if __name__ == "__main__":
    run()

