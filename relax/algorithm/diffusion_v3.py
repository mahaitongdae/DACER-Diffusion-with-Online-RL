from typing import NamedTuple, Tuple, List

import jax, jax.numpy as jnp
import optax
import haiku as hk

from relax.algorithm.base import Algorithm
from relax.network.dacer import DACERNet, DACERParams
from relax.network.diffv2 import Diffv2Net, Diffv2Params
from relax.network.diffv3 import Diffv3Net, Diffv3Params
from relax.utils.experience import Experience
from relax.utils.typing import Metric


class Diffv3OptStates(NamedTuple):
    q1: optax.OptState
    q2: optax.OptState
    policy: optax.OptState
    std: optax.OptState
    log_alpha: optax.OptState


class Diffv3TrainState(NamedTuple):
    params: Diffv3Params
    opt_state: Diffv3OptStates
    step: int
    # entropy: float

class Diffv2(Algorithm):

    def __init__(
        self,
        agent: Diffv3Net,
        params: Diffv3Params,
        *,
        gamma: float = 0.99,
        lr: float = 1e-4,
        alpha_lr: float = 3e-2,
        tau: float = 0.005,
        delay_alpha_update: int = 1000,
        delay_update: int = 2,
        reward_scale: float = 0.2,
        num_samples: int = 200,
    ):
        self.agent = agent
        self.gamma = gamma
        self.tau = tau
        self.delay_alpha_update = delay_alpha_update
        self.delay_update = delay_update
        self.reward_scale = reward_scale
        self.num_samples = num_samples
        self.optim = optax.adam(lr)
        self.alpha_optim = optax.adam(alpha_lr)
        self.entropy = 0.0

        self.state = Diffv3TrainState(
            params=params,
            opt_state=Diffv3OptStates(
                q1=self.optim.init(params.q1),
                q2=self.optim.init(params.q2),
                policy=self.optim.init(params.policy),
                std=self.optim.init(params.std),
                log_alpha=self.alpha_optim.init(params.log_alpha),
            ),
            step=jnp.int32(0),
            # entropy=jnp.float32(0.0),
        )

        @jax.jit
        def stateless_update(
            key: jax.Array, state: Diffv3TrainState, data: Experience
        ) -> Tuple[Diffv3TrainState, Metric]:
            obs, action, reward, next_obs, done = data.obs, data.action, data.reward, data.next_obs, data.done
            q1_params, q2_params, target_q1_params, target_q2_params, policy_params, std_params, log_alpha = state.params
            q1_opt_state, q2_opt_state, policy_opt_state, std_opt_state, log_alpha_opt_state = state.opt_state
            step = state.step
            (next_eval_key, new_eval_key, new_q1_eval_key, new_q2_eval_key,
             log_alpha_key, diffusion_time_key, diffusion_noise_key, exploration_noise_key) = jax.random.split(
                key, 8)

            reward *= self.reward_scale

            def get_min_q(s, a):
                q1 = self.agent.q(q1_params, s, a)
                q2 = self.agent.q(q2_params, s, a)
                q = jnp.minimum(q1, q2)
                return q

            # # compute target q
            # next_action = self.agent.get_action(next_eval_key, (policy_params, log_alpha), next_obs)
            # next_q1_mean, _, next_q1_sample = self.agent.q_evaluate(new_q1_eval_key, target_q1_params, next_obs, next_action)
            # next_q2_mean, _, next_q2_sample = self.agent.q_evaluate(new_q2_eval_key, target_q2_params, next_obs, next_action)
            # next_q_mean = jnp.minimum(next_q1_mean, next_q2_mean)
            # next_q_sample = jnp.where(next_q1_mean < next_q2_mean, next_q1_sample, next_q2_sample)
            # q_target = next_q_mean
            # q_target_sample = next_q_sample
            # q_backup = reward + (1 - done) * self.gamma * q_target
            # q_backup_sample = reward + (1 - done) * self.gamma * q_target_sample
            #
            # # update q
            # def q_loss_fn(q_params: hk.Params, mean_q_std: float) -> jax.Array:
            #     q_mean, q_std = self.agent.q(q_params, obs, action)
            #     new_mean_q_std = jnp.mean(q_std)
            #     mean_q_std = jax.lax.stop_gradient(
            #         (mean_q_std == -1.0) * new_mean_q_std +
            #         (mean_q_std != -1.0) * (self.tau * new_mean_q_std + (1 - self.tau) * mean_q_std)
            #     )
            #     q_backup_bounded = jax.lax.stop_gradient(q_mean + jnp.clip(q_backup_sample - q_mean, -3 * mean_q_std, 3 * mean_q_std))
            #     q_std_detach = jax.lax.stop_gradient(jnp.maximum(q_std, 0))
            #     epsilon = 0.1
            #     q_loss = -(mean_q_std ** 2 + epsilon) * jnp.mean(
            #         q_mean * jax.lax.stop_gradient(q_backup - q_mean) / (q_std_detach ** 2 + epsilon) +
            #         q_std * ((jax.lax.stop_gradient(q_mean) - q_backup_bounded) ** 2 - q_std_detach ** 2) / (q_std_detach ** 3 + epsilon)
            #     )
            #     return q_loss, (q_mean, q_std, mean_q_std)
            #
            # (q1_loss, (q1_mean, q1_std, mean_q1_std)), q1_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q1_params, mean_q1_std)
            # (q2_loss, (q2_mean, q2_std, mean_q2_std)), q2_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q2_params, mean_q2_std)

            # compute target q
            next_action, _ = self.agent.get_noised_actions_with_logp(next_eval_key, policy_params, std_params,
                                                          next_obs, get_min_q)
            q1_target = self.agent.q(target_q1_params, next_obs, next_action)
            q2_target = self.agent.q(target_q2_params, next_obs, next_action)
            q_target = jnp.minimum(q1_target, q2_target)  # - jnp.exp(log_alpha) * next_logp
            q_backup = reward + (1 - done) * self.gamma * q_target

            def q_loss_fn(q_params: hk.Params) -> (jax.Array, jax.Array):
                q = self.agent.q(q_params, obs, action)
                q_loss = jnp.mean((q - q_backup) ** 2)
                return q_loss, q

            (q1_loss, q1), q1_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q1_params)
            (q2_loss, q2), q2_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q2_params)
            q1_update, q1_opt_state = self.optim.update(q1_grads, q1_opt_state)
            q2_update, q2_opt_state = self.optim.update(q2_grads, q2_opt_state)
            q1_params = optax.apply_updates(q1_params, q1_update)
            q2_params = optax.apply_updates(q2_params, q2_update)

            # def cal_entropy():
            #     keys = jax.random.split(log_alpha_key, self.num_samples)
            #     actions = jax.vmap(self.agent.get_action, in_axes=(0, None, None), out_axes=1)(keys, (policy_params, jax.lax.stop_gradient(log_alpha)), obs)
            #     entropy = jax.pure_callback(estimate_entropy, jax.ShapeDtypeStruct((), jnp.float32), actions)
            #     entropy = jax.lax.stop_gradient(entropy)
            #     return entropy
            #
            # prev_entropy = state.entropy if hasattr(state, 'entropy') else jnp.float32(0.0)
            #
            # entropy = jax.lax.cond(
            #     step % self.delay_alpha_update == 0,
            #     cal_entropy,
            #     lambda: prev_entropy
            # )

            # update policy
            def policy_loss_fn(policy_params, std_params) -> (jax.Array, Tuple[jax.Array]):
                new_raw_action = self.agent.get_batch_actions(new_eval_key, policy_params, obs, get_min_q)
                new_exploration_noise, new_logp = self.agent.get_exploration_noise(exploration_noise_key, std_params, obs, new_raw_action)
                exploration_action = (new_raw_action + new_exploration_noise).clip(-1, 1)
                # q1_mean, _ = self.agent.q(q1_params, obs, new_action)
                # q2_mean, _ = self.agent.q(q2_params, obs, new_action)
                # q_mean = jnp.minimum(q1_mean, q2_mean)
                q = get_min_q(obs, new_raw_action)
                norm_q = (q - q.mean()) / q.std()
                q_weights = jnp.exp(norm_q.clip(-3., 3.))
                q_weights = q_weights / jnp.exp(log_alpha)
                nq = get_min_q(obs, exploration_action)
                def denoiser(t, x):
                    return self.agent.policy(policy_params, obs, x, t)
                t = jax.random.randint(diffusion_time_key, (obs.shape[0],), 0, self.agent.num_timesteps)

                sm_loss = self.agent.diffusion.weighted_p_loss(diffusion_noise_key, q_weights, denoiser, t,
                                                            jax.lax.stop_gradient(new_raw_action))
                q_loss = - nq.mean()
                entropy_loss = log_alpha * -1. * new_logp.mean()
                loss = sm_loss + 0.01 * q_loss + entropy_loss

                return loss, (q_weights, new_logp, sm_loss, q_loss, entropy_loss,)

            (total_loss, (q_weights, new_logp, sm_loss, q_loss, entropy_loss,)), (
            policy_grads, std_grads) = jax.value_and_grad(policy_loss_fn, has_aux=True)(policy_params, std_params)


            # update alpha
            def log_alpha_loss_fn(log_alpha: jax.Array) -> jax.Array:
                # log_alpha_loss = -jnp.mean(log_alpha * (-entropy + self.agent.target_entropy))
                log_alpha_loss = log_alpha * (-1 * new_logp - self.agent.target_entropy).mean()
                return log_alpha_loss

            # update networks
            def param_update(optim, params, grads, opt_state):
                update, new_opt_state = optim.update(grads, opt_state)
                new_params = optax.apply_updates(params, update)
                return new_params, new_opt_state

            def delay_param_update(optim, params, grads, opt_state):
                return jax.lax.cond(
                    step % self.delay_update == 0,
                    lambda params, opt_state: param_update(optim, params, grads, opt_state),
                    lambda params, opt_state: (params, opt_state),
                    params, opt_state
                )

            def delay_alpha_param_update(optim, params, opt_state):
                return jax.lax.cond(
                    step % self.delay_alpha_update == 0,
                    lambda params, opt_state: param_update(optim, params, jax.grad(log_alpha_loss_fn)(params), opt_state),
                    lambda params, opt_state: (params, opt_state),
                    params, opt_state
                )

            def delay_target_update(params, target_params, tau):
                return jax.lax.cond(
                    step % self.delay_update == 0,
                    lambda target_params: optax.incremental_update(params, target_params, tau),
                    lambda target_params: target_params,
                    target_params
                )

            q1_params, q1_opt_state = param_update(self.optim, q1_params, q1_grads, q1_opt_state)
            q2_params, q2_opt_state = param_update(self.optim, q2_params, q2_grads, q2_opt_state)
            policy_params, policy_opt_state = delay_param_update(self.optim, policy_params, policy_grads, policy_opt_state)
            std_params, std_opt_state = delay_param_update(self.optim, std_params, std_grads, std_opt_state)
            log_alpha, log_alpha_opt_state = delay_alpha_param_update(self.alpha_optim, log_alpha, log_alpha_opt_state)

            target_q1_params = delay_target_update(q1_params, target_q1_params, self.tau)
            target_q2_params = delay_target_update(q2_params, target_q2_params, self.tau)

            state = Diffv3TrainState(
                params=Diffv3Params(q1_params, q2_params, target_q1_params, target_q2_params, policy_params, std_params, log_alpha),
                opt_state=Diffv3OptStates(q1=q1_opt_state, q2=q2_opt_state, policy=policy_opt_state, log_alpha=log_alpha_opt_state),
                step=step + 1,
                # entropy=entropy,
            )
            info = {
                "q1_loss": q1_loss,
                "q1_mean": jnp.mean(q1),
                "q1_max": jnp.max(q1),
                "q1_min": jnp.min(q1),
                # "q1_std": jnp.mean(q1_std),
                "q2_loss": q2_loss,
                # "q2_mean": jnp.mean(q2_mean),
                # "q2_std": jnp.mean(q2_std),
                "policy_loss": total_loss,
                "alpha": jnp.exp(log_alpha),
                "q_weights_std": jnp.std(q_weights),
                "q_weights_max": jnp.max(q_weights),
                "sm_loss": sm_loss,
                "q_loss": q_loss,
                "entropy_loss": entropy_loss,
                "entropy": -new_logp.mean(),
                # "mean_q1_std": mean_q1_std,
                # "mean_q2_std": mean_q2_std,
            }
            return state, info

        self._implement_common_behavior(stateless_update, self.agent.get_action, self.agent.get_deterministic_action)

    def get_policy_params(self):
        return (self.state.params.policy, self.state.params.log_alpha)

def estimate_entropy(actions, num_components=3):  # (batch, sample, dim)
    import numpy as np
    from sklearn.mixture import GaussianMixture
    total_entropy = []
    for action in actions:
        gmm = GaussianMixture(n_components=num_components, covariance_type='full')
        gmm.fit(action)
        weights = gmm.weights_
        entropies = []
        for i in range(gmm.n_components):
            cov_matrix = gmm.covariances_[i]
            d = cov_matrix.shape[0]
            entropy = 0.5 * d * (1 + np.log(2 * np.pi)) + 0.5 * np.linalg.slogdet(cov_matrix)[1]
            entropies.append(entropy)
        entropy = -np.sum(weights * np.log(weights)) + np.sum(weights * np.array(entropies))
        total_entropy.append(entropy)
    final_entropy = sum(total_entropy) / len(total_entropy)
    return final_entropy