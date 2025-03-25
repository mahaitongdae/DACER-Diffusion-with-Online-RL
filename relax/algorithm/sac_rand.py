from typing import NamedTuple, Tuple

import jax, jax.numpy as jnp
import optax
import haiku as hk

from relax.algorithm.base import ReprAlgorithm
from relax.network.sac_rand import RANDSACNet, RANDSACParams
from relax.utils.experience import Experience
from relax.utils.typing import Metric


class RANDSACOptStates(NamedTuple):
    q1: optax.OptState
    q2: optax.OptState
    reward_pred: optax.OptState
    phi: optax.OptState
    mu: optax.OptState
    policy: optax.OptState
    log_alpha: optax.OptState


class RandSACTrainState(NamedTuple):
    params: RANDSACParams
    opt_state: RANDSACOptStates
    running_mean: jax.Array
    running_std: jax.Array


class RANDSAC(ReprAlgorithm):
    def __init__(self, agent: RANDSACNet, params: RANDSACParams, obs_dim, repr_dim, *, gamma: float = 0.99, lr: float = 1e-4,
                 alpha_lr: float = 3e-4, tau: float = 0.005, reward_scale: float = 0.2,):
        self.agent = agent
        self.gamma = gamma
        self.tau = tau
        self.critic_optim = optax.adam(3e-4)
        self.optim = optax.adam(lr)
        self.log_alpha_optim = optax.adam(alpha_lr)
        self.reward_scale = reward_scale

        

        self.state = RandSACTrainState(
            params=params,
            opt_state=RANDSACOptStates(
                q1=self.optim.init(params.q1),
                q2=self.optim.init(params.q2),
                reward_pred=self.optim.init(params.reward_pred),
                phi=self.optim.init(params.phi),
                mu=self.optim.init(params.mu),
                policy=self.optim.init(params.policy),
                log_alpha=self.log_alpha_optim.init(params.log_alpha),
            ),
            running_mean=jnp.zeros([1, obs_dim]),
            running_std=jnp.ones([1, obs_dim])
        )

        def reward_fn(obs, action):
            cos_th, sin_th, thdot = obs[:, 0], obs[:, 1], obs[:, 2]
            th = jnp.atan2(sin_th, cos_th)
            action = jnp.reshape(action, (action.shape[0],))
            reward = -1 * self.reward_scale * (th ** 2 + 0.1 * thdot ** 2 + 0.001 * action ** 2)
            return reward


        @jax.jit
        def stateless_update(
            key: jax.Array, state: RandSACTrainState, data: Experience
        ) -> Tuple[RandSACTrainState, Metric]:
            obs, action, reward, next_obs, done = data.obs, data.action, data.reward, data.next_obs, data.done
            (q1_params, q2_params, target_q1_params, target_q2_params, reward_pred_params,
            phi_params, mu_params, target_phi_params, policy_params, log_alpha) = state.params
            (q1_opt_state, q2_opt_state, 
             reward_pred_opt_state, phi_opt_state, mu_opt_state, policy_opt_state, log_alpha_opt_state) = state.opt_state
            next_eval_key, new_eval_key = jax.random.split(key)
            running_mean = state.running_mean
            running_std = state.running_std
            reward *= self.reward_scale
            new_mean = jnp.mean(obs, axis=0, keepdims=True)
            new_std = jnp.std(obs, axis=0, keepdims=True)
            # obs = (obs - running_mean) / running_std
            # next_obs = (next_obs - running_mean) / running_std

            # reprsentation learning
            def repr_loss_fn(phi_params, mu_params):
                phi = self.agent.phi({'params': phi_params}, obs, action)
                mu = jax.lax.stop_gradient(self.agent.mu({'params': mu_params}, next_obs))
                prob = jnp.linalg.vecdot(phi, mu, axis=-1)
                norm = jnp.sum(phi ** 2, axis=1)
                loss = jnp.mean(-2 * prob + norm)
                return loss, (phi, prob, norm)
            
            for _ in range(1):
                (repr_loss, aux), phi_grad = jax.value_and_grad(repr_loss_fn, has_aux=True)(phi_params, mu_params)
                phi, prob, phi_norm = aux
                phi_update, phi_opt_state = self.optim.update(phi_grad, phi_opt_state)
                optax.apply_updates(phi_params, phi_update)

            # def reward_pred_loss_fn(reward_pred_params: dict) -> jax.Array:
            #     reward_pred = self.agent.reward_pred({'params': reward_pred_params}, obs, action)
            #     return optax.losses.l2_loss(reward_pred, reward).mean(), reward_pred
            
            # (reward_pred_loss, reward_pred), reward_pred_grad = jax.value_and_grad(reward_pred_loss_fn, has_aux=True)(reward_pred_params)
            # reward_pred_update, reward_pred_opt_state = self.optim.update(reward_pred_grad, reward_pred_opt_state)
            # optax.apply_updates(reward_pred_params, reward_pred_update)

            # compute target q
            next_action, next_logp = self.agent.evaluate(next_eval_key, policy_params, next_obs)
            # next_phi = self.agent.phi({'params': target_phi_params}, next_obs, next_action)
            next_phi = self.agent.phi({'params': target_phi_params}, next_obs, next_action)
            q1_target = self.agent.q({'params': target_q1_params}, next_phi)
            q2_target = self.agent.q({'params': target_q2_params}, next_phi)
            q_target = jnp.minimum(q1_target, q2_target) - jnp.exp(log_alpha) * next_logp
            # next_reward = self.agent.reward_pred({'params': reward_pred_params}, next_obs, next_action)
            next_reward = reward_fn(next_obs, next_action)
            q_backup = self.gamma *next_reward + (1 - done) * self.gamma * q_target

            # update q
            def q_loss_fn(q_params: dict) -> jax.Array:
                q = self.agent.q({'params': q_params}, phi)
                q_loss = jnp.mean((q - q_backup) ** 2)
                return q_loss

            for _ in range(1):
                q1_loss, q1_grads = jax.value_and_grad(q_loss_fn)(q1_params)
                q2_loss, q2_grads = jax.value_and_grad(q_loss_fn)(q2_params)
                q1_update, q1_opt_state = self.critic_optim.update(q1_grads, q1_opt_state)
                q2_update, q2_opt_state = self.critic_optim.update(q2_grads, q2_opt_state)
                q1_params = optax.apply_updates(q1_params, q1_update)
                q2_params = optax.apply_updates(q2_params, q2_update)

            # update policy
            def policy_loss_fn(policy_params: dict, phi_params: dict) -> jax.Array:
                new_action, new_logp = self.agent.evaluate(new_eval_key, policy_params, obs)
                new_phi = self.agent.phi({'params': phi_params}, obs, new_action)
                q1 = self.agent.q({'params': q1_params}, new_phi)
                q2 = self.agent.q({'params': q2_params}, new_phi)
                q = jnp.minimum(q1, q2)
                # new_reward_pred = self.agent.reward_pred({'params': reward_pred_params}, obs, new_action)
                new_reward_pred = reward_fn(obs, new_action)
                policy_loss = jnp.mean(jnp.exp(log_alpha) * new_logp - self.gamma * q - new_reward_pred)
                return policy_loss, (q1, q2, new_logp)

            (policy_loss, aux), policy_grads = jax.value_and_grad(policy_loss_fn, has_aux=True)(policy_params, phi_params)
            q1, q2, new_logp = aux
            policy_update, policy_opt_state = self.optim.update(policy_grads, policy_opt_state)
            policy_params = optax.apply_updates(policy_params, policy_update)
            policy_grad_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(policy_grads)))

            # update alpha
            def log_alpha_loss_fn(log_alpha: jax.Array) -> jax.Array:
                log_alpha_loss = -jnp.mean(log_alpha * (new_logp + self.agent.target_entropy))
                return log_alpha_loss

            log_alpha_grads = jax.grad(log_alpha_loss_fn)(log_alpha)
            log_alpha_update, log_alpha_opt_state = self.log_alpha_optim.update(log_alpha_grads, log_alpha_opt_state)
            log_alpha = optax.apply_updates(log_alpha, log_alpha_update)

            # update target q
            target_q1_params = optax.incremental_update(q1_params, target_q1_params, self.tau)
            target_q2_params = optax.incremental_update(q2_params, target_q2_params, self.tau)
            target_phi_params = optax.incremental_update(phi_params, target_phi_params, self.tau)

            running_mean = optax.incremental_update(running_mean, new_mean, self.tau)
            running_std = optax.incremental_update(running_std, new_std, self.tau)

            state = RandSACTrainState(
                params=RANDSACParams(q1_params, q2_params, target_q1_params, target_q2_params, reward_pred_params,
                                     phi_params, mu_params, target_phi_params, policy_params, log_alpha),
                opt_state=RANDSACOptStates(q1_opt_state, q2_opt_state, reward_pred_opt_state, phi_opt_state, mu_opt_state,
                                           policy_opt_state, log_alpha_opt_state),
                running_mean = running_mean,
                running_std = running_std
            )
            info = {
                "q1_loss": q1_loss,
                "q2_loss": q2_loss,
                "q1": jnp.mean(q1),
                "q2": jnp.mean(q2),
                "policy_loss": policy_loss,
                "repr_loss": repr_loss,
                # "reward_pred_loss": reward_pred_loss, 
                "entropy": -jnp.mean(new_logp),
                "alpha": jnp.exp(log_alpha),
                "policy_grad_norm": policy_grad_norm,
                "phi_norm": jnp.linalg.norm(phi, axis=-1).mean(),
                "loss_phi_norm": phi_norm.mean(),
                "dist_q1": q1,
                "dist_q2": q2,
                "dist_prob": prob,
            }
            return state, info

        self._implement_common_behavior(stateless_update, 
                                        self.agent.get_action, 
                                        self.agent.get_deterministic_action,
                                        self.agent.get_phi)
        
    def get_phi_params(self):
        return super().get_phi_params()
