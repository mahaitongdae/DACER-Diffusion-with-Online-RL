from typing import NamedTuple, Tuple

import jax, jax.numpy as jnp
import optax
import haiku as hk

from relax.algorithm.base import Algorithm
from relax.network.sac_ctrl import CTRLSACNet, CTRLSACParams
from relax.utils.experience import Experience
from relax.utils.typing import Metric


class CTRLSACOptStates(NamedTuple):
    q1: optax.OptState
    q2: optax.OptState
    phi: optax.OptState
    mu: optax.OptState
    policy: optax.OptState
    log_alpha: optax.OptState


class CTRLSACTrainState(NamedTuple):
    params: CTRLSACParams
    opt_state: CTRLSACOptStates


class CTRLSAC(Algorithm):
    def __init__(self, agent: CTRLSACNet, params: CTRLSACParams, *, gamma: float = 0.99, lr: float = 1e-4,
                 alpha_lr: float = 3e-4, tau: float = 0.005, reward_scale: float = 0.2,):
        self.agent = agent
        self.gamma = gamma
        self.tau = tau
        self.optim = optax.adam(lr)
        self.log_alpha_optim = optax.adam(alpha_lr)
        self.reward_scale = reward_scale

        self.state = CTRLSACTrainState(
            params=params,
            opt_state=CTRLSACOptStates(
                q1=self.optim.init(params.q1),
                q2=self.optim.init(params.q2),
                phi=self.optim.init(params.phi),
                mu=self.optim.init(params.mu),
                policy=self.optim.init(params.policy),
                log_alpha=self.log_alpha_optim.init(params.log_alpha),
            ),
        )

        @jax.jit
        def stateless_update(
            key: jax.Array, state: CTRLSACTrainState, data: Experience
        ) -> Tuple[CTRLSACTrainState, Metric]:
            obs, action, reward, next_obs, done = data.obs, data.action, data.reward, data.next_obs, data.done
            (q1_params, q2_params, target_q1_params, target_q2_params, 
            phi_params, mu_params, target_phi_params, policy_params, log_alpha) = state.params
            q1_opt_state, q2_opt_state, phi_opt_state, mu_opt_state, policy_opt_state, log_alpha_opt_state = state.opt_state
            next_eval_key, new_eval_key = jax.random.split(key)

            reward *= self.reward_scale

            # reprsentation learning
            def repr_loss_fn(phi_params, mu_params):
                phi = self.agent.phi({'params': phi_params}, obs, action)
                mu = self.agent.mu({'params': mu_params}, next_obs)
                labels = jnp.eye(phi.shape[0])
                contrastive = jnp.sum(phi[:, jnp.newaxis, :] * mu[jnp.newaxis, :, :], axis=-1)
                loss = optax.losses.sigmoid_binary_cross_entropy(contrastive, labels).mean()
                return loss, phi
            
            for _ in range(3):
                (repr_loss, phi), (phi_grad, mu_grad) = jax.value_and_grad(repr_loss_fn, argnums=(0, 1), has_aux=True)(phi_params, mu_params)
                phi_update, phi_opt_state = self.optim.update(phi_grad, phi_opt_state)
                mu_update, mu_opt_state = self.optim.update(mu_grad, mu_opt_state)
                optax.apply_updates(phi_params, phi_update)
                optax.apply_updates(mu_params, mu_update)

            # compute target q
            next_action, next_logp = self.agent.evaluate(next_eval_key, policy_params, next_obs)
            next_phi = self.agent.phi({'params': target_phi_params}, next_obs, next_action)
            q1_target = self.agent.q({'params': target_q1_params}, next_phi)
            q2_target = self.agent.q({'params': target_q2_params}, next_phi)
            q_target = jnp.minimum(q1_target, q2_target) - jnp.exp(log_alpha) * next_logp
            q_backup = reward + (1 - done) * self.gamma * q_target

            # update q
            def q_loss_fn(q_params: dict) -> jax.Array:
                q = self.agent.q({'params': q_params}, phi)
                q_loss = jnp.mean((q - q_backup) ** 2)
                return q_loss

            q1_loss, q1_grads = jax.value_and_grad(q_loss_fn)(q1_params)
            q2_loss, q2_grads = jax.value_and_grad(q_loss_fn)(q2_params)
            q1_update, q1_opt_state = self.optim.update(q1_grads, q1_opt_state)
            q2_update, q2_opt_state = self.optim.update(q2_grads, q2_opt_state)
            q1_params = optax.apply_updates(q1_params, q1_update)
            q2_params = optax.apply_updates(q2_params, q2_update)

            # update policy
            def policy_loss_fn(policy_params: dict, phi_params) -> jax.Array:
                new_action, new_logp = self.agent.evaluate(new_eval_key, policy_params, obs)
                new_phi = self.agent.phi({'params': phi_params}, obs, new_action)
                q1 = self.agent.q({'params': q1_params}, new_phi)
                q2 = self.agent.q({'params': q2_params}, new_phi)
                q = jnp.minimum(q1, q2)
                policy_loss = jnp.mean(jnp.exp(log_alpha) * new_logp - q)
                return policy_loss, (q1, q2, new_logp)

            (policy_loss, aux), policy_grads = jax.value_and_grad(policy_loss_fn, has_aux=True)(policy_params, target_phi_params)
            q1, q2, new_logp = aux
            policy_update, policy_opt_state = self.optim.update(policy_grads, policy_opt_state)
            policy_params = optax.apply_updates(policy_params, policy_update)

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

            state = CTRLSACTrainState(
                params=CTRLSACParams(q1_params, q2_params, target_q1_params, target_q2_params, 
                                     phi_params, mu_params, target_phi_params, policy_params, log_alpha),
                opt_state=CTRLSACOptStates(q1_opt_state, q2_opt_state, phi_opt_state, mu_opt_state, 
                                           policy_opt_state, log_alpha_opt_state),
            )
            info = {
                "q1_loss": q1_loss,
                "q2_loss": q2_loss,
                "q1": jnp.mean(q1),
                "q2": jnp.mean(q2),
                "policy_loss": policy_loss,
                "repr_loss": repr_loss,
                "entropy": -jnp.mean(new_logp),
                "alpha": jnp.exp(log_alpha),
            }
            return state, info

        self._implement_common_behavior(stateless_update, self.agent.get_action, self.agent.get_deterministic_action)
