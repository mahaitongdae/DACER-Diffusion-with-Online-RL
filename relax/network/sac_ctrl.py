from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence, Tuple

import jax, jax.numpy as jnp

# from relax.network.blocks import Activation, QNet, PolicyNet
from relax.network.blocks_flax import Activation, ReprQNet, PolicyNet, PhiNetMLP, MuNetMLP
from relax.network.common import WithSquashedGaussianPolicy

import flax.linen as nn
from flax.training import train_state

class CTRLSACParams(NamedTuple):
    q1: dict
    q2: dict
    target_q1: dict
    target_q2: dict
    phi: dict
    mu: dict
    target_phi: dict
    policy: dict
    log_alpha: jax.Array


@dataclass
class CTRLSACNet(WithSquashedGaussianPolicy):
    q: Callable[[dict, jax.Array, jax.Array], jax.Array]
    target_entropy: float
    phi: Callable[[dict, jax.Array, jax.Array], jax.Array]
    mu: Callable[[dict, jax.Array], jax.Array]


def create_ctrl_sac_net(
    key: jax.Array,
    obs_dim: int,
    act_dim: int,
    repr_dim: int,
    hidden_sizes: Sequence[int],
    w_hidden_sizes: Sequence[int],
    activation: Activation = jax.nn.relu,
    w_activation: Activation = jax.nn.relu,
) -> Tuple[CTRLSACNet, CTRLSACParams]:
    q = ReprQNet(w_hidden_sizes, w_activation)
    policy = PolicyNet(act_dim, hidden_sizes, activation)
    phi = PhiNetMLP(hidden_sizes, repr_dim, activation)
    mu = MuNetMLP(hidden_sizes, repr_dim, activation)

    @jax.jit
    def init(key, obs, act, repr):
        q1_key, q2_key, policy_key, phi_key, mu_key = jax.random.split(key, 5)
        q1_params = q.init(q1_key, repr)['params']
        q2_params = q.init(q2_key, repr)['params']
        target_q1_params = q1_params
        target_q2_params = q2_params
        policy_params = policy.init(policy_key, obs)['params']
        phi_params = phi.init(phi_key, obs, act)['params']
        target_phi_params = phi_params
        mu_params = mu.init(mu_key, obs)['params']
        log_alpha = jnp.array(1.0, dtype=jnp.float32)
        return CTRLSACParams(q1_params, q2_params, target_q1_params, target_q2_params, phi_params, 
                             mu_params, target_phi_params, policy_params, log_alpha)

    sample_obs = jnp.zeros((1, obs_dim))
    sample_act = jnp.zeros((1, act_dim))
    sample_repr = jnp.zeros((1, repr_dim))
    params = init(key, sample_obs, sample_act, sample_repr)

    net = CTRLSACNet(policy=policy.apply, q=q.apply, phi=phi.apply, mu=mu.apply, target_entropy=-act_dim)
    return net, params

if __name__ == '__main__':
    ctrl_sac_net, params = create_ctrl_sac_net(jax.random.PRNGKey(0), 12, 1, 256, [128,128])
    print(type(params.q1))

