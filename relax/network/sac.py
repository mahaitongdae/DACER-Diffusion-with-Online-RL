from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence, Tuple

import jax, jax.numpy as jnp
import haiku as hk

# from relax.network.blocks import Activation, QNet, PolicyNet
from relax.network.blocks_flax import Activation, QNet, PolicyNet
from relax.network.common import WithSquashedGaussianPolicy

import flax.linen as nn
from flax.training import train_state

class SACParams(NamedTuple):
    q1: dict
    q2: dict
    target_q1: dict
    target_q2: dict
    policy: dict
    log_alpha: jax.Array


@dataclass
class SACNet(WithSquashedGaussianPolicy):
    q: Callable[[dict, jax.Array, jax.Array], jax.Array]
    target_entropy: float


def create_sac_net(
    key: jax.Array,
    obs_dim: int,
    act_dim: int,
    hidden_sizes: Sequence[int],
    activation: Activation = jax.nn.relu,
) -> Tuple[SACNet, SACParams]:
    # q = hk.without_apply_rng(hk.transform(lambda obs, act: QNet(hidden_sizes, activation)(obs, act)))
    # policy = hk.without_apply_rng(hk.transform(lambda obs: PolicyNet(act_dim, hidden_sizes, activation)(obs)))
    q = QNet(hidden_sizes, activation)
    policy = PolicyNet(act_dim, hidden_sizes, activation)

    @jax.jit
    def init(key, obs, act):
        q1_key, q2_key, policy_key = jax.random.split(key, 3)
        q1_params = q.init(q1_key, obs, act)['params']
        q2_params = q.init(q2_key, obs, act)['params']
        target_q1_params = q1_params
        target_q2_params = q2_params
        policy_params = policy.init(policy_key, obs)['params']
        log_alpha = jnp.array(1.0, dtype=jnp.float32)
        return SACParams(q1_params, q2_params, target_q1_params, target_q2_params, policy_params, log_alpha)

    sample_obs = jnp.zeros((1, obs_dim))
    sample_act = jnp.zeros((1, act_dim))
    params = init(key, sample_obs, sample_act)

    net = SACNet(policy=policy.apply, q=q.apply, target_entropy=-act_dim)
    return net, params

if __name__ == '__main__':
    sac_net, params = create_sac_net(jax.random.PRNGKey(0), 12, 1, [128,128])
    print(type(params.q1))

