from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence, Tuple

import jax, jax.numpy as jnp

# from relax.network.blocks import Activation, QNet, PolicyNet
from relax.network.blocks_flax import Activation, ReprQNet, PolicyNet, PhiNetMLP, RandomMuNet, QNet
from relax.network.common import WithSquashedGaussianPolicy

import flax.linen as nn
from flax.training import train_state

class RANDSACParams(NamedTuple):
    q1: dict
    q2: dict
    target_q1: dict
    target_q2: dict
    reward_pred: dict
    phi: dict
    mu: dict
    target_phi: dict
    policy: dict
    log_alpha: jax.Array


@dataclass
class RANDSACNet(WithSquashedGaussianPolicy):
    q: Callable[[dict, jax.Array, jax.Array], jax.Array]
    target_entropy: float
    phi: Callable[[dict, jax.Array, jax.Array], jax.Array]
    mu: Callable[[dict, jax.Array], jax.Array]
    reward_pred: Callable[[dict, jax.Array, jax.Array], jax.Array]
    
    def get_phi(self, key: jax.Array, phi_params: dict, obs: jax.Array, action: jax.Array) -> jax.Array:
            """for data collection"""
            return self.phi({'params': phi_params}, obs, action)


def create_rand_sac_net(
    key: jax.Array,
    obs_dim: int,
    act_dim: int,
    repr_dim: int,
    hidden_sizes: Sequence[int],
    w_hidden_sizes: Sequence[int],
    policy_hidden_sizes: Sequence[int] | None = None,
    activation: Activation = jax.nn.relu,
    w_activation: Activation = jax.nn.relu,
    mu_random_feature_dim: int = 4096,
    mu_random_feature_sigma: float = 1.,
) -> Tuple[RANDSACNet, RANDSACParams]:
    q = ReprQNet(w_hidden_sizes, w_activation)
    policy_hidden_sizes = hidden_sizes if policy_hidden_sizes is None else policy_hidden_sizes
    reward_pred = QNet(hidden_sizes, activation)
    policy = PolicyNet(act_dim, policy_hidden_sizes, activation)
    phi = PhiNetMLP(hidden_sizes, repr_dim, activation, output_activation=lambda x: x)
    mu = RandomMuNet(hidden_sizes, repr_dim, activation, 
                     sigma=mu_random_feature_sigma,
                     random_feature_dim=mu_random_feature_dim)

    @jax.jit
    def init(key, obs, act, repr):
        q1_key, q2_key, policy_key, phi_key, mu_key, reward_pred_key = jax.random.split(key, 6)
        q1_params = q.init(q1_key, repr)['params']
        q2_params = q.init(q2_key, repr)['params']
        target_q1_params = q1_params
        target_q2_params = q2_params
        policy_params = policy.init(policy_key, obs)['params']
        phi_params = phi.init(phi_key, obs, act)['params']
        target_phi_params = phi_params
        mu_params = mu.init(mu_key, obs)['params']
        reward_pred_params = reward_pred.init(reward_pred_key, obs, act)['params']
        log_alpha = jnp.array(1.0, dtype=jnp.float32)
        return RANDSACParams(q1_params, q2_params, target_q1_params, target_q2_params, reward_pred_params, phi_params, 
                             mu_params, target_phi_params, policy_params, log_alpha)

    sample_obs = jnp.zeros((1, obs_dim))
    sample_act = jnp.zeros((1, act_dim))
    sample_repr = jnp.zeros((1, repr_dim))
    params = init(key, sample_obs, sample_act, sample_repr)

    net = RANDSACNet(policy=policy.apply, q=q.apply, phi=phi.apply, mu=mu.apply, reward_pred=reward_pred.apply, target_entropy=-act_dim)
    return net, params

if __name__ == '__main__':
    import pandas as pd
    ctrl_sac_net, params = create_rand_sac_net(jax.random.PRNGKey(0), 12, 1, 256, [128,128], [])
    param_shapes = []
    for layer, params in params.reward_pred.items():
        for param_name, value in params.items():
            param_shapes.append((layer, param_name, value.shape))

    # Convert to DataFrame
    df = pd.DataFrame(param_shapes, columns=["Layer", "Parameter", "Shape"])
    import ace_tools_open as tools
    tools.display_dataframe_to_user(name="Flax Parameter Shapes", dataframe=df)

