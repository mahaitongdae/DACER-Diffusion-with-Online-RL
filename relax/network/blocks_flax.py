from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional, Sequence, Tuple, Union

import flax.linen
import flax.linen
import jax, jax.numpy as jnp
import haiku as hk
from haiku.initializers import Constant

from relax.utils.jax_utils import fix_repr, is_broadcastable

Activation = Callable[[jax.Array], jax.Array]
Identity: Activation = lambda x: x
Tanh: Activation = lambda x: jnp.tanh(x)

import flax.linen as nn
import flax


@dataclass
class ValueNet(nn.Module):
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    name: str = None

    @nn.compact
    def __call__(self, obs: jax.Array) -> jax.Array:
        return mlp(self.hidden_sizes, 1, self.activation, self.output_activation, squeeze_output=True)(obs)


@dataclass
class QNet(nn.Module):
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    name: str = None

    @nn.compact
    def __call__(self, obs: jax.Array, act: jax.Array) -> jax.Array:
        input = jnp.concatenate((obs, act), axis=-1)
        return mlp(self.hidden_sizes, 1, self.activation, self.output_activation, squeeze_output=True)(input)
    
@dataclass
class ReprQNet(nn.Module):
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    name: str = None

    @nn.compact
    def __call__(self, repr: jax.Array) -> jax.Array:
        repr = flax.linen.LayerNorm()(repr)
        return mlp(self.hidden_sizes, 1, self.activation, self.output_activation, squeeze_output=True)(repr)
        # return nn.Dense(1, use_bias=False)(repr).squeeze()
    
@dataclass
class RandomMuNet(nn.Module):
    hidden_sizes: Sequence[int]
    repr_dim: int
    activation: Activation
    output_activation: Activation = nn.relu
    name: str = None
    sigma: float = 1.
    random_feature_dim: int = 4096
    out_normalized = True

    @nn.compact
    def __call__(self, obs: jax.Array):
        # Define Gaussian initialization (stddev=0.1)
        init_fn_kernel_fourier = nn.initializers.normal(stddev=self.sigma)
        init_fn_bias_fourier = nn.initializers.uniform(2 * jnp.pi)
        init_fn_kernel_rand_func = nn.initializers.normal(stddev=1.0)
        init_fn_bias_rand_func = nn.initializers.constant(0.)

        # Apply initialization to a Dense layer
        x = nn.Dense(features=self.random_feature_dim, 
                     kernel_init=init_fn_kernel_fourier, 
                     bias_init=init_fn_bias_fourier)(obs)
        x = jnp.cos(x)
        x = nn.Dense(features=self.repr_dim, 
                     kernel_init=init_fn_kernel_rand_func,
                     bias_init=init_fn_bias_rand_func)(x)  # Output layer
        if self.out_normalized:
            return x / jnp.sqrt(self.repr_dim)
        else:
            return x
    
@dataclass
class RFFQNet(nn.Module):
    hidden_dim: int
    activation: Activation
    output_activation: Activation = Identity
    name: str = None

    @nn.compact
    def __call__(self, repr: jax.Array) -> jax.Array:
        x = jnp.sin(nn.Dense(self.hidden_dim)(repr))
        x = nn.elu(nn.Dense(self.hidden_dim)(x))
        x = nn.Dense(1)(x)
        return x
    
@dataclass
class PhiNetMLP(nn.Module):
    hidden_sizes: Sequence[int]
    repr_dim: int
    activation: Activation
    output_activation: Activation = Identity
    name: str = None
    out_normalized = False

    @nn.compact
    def __call__(self, obs: jax.Array, act: jax.Array) -> jax.Array:
        input = jnp.concatenate((obs, act), axis=-1)
        # input = flax.linen.LayerNorm()(input)
        out = mlp(self.hidden_sizes, self.repr_dim, self.activation, self.output_activation, final_layer_bias=True)(input)
        if self.out_normalized:
            return out / jnp.sqrt(self.repr_dim)
        else:
            return out
    
@dataclass
class MuNetMLP(nn.Module):
    hidden_sizes: Sequence[int]
    repr_dim: int
    activation: Activation
    output_activation: Activation = nn.relu
    name: str = None

    @nn.compact
    def __call__(self, obs: jax.Array) -> jax.Array:
        obs = flax.linen.LayerNorm()(obs)
        out = mlp(self.hidden_sizes, self.repr_dim, self.activation, self.output_activation, final_layer_bias=True)(obs)
        return out # / jnp.sqrt(self.repr_dim)

@dataclass
class DistributionalQNet(nn.Module):
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    min_log_std: float = -0.1
    max_log_std: float = 4.0
    name: str = None

    @nn.compact
    def __call__(self, obs: jax.Array, act: jax.Array) -> Tuple[jax.Array, jax.Array]:
        input = jnp.concatenate((obs, act), axis=-1)
        value_mean = mlp(self.hidden_sizes, 1, self.activation, self.output_activation, squeeze_output=True)(input)
        value_log_std = mlp(self.hidden_sizes, 1, self.activation, self.output_activation, squeeze_output=True)(input)
        denominator = max(abs(self.min_log_std), abs(self.max_log_std))
        value_log_std = (
            jnp.maximum( self.max_log_std * jnp.tanh(value_log_std / denominator), 0.0) +
            jnp.minimum(-self.min_log_std * jnp.tanh(value_log_std / denominator), 0.0)
        )
        return value_mean, value_log_std

@dataclass
class DistributionalQNet2(nn.Module):
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    name: str = None

    def __call__(self, obs: jax.Array, act: jax.Array) -> Tuple[jax.Array, jax.Array]:
        input = jnp.concatenate((obs, act), axis=-1)
        output = mlp(self.hidden_sizes, 2, self.activation, self.output_activation)(input)
        value_mean = output[..., 0]
        value_std = jax.nn.softplus(output[..., 1])
        return value_mean, value_std


@dataclass
class PolicyNet(nn.Module):
    act_dim: int
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    min_log_std: float = -20.0
    max_log_std: float = 0.5
    log_std_mode: Union[str, float] = 'shared'  # shared, separate, global (provide initial value)
    name: str = None

    @nn.compact
    def __call__(self, obs: jax.Array, *, return_log_std: bool = False) -> jax.Array:
        if self.log_std_mode == 'shared':
            output = mlp(self.hidden_sizes, self.act_dim * 2, self.activation, self.output_activation)(obs)
            mean, log_std = jnp.split(output, 2, axis=-1)
        elif self.log_std_mode == 'separate':
            mean = mlp(self.hidden_sizes, self.act_dim, self.activation, self.output_activation)(obs)
            log_std = mlp(self.hidden_sizes, self.act_dim, self.activation, self.output_activation)(obs)
        else:
            initial_log_std = float(self.log_std_mode)
            mean = mlp(self.hidden_sizes, self.act_dim, self.activation, self.output_activation)(obs)
            log_std = self.param('log_std', nn.initializers.ones_init(), (self.act_dim,))
            log_std = jnp.broadcast_to(log_std, mean.shape)
        if not (self.min_log_std is None and self.max_log_std is None):
            log_std = jnp.clip(log_std, self.min_log_std, self.max_log_std)
        if return_log_std:
            return mean, log_std
        else:
            return mean, jnp.exp(log_std)

@dataclass
class PolicyStdNet(nn.Module):
    act_dim: int
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Tanh
    min_log_std: float = -5.0
    max_log_std: float = 2.0
    name: str = None

    def __call__(self, obs: jax.Array) -> jax.Array:
        log_std = mlp(self.hidden_sizes, self.act_dim, self.activation, self.output_activation)(obs)
        return self.min_log_std + (log_std + 1) / 2 * (self.max_log_std - self.min_log_std)


@dataclass
class DeterministicPolicyNet(nn.Module):
    act_dim: int
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    name: str = None

    def __call__(self, obs: jax.Array) -> jax.Array:
        return mlp(self.hidden_sizes, self.act_dim, self.activation, self.output_activation)(obs)


@dataclass
class ModelNet(nn.Module):
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    name: str = None

    def __call__(self, obs: jax.Array, act: jax.Array) -> jax.Array:
        obs_dim = obs.shape[-1]
        input = jnp.concatenate((obs, act), axis=-1)
        return mlp(self.hidden_sizes, obs_dim, self.activation, self.output_activation)(input)


@dataclass
class QScoreNet(nn.Module):
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    name: str = None

    @nn.compact
    def __call__(self, obs: jax.Array, act: jax.Array) -> jax.Array:
        act_dim = act.shape[-1]
        input = jnp.concatenate((obs, act), axis=-1)
        return mlp(self.hidden_sizes, act_dim, self.activation, self.output_activation)(input)


@dataclass
class DiffusionPolicyNet(nn.Module):
    time_dim: int
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    name: str = None

    def __call__(self, obs: jax.Array, act: jax.Array, t: jax.Array) -> jax.Array:
        act_dim = act.shape[-1]
        te = scaled_sinusoidal_encoding(t, dim=self.time_dim, batch_shape=obs.shape[:-1])
        input = jnp.concatenate((obs, act, te), axis=-1)
        return mlp(self.hidden_sizes, act_dim, self.activation, self.output_activation)(input)

@dataclass
class DACERPolicyNet(nn.Module):
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    time_dim: int = 16
    name: str = None

    @nn.compact
    def __call__(self, obs: jax.Array, act: jax.Array, t: jax.Array) -> jax.Array:
        act_dim = act.shape[-1]
        te = scaled_sinusoidal_encoding(t, dim=self.time_dim, batch_shape=obs.shape[:-1])
        te = nn.Dense(self.time_dim * 2)(te)
        te = self.activation(te)
        te = nn.Dense(self.time_dim)(te)
        input = jnp.concatenate((obs, act, te), axis=-1)
        return mlp(self.hidden_sizes, act_dim, self.activation, self.output_activation)(input)

def mlp(hidden_sizes: Sequence[int],
        output_size: int,
        activation: Activation,
        output_activation: Activation,
        *,
        squeeze_output: bool = False,
        final_layer_bias=True) -> Callable[[jax.Array], jax.Array]:
    layers = []
    if len(hidden_sizes) > 0:
        for hidden_size in hidden_sizes:
            layers += [nn.Dense(hidden_size, kernel_init=nn.initializers.orthogonal(),
                                bias_init=nn.initializers.constant(0.)), activation]
    layers += [nn.Dense(output_size, use_bias=final_layer_bias), output_activation]
    if squeeze_output:
        layers.append(partial(jnp.squeeze, axis=-1))
    return flax.linen.Sequential(layers)

def mlp_with_layer_norm(hidden_sizes: Sequence[int],
        output_size: int,
        activation: Activation,
        output_activation: Activation,
        *,
        squeeze_output: bool = False) -> Callable[[jax.Array], jax.Array]:
    hidden_sizes = list(hidden_sizes)
    first_hidden = hidden_sizes.pop(0)
    layers = [nn.Dense(first_hidden), nn.LayerNorm()]
    for hidden_size in hidden_sizes:
        layers += [nn.Dense(hidden_size), activation]
    layers += [nn.Dense(output_size), output_activation]
    if squeeze_output:
        layers.append(partial(jnp.squeeze, axis=-1))
    return flax.linen.Sequential(layers)


def scaled_sinusoidal_encoding(t: jax.Array, *, dim: int, theta: int = 10000, batch_shape = None) -> jax.Array:
    assert dim % 2 == 0
    if batch_shape is not None:
        assert is_broadcastable(jnp.shape(t), batch_shape)

    scale = 1 / dim ** 0.5
    half_dim = dim // 2
    freq_seq = jnp.arange(half_dim) / half_dim
    inv_freq = theta ** -freq_seq

    emb = jnp.einsum('..., j -> ... j', t, inv_freq)
    emb = jnp.concatenate((
        jnp.sin(emb),
        jnp.cos(emb),
    ), axis=-1)
    emb *= scale

    if batch_shape is not None:
        emb = jnp.broadcast_to(emb, (*batch_shape, dim))

    return emb

if __name__ == '__main__':
    print(mlp([128], 1, jax.nn.relu, Identity))
