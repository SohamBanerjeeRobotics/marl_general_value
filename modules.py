from typing import Any, Dict
import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
from equinox import nn
import numpy as np
import math

def leaky_relu(x, key=None):
    return jax.nn.leaky_relu(x)

def default_init(key, linear, scale=1.0, zero_bias=False, fixed_bias=None):
    """Default init used in pytorch"""
    lim = math.sqrt(scale / linear.in_features)
    linear = eqx.tree_at(lambda l: l.weight, linear, jax.random.uniform(key, linear.weight.shape, minval=-lim, maxval=lim))
    if zero_bias:
        linear = eqx.tree_at(lambda l: l.bias, linear, jnp.zeros_like(linear.bias))
    elif fixed_bias is not None:
        linear = eqx.tree_at(lambda l: l.bias, linear, jnp.full_like(linear.bias, fixed_bias))
    return linear

def final_linear(key, input_size, output_size, scale=0.01):
    """a nn.Linear layer with initialization for the final layer of a value function"""
    #linear = ortho_linear(key, input_size, output_size, scale=scale)
    linear = nn.Linear(input_size, output_size, key=key)
    linear = default_init(key, linear, scale=scale, zero_bias=True)
    return linear

class Block(eqx.Module):
    """A standard nn layer with linear, norm, and activation."""
    net: eqx.Module
    def __init__(self, input_size, output_size, dropout, key):
        if dropout == 0.0:
            self.net = RandomSequential([
                nn.Linear(input_size, output_size, key=key), 
                nn.LayerNorm(output_size, use_weight=False, use_bias=False),
                leaky_relu,
            ])
        else:
            self.net = RandomSequential([
                nn.Linear(input_size, output_size, key=key), 
                nn.LayerNorm(output_size, use_weight=False, use_bias=False),
                nn.Dropout(dropout),
                leaky_relu,
            ])

    def __call__(self, x, key=None):
        return self.net(x, key=key)

class RandomSequential(nn.Sequential):
    """A nn.Sequential layer that passes through random keys"""
    def __call__(self, x, key=None):
        return super().__call__(x, key=key)

class QHead(eqx.Module):
    post0: eqx.Module
    post1: eqx.Module
    value: nn.Linear
    advantage: nn.Linear

    def __init__(self, input_size, hidden_size, output_size, dropout, key):
        keys = random.split(key, 3)

        self.post0 = Block(input_size, hidden_size, dropout, keys[0])
        self.post1 = Block(hidden_size, hidden_size, dropout, keys[1])
        self.value = final_linear(keys[2], input_size, 1, scale=0.01)
        self.advantage = final_linear(keys[3], input_size, output_size, scale=0.01)

    def __call__(self, x, key):
        T = x.shape[0]
        net_keys = random.split(key, 2)
        x = self.post0(x, net_keys[0])
        x = self.post1(x, net_keys[1])
        V = self.value(x) 
        A = self.advantage(x)
        # Dueling DQN
        return V + (A - A.mean(keepdims=True))

class QNetwork(eqx.Module):
    """Single agent Q network for DDPG"""
    observation_size: int
    action_size: int
    hidden_size: int
    mlp: nn.Sequential

    def __init__(self, obs_size, action_size, key):
        self.observation_size = obs_size#8 * num_agents
        self.hidden_size = 256
        self.action_size = action_size
        keys = jax.random.split(key, 3)
        self.mlp = nn.Sequential([
            nn.Linear(self.observation_size + self.action_size, self.hidden_size, key=keys[0]),
            nn.LayerNorm((self.hidden_size,)),
            leaky_relu,
            nn.Linear(self.hidden_size, self.hidden_size, key=keys[1]),
            nn.LayerNorm((self.hidden_size,)),
            leaky_relu,
            final_linear(keys[2], self.hidden_size, 1, scale=0.01)
        ])

    def __call__(self, state, action, key=None):
        x = jnp.concatenate([state, action], axis=-1)
        values = self.mlp(x)
        return values


class Policy(eqx.Module):
    """Single agent policy for DDPG"""
    observation_size: int
    hidden_size: int
    action_low: np.array
    action_high: np.array
    mlp: nn.Sequential

    def __init__(self, obs_size, action_low, action_high, key):
        keys = jax.random.split(key, 3)
        self.observation_size = obs_size
        self.action_low = action_low
        self.action_high = action_high
        self.hidden_size = 256
        self.mlp = nn.Sequential([
            nn.Linear(self.observation_size, self.hidden_size, key=keys[0]),
            nn.LayerNorm((self.hidden_size,)),
            leaky_relu,
            nn.Linear(self.hidden_size, self.hidden_size, key=keys[1]),
            nn.LayerNorm((self.hidden_size,)),
            leaky_relu,
            nn.Linear(self.hidden_size, self.action_low.size, key=keys[2]),
        ])

    def __call__(self, state, noise_scale, key):
        x = self.mlp(state)
        #mu, sigma = jnp.split(x, 2, axis=-1)
        noise = noise_scale * jax.random.normal(key, shape=(self.action_low.size,))
        unclamped_action = self.mlp(state) + noise * noise_scale
        # Ensure in action space
        scale = (self.action_high - self.action_low) / 2
        shift = self.action_low + self.action_high
        action = scale * jnp.tanh(unclamped_action) + shift
        return action


class EnsembleQNetwork(eqx.Module):
    """The core model used in experiments.
    
    This is a discrete Q function with a shared trunk and multiple ensemble
    heads. The ensemble dimension output is along axis -2.
    """
    input_size: int
    output_size: int
    config: Dict[str, Any]
    torso0: Block
    torso1: Block
    q: eqx.Module

    def __init__(self, obs_shape, act_shape, config, key):
        self.config = config
        self.output_size = act_shape
        keys = random.split(key, 2)
        [self.input_size] = obs_shape
        self.torso0 = Block(self.input_size, config["mlp_size"], 0, keys[0])
        self.torso1 = Block(config["mlp_size"], config["mlp_size"], 0, keys[1])

        ensemble_keys = random.split(keys[1], config["ensemble_size"])

        @eqx.filter_vmap
        def make_heads(key):
            return QHead(config["head_size"], config["mlp_size"], act_shape, config["dropout"], key)
                    
        self.q = make_heads(ensemble_keys)


    def __call__(self, x, key):
        """Returns an ensemble of Q values of shape [ensemble, actions]"""
        T = x.shape[0]
        net_keys = random.split(key, 2 * T + self.config["ensemble_size"])
        x = self.torso0(x, net_keys[:T])
        x = self.torso1(x, net_keys[T:2 * T])

        @eqx.filter_vmap(in_axes=(eqx.if_array(0), None, 0))
        def ensemble(model, x, key):
            return model(x, key=key)

        #q = eqx.filter_vmap(self.q, in_axes=(eqx.if_array(0), None, None))
        q = ensemble(self.q, x, net_keys[2 * T:])
        return q

class GeneralQNetwork(eqx.Module):
    config: Dict[str, Any]
    torso0: Block
    torso1: Block
    q: eqx.Module

    def __init__(self, obs_size, task_size, act_size, config, key):
        self.config = config
        keys = random.split(key, 3)
        self.torso0 = Block(obs_size + task_size, config["mlp_size"], 0, keys[0])
        self.torso1 = Block(config["mlp_size"], config["mlp_size"], 0, keys[1])

        self.q = QHead(config["head_size"], config["mlp_size"], act_size, config["dropout"], keys[2])
                    
    def __call__(self, x, task, key):
        """Returns an ensemble of Q values of shape [ensemble, actions]"""
        # Expects x to be of shape [S]
        net_keys = random.split(key, 3)
        x = jnp.concatenate([x, task])
        x = self.torso0(x, net_keys[0])
        x = self.torso1(x, net_keys[1])

        q = self.q(x, net_keys[2])
        return q



def greedy_policy(
    q_network, x, key=None
):
    # Expand for ensemble
    q_values = q_network(jnp.expand_dims(x, 0), key=key)
    action = jnp.argmax(q_values)
    return action