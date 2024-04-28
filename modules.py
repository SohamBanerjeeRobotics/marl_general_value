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
        self.value = final_linear(keys[2], hidden_size, 1, scale=0.01)
        self.advantage = final_linear(keys[3], hidden_size, output_size, scale=0.01)

    def __call__(self, x, key):
        T = x.shape[0]
        net_keys = random.split(key, 2)
        x = self.post0(x, net_keys[0])
        x = self.post1(x, net_keys[1])
        V = self.value(x) 
        A = self.advantage(x)
        # Dueling DQN
        return V + (A - A.mean(keepdims=True))


class GeneralQNetwork(eqx.Module):
    config: Dict[str, Any]
    q: eqx.Module
    torso0: eqx.Module
    torso1: eqx.Module

    def __init__(self, obs_size, task_size, act_size, config, key):
        self.config = config
        keys = random.split(key, 3)
        self.torso0 = Block(obs_size + task_size, config["mlp_size"], 0, keys[0])
        self.torso1 = Block(config["mlp_size"], config["mlp_size"], 0, keys[1])

        self.q = QHead(obs_size + task_size, config["head_size"], act_size, config["dropout"], keys[2])
                    
    def __call__(self, x, task, key):
        """Returns an ensemble of Q values of shape [ensemble, actions]"""
        assert x.ndim == 1 and task.ndim == 1, "x dim: {}, task dim: {}".format(x.shape, task.shape)
        # Expects x to be of shape [S]
        net_keys = random.split(key, 3)
        x = jnp.concatenate([x, task])
        #x = self.torso0(x, net_keys[0])
        #x = self.torso1(x, net_keys[1])

        q = self.q(x, net_keys[2])
        return q



def greedy_policy(
    q_network, x, task, key=None
):
    # Expand for ensemble
    q_values = q_network(x, task, key=key)
    action = jnp.argmax(q_values)
    return action

def epsilon_greedy_policy(
    q_network, x, task, epsilon, key=None,
):
    q_values = q_network(x, task, key=key)
    rand_action = random.randint(key, (1,), 0, q_values.shape[0])
    mask = random.uniform(key) < epsilon
    action = rand_action * mask + jnp.argmax(q_values) * ~mask
    return action.squeeze(0)
