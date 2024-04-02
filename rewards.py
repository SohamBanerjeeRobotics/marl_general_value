"""This file contains a bunch of reward functions that we use to generate reward labels for the offline dataset"""

import jax.numpy as jnp
import jax

from dataset import STATE_IDX

# TODO: Rewards should all be R(s, a, s') where s is global state


# It is actually harder than expected to come up with reward functions given just the state
# Therefore, we should encode goals/objectives in the LLM context via plain english.
def boundary_reward(dataset, x_bounds, y_bounds):
    """Positive reward for remaining in bounds"""
    return (
        (x_bounds[0] < dataset["state"][STATE_IDX["pos_x"]] < x_bounds[1]).astype(jnp.float32) + 
        (y_bounds[0] < dataset["state"][STATE_IDX["pos_y"]] < y_bounds[1]).astype(jnp.float32)
    )

def speed_reward(dataset, sign=1.0):
    """Reward for going either slow or fast"""
    return jnp.sign(sign) * jnp.linalg.norm(dataset["state"]["vel"], axis=-1)

def goal_pos_reward(dataset, goal_pos):
    """Reward for relative distance to goal"""
    return jnp.sum((dataset["state"][...,STATE_IDX["pos"]] - goal_pos) ** 2, axis=-1)

def goal_vel_reward(dataset, goal_vel):
    """Reward for relative velocity to goal"""
    return jnp.sum((dataset["state"][...,STATE_IDX["vel"]] - goal_vel) ** 2, axis=-1)
