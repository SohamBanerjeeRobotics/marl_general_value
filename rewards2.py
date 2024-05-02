"""This file contains a bunch of reward functions that we use to generate reward labels for the offline dataset"""

import jax.numpy as jnp

from constants import ARENA_BOUNDS_E, ARENA_BOUNDS_N, STATE_IDX

# TODO: Rewards should all be R(s, a, s') where s is global state


# It is actually harder than expected to come up with reward functions given just the state
# Therefore, we should encode goals/objectives in the LLM context via plain english.
def boundary_reward(dataset, e_bounds, n_bounds):
    """Positive reward for remaining in bounds"""
    return boundary_done(dataset, e_bounds, n_bounds).astype(jnp.float32)

def boundary_done(dataset, e_bounds, n_bounds):
    assert dataset["state"].ndim == 1
    return (
        (dataset["state"][STATE_IDX["e_pos"]] < e_bounds[0]) 
        |  (dataset["state"][STATE_IDX["e_pos"]] > e_bounds[1])
        | (dataset["state"][STATE_IDX["n_pos"]] < n_bounds[0])
        | (dataset["state"][STATE_IDX["n_pos"]] > n_bounds[1])
    )

def goal_pos_reward(dataset, goal):
    """Reward for relative distance to goal"""
    assert dataset["state"].ndim == 1
    return jnp.sum(jnp.linalg.norm(dataset["state"][STATE_IDX["pos"]] - goal))

def relative_goal_pos_reward(dataset, goal):
    """Reward for taking a step in the correct direction"""
    assert dataset["state"].ndim == 1
    return jnp.sum(
        # Distance to goal before
        jnp.linalg.norm(dataset["state"][STATE_IDX["pos"]] - goal)
        # Distance to goal now
        - jnp.linalg.norm(dataset["next_state"][STATE_IDX["pos"]] - goal)
    )

def goal_pos_done(dataset, goal_pos, threshold=0.1):
    return goal_pos_reward(dataset, goal_pos) < threshold

def point_navigation_reward(dataset, goal):
    return (
        relative_goal_pos_reward(dataset, goal) 
        - 2 * boundary_reward(dataset, ARENA_BOUNDS_E, ARENA_BOUNDS_N).squeeze(-1)
    )

def point_navigation_done(dataset, goal, threshold=0.1):
    return (
        boundary_done(dataset, ARENA_BOUNDS_E, ARENA_BOUNDS_N).squeeze(-1)
    )