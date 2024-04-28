"""This file contains a bunch of reward functions that we use to generate reward labels for the offline dataset"""

import numpy as np

from dataset import STATE_IDX

# TODO: Rewards should all be R(s, a, s') where s is global state


# It is actually harder than expected to come up with reward functions given just the state
# Therefore, we should encode goals/objectives in the LLM context via plain english.
def boundary_reward(dataset, e_bounds, n_bounds):
    """Positive reward for remaining in bounds"""
    return boundary_done(dataset, e_bounds, n_bounds).astype(np.float32)

def boundary_done(dataset, e_bounds, n_bounds):
    return (
        (dataset["state"][...,STATE_IDX["e_pos"]] < e_bounds[0]) 
        |  (dataset["state"][...,STATE_IDX["e_pos"]] > e_bounds[1])
        | (dataset["state"][...,STATE_IDX["n_pos"]] < n_bounds[0])
        | (dataset["state"][...,STATE_IDX["n_pos"]] > n_bounds[1])
    )

def speed_reward(dataset):
    """Reward for going either slow or fast"""
    return np.linalg.norm(dataset["state"]["vel"], axis=-1)

def goal_pos_reward(dataset, goal_pos):
    """Reward for relative distance to goal"""
    return np.sum((dataset["state"][...,STATE_IDX["pos"]] - goal_pos) ** 2, axis=-1)

def relative_goal_pos_reward(dataset, goal_pos):
    """Reward for taking a step in the correct direction"""
    return (
        # Distance to goal before
        np.sum(np.abs(dataset["state"][...,STATE_IDX["pos"]] - goal_pos), axis=-1)
        # Distance to goal now
        - np.sum(np.abs(dataset["next_state"][...,STATE_IDX["pos"]] - goal_pos), axis=-1)
    )

def goal_pos_done(dataset, goal_pos, threshold=0.1):
    return goal_pos_reward(dataset, goal_pos) < threshold

def goal_vel_reward(dataset, goal_vel):
    """Reward for relative velocity to goal"""
    return np.sum((dataset["state"][...,STATE_IDX["vel"]] - goal_vel) ** 2, axis=-1)

def goal_vel_done(dataset, goal_vel, threshold=0.1):
    return goal_vel_reward(dataset, goal_vel) < threshold
