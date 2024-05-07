"""This file contains a bunch of reward functions that we use to generate reward labels for the offline dataset"""

import numpy as np
import jax
import jax.numpy as jnp

from constants import STATE_IDX

# TODO: Rewards should all be R(s, a, s') where s is global state
def pairwise_distances(A):
    assert A.ndim == 2
    # Compute the squared norms of each row in A
    norm = jnp.sum(jnp.square(A), axis=1)
    # Compute the squared distances matrix using the formula
    distances = norm[:, None] + norm[None, :] - 2 * jnp.dot(A, A.T)
    # Since due to numerical issues, small negative numbers could appear, we use maximum to avoid NaNs in sqrt
    distances = jnp.maximum(distances, 0.0)
    # Take the square root to get the actual distances
    distances = jnp.sqrt(distances)
    # Do not compare distance to self
    distances = distances + 100 * jnp.eye(A.shape[0])

    return distances

def fast_pairwise_distances(A):
    assert A.ndim == 2
    return jnp.linalg.norm(A[:, None] - A[None, :], axis=-1) + 100 * jnp.eye(A.shape[0])

def ma_collision_done(dataset, safe_radius=0.3):
    return jnp.sum(fast_pairwise_distances(dataset['state'][:, STATE_IDX["pos"]]) < safe_radius, axis=0).astype(bool)

# TODO: We need to randomly sample for MA
# but these rewards must be computed AFTER sampling
def ma_collision_reward(dataset, safe_radius=0.3):
    # Dataset shape: [agent, *]
    # Reward shape: [agent, *]
    #B, T, A, F = dataset['state'].shape
    #state_in = dataset['state'].reshape(B, A, F)
    return jnp.sum(fast_pairwise_distances(dataset['state'][:, STATE_IDX["pos"]]) < safe_radius, axis=0).astype(jnp.float32) / dataset['state'].shape[0]

def ma_collision_reward_and_done(state, reward, done, safe_radius=jnp.array(0.3), scale=jnp.array(1.0)):
    collisions = jnp.expand_dims(jnp.sum(fast_pairwise_distances(state[:, STATE_IDX["pos"]]) < safe_radius, axis=0), 1)
    # TODO: We need to draw lines and see if the lines intersect
    # the policy is abusing the 1s timesteps
    return (
        reward - collisions.astype(jnp.float32) / state.shape[0] * scale, 
        done #| collisions.astype(bool)
    )


# It is actually harder than expected to come up with reward functions given just the state
# Therefore, we should encode goals/objectives in the LLM context via plain english.
def boundary_reward(dataset, e_bounds, n_bounds):
    """Positive reward for remaining in bounds"""
    return boundary_done(dataset, e_bounds, n_bounds).astype(np.float32)

def boundary_done(dataset, e_bounds, n_bounds):
    return (
        (dataset["next_state"][...,STATE_IDX["e_pos"]] < e_bounds[0]) 
        |  (dataset["next_state"][...,STATE_IDX["e_pos"]] > e_bounds[1])
        | (dataset["next_state"][...,STATE_IDX["n_pos"]] < n_bounds[0])
        | (dataset["next_state"][...,STATE_IDX["n_pos"]] > n_bounds[1])
    )

def speed_reward(dataset):
    """Reward for going either slow or fast"""
    return np.linalg.norm(dataset["state"]["vel"], axis=-1)

def goal_pos_reward(dataset, goal_pos):
    """Reward for relative distance to goal"""
    return np.sum((dataset["state"][...,STATE_IDX["pos"]] - goal_pos) ** 2, axis=-1)

# TODO: Check this indexing is correct...
def relative_goal_pos_reward(dataset, goal_pos):
    """Reward for taking a step in the correct direction"""
    return (
        # Distance to goal before
        np.sum(np.linalg.norm(dataset["state"][...,STATE_IDX["pos"]] - goal_pos, ord=1, axis=-1, keepdims=True), axis=-1)
        # Distance to goal now
        - np.sum(np.linalg.norm(dataset["next_state"][...,STATE_IDX["pos"]] - goal_pos, ord=1, axis=-1, keepdims=True), axis=-1)
    )

def goal_pos_done(dataset, goal_pos, threshold=0.1):
    #return goal_pos_reward(dataset, goal_pos) < threshold
    return (
        (np.sum((dataset["state"][...,STATE_IDX["pos"]] - goal_pos) ** 2, axis=-1) < threshold)
        & (np.sum((dataset["next_state"][...,STATE_IDX["pos"]] - goal_pos) ** 2, axis=-1) < threshold)
    )

def goal_vel_reward(dataset, goal_vel):
    """Reward for relative velocity to goal"""
    return np.sum((dataset["state"][...,STATE_IDX["vel"]] - goal_vel) ** 2, axis=-1)

def goal_vel_done(dataset, goal_vel, threshold=0.1):
    return goal_vel_reward(dataset, goal_vel) < threshold
