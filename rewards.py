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


import numpy as np
from numpy.linalg import norm


import numpy as np
from numpy.linalg import norm


import numpy as np
from numpy.linalg import norm

def segment_distance(starts1, ends1, starts2, ends2):
    # Calculate direction vectors for each segment
    dirs1 = ends1 - starts1
    dirs2 = ends2 - starts2
    
    # Prepare for broadcasting
    p = starts1[:, jnp.newaxis, :]
    q = starts2[jnp.newaxis, :, :]
    r = dirs1[:, jnp.newaxis, :]
    s = dirs2[jnp.newaxis, :, :]
    
    # Cross product of direction vectors in 2D
    r_cross_s = jnp.cross(r, s, axis=2)
    
    # Compute the vector between the starts of the line segments
    pq = q - p
    
    # Avoid divide by zero by replacing zeros with the smallest positive float
    safe_denominator = jnp.where(r_cross_s == 0, jnp.finfo(float).eps, r_cross_s)
    
    # Distance calculation based on the cross product in 2D
    distance = jnp.abs(jnp.cross(pq, r, axis=2) / safe_denominator)

    # Clip the distances calculated to the length of each segment
    projected_length1 = jnp.sum(pq * r, axis=2) / jnp.sum(r * r, axis=2)
    projected_length2 = jnp.sum(pq * s, axis=2) / jnp.sum(s * s, axis=2)
    projected_length1_clipped = jnp.clip(projected_length1, 0, 1)
    projected_length2_clipped = jnp.clip(projected_length2, 0, 1)

    closest_p = p + projected_length1_clipped[..., jnp.newaxis] * r
    closest_q = q + projected_length2_clipped[..., jnp.newaxis] * s

    final_distances = jnp.linalg.norm(closest_p - closest_q, axis=2)
    
    return final_distances



def segment_collision(state, next_state, safe_radius):
    segment_a_start = state[:, :2]
    segment_a_end = next_state[:, :2]
    segment_b_start = state[:, :2]
    segment_b_end = next_state[:, :2]
    res = segment_distance(segment_a_start, segment_a_end, segment_b_start, segment_b_end)
    collisions = (res < safe_radius).sum(-1)
    return collisions
    
     

def ma_collision_done(dataset, safe_radius=jnp.array(0.3)):
    return jnp.sum(fast_pairwise_distances(dataset['state'][:, STATE_IDX["pos"]]) < safe_radius, axis=0).astype(bool)

# TODO: We need to randomly sample for MA
# but these rewards must be computed AFTER sampling
def ma_collision_reward(dataset, safe_radius=0.3):
    # Dataset shape: [agent, *]
    # Reward shape: [agent, *]
    #B, T, A, F = dataset['state'].shape
    #state_in = dataset['state'].reshape(B, A, F)
    return jnp.sum(fast_pairwise_distances(dataset['state'][:, STATE_IDX["pos"]]) < safe_radius, axis=0).astype(jnp.float32) / dataset['state'].shape[0]

def ma_collision_reward_and_done(state, next_state, reward, done, safe_radius=jnp.array(0.3)):
    overlap = jnp.expand_dims(jnp.sum(fast_pairwise_distances(state[:, STATE_IDX["pos"]]) < safe_radius, axis=0), 1)
    collisions = jnp.expand_dims(segment_collision(state, next_state, safe_radius), 1)
    # TODO: We need to draw lines and see if the lines intersect
    # the policy is abusing the 1s timesteps
    return (
        reward 
        - 0.5 * collisions.astype(jnp.float32) # Prevent collisions
        - 0.5 * overlap.astype(jnp.float32), # Prevent overlapping
        done 
#        | collisions.astype(bool)
#        | overlap.astype(bool) 
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
    return np.linalg.norm(
        dataset["state"][...,STATE_IDX["vel"]] + dataset["next_state"][...,STATE_IDX["vel"]], 
    axis=-1)

def goal_pos_reward(dataset, goal_pos):
    """Reward for relative distance to goal"""
    return np.sum((dataset["state"][...,STATE_IDX["pos"]] - goal_pos) ** 2, axis=-1)

# TODO: Check this indexing is correct...
def relative_goal_pos_reward(dataset, goal_pos):
    """Reward for taking a step in the correct direction"""
    return (
        # Distance to goal before
        np.sum(np.linalg.norm(dataset["state"][...,STATE_IDX["pos"]] - goal_pos, ord=2, axis=-1, keepdims=True), axis=-1)
        # Distance to goal now
        - np.sum(np.linalg.norm(dataset["next_state"][...,STATE_IDX["pos"]] - goal_pos, ord=2, axis=-1, keepdims=True), axis=-1)
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
