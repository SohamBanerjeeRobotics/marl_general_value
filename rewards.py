"""This file contains a bunch of reward functions that we use to generate reward labels for the offline dataset"""

import numpy as np
import jax
import jax.numpy as jnp

from constants import ARENA_BOUNDS_E, ARENA_BOUNDS_N, STATE_IDX, ROBOT_DIAMETER, VELOCITY

POSITION_SCALE = np.linalg.norm(np.array([ARENA_BOUNDS_N, ARENA_BOUNDS_E])[:,0] - np.array([ARENA_BOUNDS_N, ARENA_BOUNDS_E])[:,1])
VELOCITY_SCALE = 2 * VELOCITY

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


def segment_distance_dumb(start1, end1, start2, end2):
    l0 = jnp.stack([
        jnp.linspace(start1[0], end1[0], 50),
        jnp.linspace(start1[1], end1[1], 50)
    ], axis=-1)
    l1 = jnp.stack([
        jnp.linspace(start2[0], end2[0], 50),
        jnp.linspace(start2[1], end2[1], 50)
    ], axis=-1)

    l0 = jnp.repeat(l0[:, None, :], l0.shape[0], axis=1)
    l1 = jnp.repeat(l1[:, None, :], l0.shape[0], axis=1).transpose(1, 0, 2)
    return jnp.linalg.norm(l0 - l1, axis=-1).min()

def test_segment_distance():
    # Intersection, distance should be 0
    start0 = jnp.array([0, 0])
    end0 = jnp.array([2, 0])

    start1 = jnp.array([1, 1])
    end1 = jnp.array([1, -1])

    # Distance should be exactly 1
    start2 = jnp.array([2, 1])
    end2 = jnp.array([3, 3])

    # Distance should be exactly 1
    start3 = jnp.array([-1, 1])
    end3 = jnp.array([-1, -1])

    # Overlap, distance should be 0
    start4 = jnp.array([0, 0])
    end4 = jnp.array([2, 0])

    # Reverse overlap, distance should be 0
    start5 = jnp.array([2, 0])
    end5 = jnp.array([0, 0])


    d0 = segment_distance_dumb(start0, end0, start1, end1)
    d1 = segment_distance_dumb(start0, end0, start2, end2)
    d2 = segment_distance_dumb(start0, end0, start3, end3)
    d3 = segment_distance_dumb(start0, end0, start4, end4)
    d4 = segment_distance_dumb(start0, end0, start5, end5)
    breakpoint()
    assert d0 == 0
    assert d1 == 1
    assert d2 == 1

def test_segment_collision_real():
    starts = jnp.array([
       [-1.10234327, -1.23581204],
       [-0.71478854,  0.96816025],
       [ 0.67988497,  1.20852254]
    ])
    ends = jnp.array([
      [-1.10234327, -1.23581204],
      [-0.71478854,  0.96816025],
      [ 0.97988499,  1.20852254]
    ])
    
    res = batched_segment_distance(starts, ends)
    breakpoint()

def test_segment_distance_real():
    # Two points with no motion
    starts = jnp.array([
       [-1.10234327, -1.23581204],
       [-0.71478854,  0.96816025],
    ])
    
    res = segment_distance_dumb(starts[0], starts[0], starts[1], starts[1])
    breakpoint()

def test_segment_collision():
    start0 = jnp.array([0, 0])
    end0 = jnp.array([2, 0])

    # Intersection, distance should be 0
    start1 = jnp.array([1, 1])
    end1 = jnp.array([1, -1])

    # Distance should be exactly 1
    start2 = jnp.array([2, 1])
    end2 = jnp.array([3, 3])

    # Distance should be exactly 1
    start3 = jnp.array([-1, 1])
    end3 = jnp.array([-1, -1])

    state = jnp.stack([start0, start1, start2, start3])
    next_state = jnp.stack([end0, end1, end2, end3])
    # Extra zero (collision) for self-distance
    res = segment_collision(state, next_state, 1.0)
    breakpoint()

def batched_segment_distance(state, next_state):
    # Shape [A, 2]
    A = state.shape[0]
    segment_a_start = jnp.repeat(state[:, None, :2], A, axis=1)
    segment_a_end = jnp.repeat(next_state[:, None, :2], A, axis=1)
    segment_b_start = segment_a_start.transpose(1, 0, 2)
    segment_b_end = segment_a_end.transpose(1, 0, 2)
    # Identity will always be zero
    res = jax.vmap(jax.vmap(segment_distance_dumb))(segment_a_start, segment_a_end, segment_b_start, segment_b_end)
    return res

def segment_collision(state, next_state, radius):
    # Shape [A, 2]
    dists = batched_segment_distance(state, next_state)
    # Subtract 1 because there will always be a self collision
    collisions = (dists < radius).sum(-1) - 1.0
    return collisions
    
def ma_collision_reward_and_done(state, next_state, reward, done):
    collisions = jnp.expand_dims(segment_collision(state, next_state, ROBOT_DIAMETER), 1)
    # TODO: We need to draw lines and see if the lines intersect
    # the policy is abusing the 1s timesteps
    return (
        reward - collisions.astype(jnp.float32),
        done | collisions.astype(bool)
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
        | (dataset["state"][...,STATE_IDX["e_pos"]] < e_bounds[0]) 
        | (dataset["state"][...,STATE_IDX["e_pos"]] > e_bounds[1])
        | (dataset["state"][...,STATE_IDX["n_pos"]] < n_bounds[0])
        | (dataset["state"][...,STATE_IDX["n_pos"]] > n_bounds[1])
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

def relative_radius_reward(dataset, goal_radius=1.0):
    return np.sum(np.linalg.norm(dataset["state"][...,STATE_IDX["pos"]], ord=2, axis=-1, keepdims=True) - goal_radius, axis=-1)

def line_reward(dataset, goal):
    return np.sum(np.linalg.norm(dataset['state'][...,STATE_IDX['pos']] * goal, axis=-1, keepdims=True), axis=-1)
    

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

if __name__ == '__main__':
    test_segment_distance()
    #test_segment_collision()
    #test_segment_collision_real()
    #test_segment_distance_real()
