# assignment.py

import jax.numpy as jnp
import numpy as np
from scipy.optimize import linear_sum_assignment

def compute_cost_matrix(agent_positions, task_goals):
    """
    Compute pairwise distances between agents and tasks.
    
    Args:
        agent_positions: [num_agents, 2] - current positions (n_pos, e_pos)
        task_goals: [num_tasks, 2] - goal positions (n_pos, e_pos)
    
    Returns:
        cost_matrix: [num_agents, num_tasks] - Euclidean distances
    """
    # Expand for broadcasting: [A, 1, 2] - [1, T, 2] → [A, T, 2]
    agents_expanded = jnp.expand_dims(agent_positions, axis=1)
    tasks_expanded = jnp.expand_dims(task_goals, axis=0)
    
    # Compute Euclidean distances: [A, T]
    distances = jnp.linalg.norm(agents_expanded - tasks_expanded, axis=-1)
    
    return distances


def hungarian_assignment(cost_matrix):
    """
    Solve optimal assignment problem using Hungarian algorithm.
    Minimizes total cost (distance).
    
    Args:
        cost_matrix: [num_agents, num_tasks]
    
    Returns:
        assignments: [num_agents] - task ID for each agent (-1 if unassigned)
        total_cost: scalar - total distance
    """
    cost_matrix_np = np.array(cost_matrix)
    
    # Solve: returns (agent_indices, task_indices)
    agent_indices, task_indices = linear_sum_assignment(cost_matrix_np)
    
    # Create assignment array
    num_agents = cost_matrix.shape[0]
    assignments = jnp.full(num_agents, -1, dtype=jnp.int32)
    
    for agent_idx, task_idx in zip(agent_indices, task_indices):
        assignments = assignments.at[agent_idx].set(task_idx)
    
    # Calculate total cost
    total_cost = cost_matrix_np[agent_indices, task_indices].sum()
    
    return assignments, total_cost


def greedy_nearest_neighbor(cost_matrix):
    """
    Greedy assignment: each agent takes nearest unassigned task.
    
    Args:
        cost_matrix: [num_agents, num_tasks]
    
    Returns:
        assignments: [num_agents]
    """
    num_agents = cost_matrix.shape[0]
    assignments = []
    used_tasks = set()
    
    for agent_id in range(num_agents):
        costs = list(cost_matrix[agent_id])
        best_task = -1
        best_cost = float('inf')
        
        for task_id, cost in enumerate(costs):
            if task_id not in used_tasks and cost < best_cost:
                best_task = task_id
                best_cost = cost
        
        assignments.append(best_task)
        if best_task >= 0:
            used_tasks.add(best_task)
    
    return jnp.array(assignments)
