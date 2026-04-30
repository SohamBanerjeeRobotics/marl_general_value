# evaluate_policy_ma.py - NEW FUNCTION

import jax
import jax.numpy as jnp
import equinox as eqx
from tasks import make_language_navigation_tasks
from modules import GeneralMAQNetwork
from assignment import compute_cost_matrix, hungarian_assignment, greedy_nearest_neighbor
from evaluate_policy import MARLEnv, rollout_policy, global_fn
from rewards2 import point_navigation_reward


def evaluate_ma_policy_with_assignment(
    env=None,
    env_kwargs={},
    model_path=None,
    q_function=None,
    config=None,
    eval_split=True,
    timesteps=50,
    seed=0,
    use_hungarian=True,
    use_all_tasks=False,
    num_agents_override=None
):
    """
    Evaluate multi-agent policy with STATIC task assignment.
    
    Args:
        use_hungarian: True = Hungarian (optimal), False = greedy
        use_all_tasks: True = use all available tasks, False = random subset
        num_agents_override: Number of agents (if None, use config)
    
    Returns:
        data: trajectory data
        goals: assigned goal positions [timesteps, num_agents, 2]
        frames: rendered frames
        rewards: computed rewards
        dones: episode termination flags
        assignments: task assignments [num_agents]
        task_info: info about assigned tasks
    """
    
    key = jax.random.PRNGKey(seed)
    
    # ========== SETUP ==========
    if config is None:
        config = {
            "seed": 0,
            "lr": 0.0001,
            "loss": "meanq",
            "weight_decay": 0.0001,
            "gamma": 0.95,
            "batch_size": 32,
            "num_agents": 5,
            "tau": 0.001,
            "epochs": 3000,
            "eval_interval": 50,
            "q_config": {
                "mlp_size": 1024,
                "head_size": 1024,
                "ensemble_size": 2,
                "dropout": 0.0,
                "ensemble_reduce": "min",
            },
            "task_size": 768,
            "obs_size": 4,
            "act_size": 9,
            "simulator_weights": "data/dynamics_model_weights.eqx",
        }
    
    if num_agents_override is not None:
        config["num_agents"] = num_agents_override
    
    num_agents = config["num_agents"]
    
    # ========== STEP 1: LOAD ALL TASKS ==========
    print("\n" + "=" * 70)
    print("STEP 1: Loading tasks")
    print("=" * 70)
    
    all_tasks = make_language_navigation_tasks(eval=eval_split)
    all_task_embeddings = jnp.array(all_tasks["task_embedding"])  # [num_tasks, 768]
    all_task_goals = jnp.array(all_tasks["reward_kwargs"]["goal"])  # [num_tasks, 2]
    all_task_strings = all_tasks["task_string"]
    
    num_all_tasks = len(all_task_strings)
    print(f"✓ Loaded {num_all_tasks} available tasks")
    
    # ========== STEP 2: SELECT SUBSET OF TASKS ==========
    if use_all_tasks:
        print(f"✓ Using all {num_all_tasks} tasks")
        selected_task_indices = jnp.arange(num_all_tasks)
        task_embeddings = all_task_embeddings
        task_goals = all_task_goals
        task_strings = all_task_strings
    else:
        print(f"✓ Selecting {num_agents} tasks from {num_all_tasks} available")
        key, task_key = jax.random.split(key)
        selected_task_indices = jax.random.choice(
            task_key, 
            num_all_tasks, 
            (num_agents,), 
            replace=False
        )
        task_embeddings = all_task_embeddings[selected_task_indices]
        task_goals = all_task_goals[selected_task_indices]
        task_strings = [all_task_strings[i] for i in selected_task_indices]
    
    # ========== STEP 3: LOAD Q-NETWORK ==========
    print("\n" + "=" * 70)
    print("STEP 2: Loading Q-network")
    print("=" * 70)
    
    if q_function is None:
        q_function = GeneralMAQNetwork(
            obs_size=config["obs_size"],
            task_size=config["task_size"],
            act_size=config["act_size"],
            config=config["q_config"],
            key=key
        )
    
    if model_path is not None:
        q_function = eqx.tree_deserialise_leaves(model_path, q_function)
        print(f"✓ Loaded model from: {model_path}")
    else:
        print(f"⚠ Using untrained Q-network")
    
    # ========== STEP 4: INITIALIZE ENVIRONMENT ==========
    print("\n" + "=" * 70)
    print("STEP 3: Initializing environment")
    print("=" * 70)
    
    if env is None:
        env = MARLEnv(**env_kwargs, num_agents=num_agents)
    
    key, reset_key = jax.random.split(key)
    initial_agent_state = env.reset(reset_key)
    print(f"✓ Environment initialized with {num_agents} agents")
    
    # ========== STEP 5: COMPUTE TASK ASSIGNMENTS ==========
    print("\n" + "=" * 70)
    print("STEP 4: Computing task assignments")
    print("=" * 70)
    
    agent_positions = initial_agent_state[:, :2]  # [num_agents, 2]
    
    # Compute cost matrix (distances)
    cost_matrix = compute_cost_matrix(agent_positions, task_goals)
    print(f"Cost matrix shape: {cost_matrix.shape}")
    
    # Solve assignment problem
    if use_hungarian:
        assignments, total_cost = hungarian_assignment(cost_matrix)
        algorithm_name = "Hungarian (optimal)"
    else:
        assignments = greedy_nearest_neighbor(cost_matrix)
        total_cost = cost_matrix[jnp.arange(len(assignments)), assignments].sum()
        algorithm_name = "Greedy nearest-neighbor"
    
    print(f"✓ {algorithm_name} solved")
    print(f"  Total distance cost: {total_cost:.3f}")
    
    # Print assignments
    print(f"\nTask assignments:")
    for agent_id, task_id in enumerate(assignments):
        if task_id >= 0:
            dist = cost_matrix[agent_id, task_id]
            print(f"  Agent {agent_id} → Task {task_id}: {task_strings[task_id]}")
            print(f"           Distance: {dist:.3f}")
        else:
            print(f"  Agent {agent_id} → UNASSIGNED")
    
    # ========== STEP 6: CREATE AGENT-SPECIFIC TASK DICT ==========
    print("\n" + "=" * 70)
    print("STEP 5: Preparing task embeddings for rollout")
    print("=" * 70)
    
    # Get embeddings for assigned tasks
    assigned_embeddings = task_embeddings[assignments]  # [num_agents, 768]
    assigned_goals = task_goals[assignments]  # [num_agents, 2]
    assigned_task_strings = [task_strings[int(a)] for a in assignments]
    
    agent_tasks = {
        "task_embedding": assigned_embeddings,
        "reward_function": all_tasks["reward_function"],
        "done_function": all_tasks["done_function"],
        "reward_kwargs": {"goal": assigned_goals}
    }
    
    print(f"✓ Prepared {num_agents} task embeddings for rollout")
    
    # ========== STEP 7: RUN ROLLOUT ==========
    print("\n" + "=" * 70)
    print("STEP 6: Running policy rollout")
    print("=" * 70)
    
    data = eqx.filter_jit(rollout_policy)(
        env, q_function, agent_tasks, num_agents, key, timesteps
    )
    
    print(f"✓ Rollout complete ({timesteps} timesteps)")
    
    # ========== STEP 8: COMPUTE REWARDS ==========
    print("\n" + "=" * 70)
    print("STEP 7: Computing rewards")
    print("=" * 70)
    
    bgoals = jnp.repeat(
        jnp.expand_dims(agent_tasks['reward_kwargs']['goal'], 0),
        timesteps,
        axis=0
    )
    
    rewards = jax.vmap(jax.vmap(point_navigation_reward))(data, goal=bgoals)
    rewards, dones = global_fn(
        data["state"],
        data["next_state"],
        rewards,
        jnp.zeros_like(rewards, dtype=bool)
    )
    
    rewards = rewards[..., jnp.arange(env.num_agents)]
    dones = dones[..., jnp.arange(env.num_agents)]
    
    print(f"✓ Rewards computed")
    print(f"  Mean reward: {rewards.mean():.3f}")
    print(f"  Total reward: {rewards.sum():.3f}")
    
    # ========== STEP 9: RENDER FRAMES ==========
    print("\n" + "=" * 70)
    print("STEP 8: Rendering frames")
    print("=" * 70)
    
    frames = env.render_seq(data['state'], bgoals)
    print(f"✓ Rendered {len(frames)} frames")
    
    # ========== RETURN RESULTS ==========
    task_info = {
        "task_strings": assigned_task_strings,
        "task_goals": assigned_goals,
        "assignments": assignments,
        "cost_matrix": cost_matrix,
        "total_cost": total_cost,
        "algorithm": algorithm_name
    }
    
    return data, bgoals, frames, rewards, dones, assignments, task_info
