# evaluate_policy.py (FIXED VERSION)
import copy
from constants import ACTION_IDX, ROBOT_DIAMETER
from dynamics_model import StateTransitionModel
import equinox as eqx
import jax.numpy as jnp
import jax
import numpy as np
from constants import ARENA_BOUNDS_E, ARENA_BOUNDS_N, ACTION_VEL
from rewards2 import point_navigation_reward
from rewards import ma_collision_reward_and_done
import pygame
from scipy.optimize import linear_sum_assignment  # Hungarian method


class MARLEnv:
    def __init__(self, num_agents=1, scale=128, padding=0.1, headless=True):
        self.initial_velocities = jnp.concatenate(list(ACTION_VEL.values()))
        model = StateTransitionModel(state_size=4, num_actions=9, dropout=0, key=jax.random.PRNGKey(0))
        model = eqx.tree_deserialise_leaves("data/dynamics_model_weights.eqx", model)
        self.model = eqx.filter_jit(eqx.filter_vmap(model))
        self.num_agents = num_agents
        self.scale = scale
        self.padding = scale * padding

        if headless:
            import os
            os.environ['SDL_VIDEODRIVER'] = 'dummy'

        pygame.display.init()
        self.screen = pygame.display.set_mode((int(self.scale + self.padding), int(self.scale + self.padding)))
        self.border_vis = pygame.Rect(int(self.padding // 2), int(self.padding // 2), int(self.scale), int(self.scale))

    def reset(self, key, eps=0.1):
        keys = jax.random.split(key, 6)
        # Zero vel prevents initial collisions
        vel = jnp.zeros((self.num_agents, 2))

        # Ensure agents do not start overlapped
        n_coords = jnp.arange(ARENA_BOUNDS_N[0] + eps, ARENA_BOUNDS_N[1] - eps, ROBOT_DIAMETER + 0.01) 
        e_coords = jnp.arange(ARENA_BOUNDS_E[0] + eps, ARENA_BOUNDS_E[1] - eps, ROBOT_DIAMETER + 0.01) 

        n_pos = jax.random.choice(keys[2], n_coords, (self.num_agents,))
        e_pos = jax.random.choice(keys[3], e_coords, (self.num_agents,))  # FIX: Use e_coords, not n_coords
        pos = jnp.stack([n_pos, e_pos], axis=-1)
        agent_state = jnp.concatenate([pos, vel], axis=-1)
        return agent_state

    def step(self, agent_state, action_idx):
        next_state = self.model(agent_state, action_idx)
        next_state = jnp.clip(
            next_state,
            a_min=jnp.array([
                ARENA_BOUNDS_N[0], ARENA_BOUNDS_E[0], -1, -1 
            ]),
            a_max=jnp.array([
                ARENA_BOUNDS_N[1], ARENA_BOUNDS_E[1], 1, 1
            ])
        )
        return next_state

    def state_pos_to_screen_pos(self, pos):
        n_scale = (self.scale) / (ARENA_BOUNDS_N[1] - ARENA_BOUNDS_N[0])
        e_scale = (self.scale) / (ARENA_BOUNDS_E[1] - ARENA_BOUNDS_E[0])
        shift = 0.5 * jnp.array([ARENA_BOUNDS_N[1] - ARENA_BOUNDS_N[0], ARENA_BOUNDS_E[1] - ARENA_BOUNDS_E[0]])
        # Flip n/y axis and change from n,e to x,y
        flipped = (jnp.array([-1, 1]) * pos).T
        # Scale to screen and shift by screen padding
        screen_pos = (flipped + shift) * jnp.array([n_scale, e_scale]) + jnp.array([self.padding / 2, self.padding / 2]) 
        return screen_pos

    def render_seq(self, agent_states, goals):
        frames = []
        for t in range(agent_states.shape[0]):
            frames.append(self.render(agent_states[t], goals[t]))
        video = np.stack(frames)
        return video

    def render(self, agent_state, goal):
        agent_pos = eqx.filter_vmap(self.state_pos_to_screen_pos)(agent_state[:, :2])
        agent_goal = eqx.filter_vmap(self.state_pos_to_screen_pos)(goal)

        color = [
            "red", "green", "blue", "yellow", "purple", "orange", "cyan", "magenta", "pink",
            "bisque", "brown", "burlywood", "cadetblue1", "darkgoldenrod1", "gold", 
            "light salmon", "light steel blue", "olive"
        ]
        cross_length = 0.05 * self.scale
        x_length = 0.033 * self.scale
        agent_radius = 0.5 * ROBOT_DIAMETER * (self.scale) / (ARENA_BOUNDS_N[1] - ARENA_BOUNDS_N[0])

        self.screen.fill("gray")
        self.rect = pygame.draw.rect(self.screen, "white", self.border_vis)
        for i in range(len(agent_pos)):
            pygame.draw.circle(
                self.screen, 
                color[i % len(color)],
                (int(agent_pos[i][0]), int(agent_pos[i][1])), 
                int(agent_radius)
            )
        for i in range(len(agent_pos)):
            if i % 2 == 0:
                pygame.draw.line(
                    self.screen,
                    color[i % len(color)],
                    (int(agent_goal[i][0] - x_length), int(agent_goal[i][1] - x_length)),
                    (int(agent_goal[i][0] + x_length), int(agent_goal[i][1] + x_length)),
                    2
                )
                pygame.draw.line(
                    self.screen,
                    color[i % len(color)],
                    (int(agent_goal[i][0] + x_length), int(agent_goal[i][1] - x_length)),
                    (int(agent_goal[i][0] - x_length), int(agent_goal[i][1] + x_length)),
                    2
                )
            else:
                pygame.draw.line(
                    self.screen,
                    color[i % len(color)],
                    (int(agent_goal[i][0] - cross_length), int(agent_goal[i][1])),
                    (int(agent_goal[i][0] + cross_length), int(agent_goal[i][1])),
                    2
                )
                pygame.draw.line(
                    self.screen,
                    color[i % len(color)],
                    (int(agent_goal[i][0]), int(agent_goal[i][1] - cross_length)),
                    (int(agent_goal[i][0]), int(agent_goal[i][1] + cross_length)),
                    2
                )

        pygame.display.flip()
        return pygame.surfarray.array3d(self.screen)


def rollout_policy(env, q_function, tasks, num_agents, key, timesteps=50, initial_state=None):
    embeds = tasks["task_embedding"]

    def scan_fn(carry, _):
        key, agent_state, prev_action = carry
        key, reset_key = jax.random.split(key)
        q_values = q_function(agent_state, embeds, jax.random.PRNGKey(0))
        action = q_values.argmax(-1)
        next_state = env.step(agent_state, action)
        return (reset_key, next_state, action), (agent_state, action, next_state)

    key, reset_key = jax.random.split(key)
    if initial_state is None:
        agent_states = env.reset(reset_key)
    _, (state, action, next_state) = jax.lax.scan(
        f=scan_fn, 
        init=(reset_key, agent_states, jnp.zeros((agent_states.shape[0],), dtype=jnp.int32)),
        xs=(), 
        length=timesteps
    )
    return {
        "state": state,
        "next_state": next_state,
        "action": jnp.expand_dims(action, -1),
    }


global_fn = jax.jit(jax.vmap(ma_collision_reward_and_done))


def hungarian_task_assignment(agent_initial_state, task_goals, task_indices):
    """
    Use the Hungarian method to optimally assign tasks to agents.

    The cost of assigning agent i to task j is the Euclidean distance
    between agent i's current position and task j's goal coordinate.

    Args:
        agent_initial_state : jnp array, shape [num_agents, 4]
                              Each row = [N_pos, E_pos, vel_N, vel_E]
                              *** Current position lives in columns 0 and 1 ***
                              This is populated inside env.reset() as:
                                  pos = jnp.stack([n_pos, e_pos], axis=-1)
                                  agent_state = jnp.concatenate([pos, vel], axis=-1)
                              so agent_initial_state[i, :2] = [N, E] of agent i.

        task_goals          : numpy array, shape [num_candidate_tasks, 2]
                              Each row = [N_goal, E_goal] for one task.
                              These come from tasks['reward_kwargs']['goal'][task_indices].

        task_indices        : list of int, length num_candidate_tasks
                              Indices into the full tasks dict (embeddings, strings, goals).
                              Must have len(task_indices) >= num_agents.

    Returns:
        assigned_task_indices : list of int, length num_agents
                                The task index (into the full tasks dict) assigned
                                to each agent after Hungarian optimisation.

    How it works (Hungarian method recap):
        1. Build cost matrix C of shape [num_agents, num_candidate_tasks]
           where C[i][j] = Euclidean distance from agent i's position to task j's goal.
        2. scipy.optimize.linear_sum_assignment solves the assignment problem:
           it finds the one-to-one matching that minimises total cost.
        3. The result is a permutation: agent_rows[k] gets task task_cols[k].
    """
    # agent_initial_state[:, :2] — columns 0 and 1 are [N_pos, E_pos]
    # Shape: [num_agents, 2]
    agent_positions = np.array(agent_initial_state[:, :2])

    # task_goals shape: [num_candidate_tasks, 2]
    # ---------------------------------------------------------------
    # Build cost matrix: C[i, j] = distance(agent i, task j)
    # agent_positions[:, None, :] -> [num_agents, 1, 2]
    # task_goals[None, :, :]     -> [1, num_candidate_tasks, 2]
    # broadcasting gives         -> [num_agents, num_candidate_tasks]
    # ---------------------------------------------------------------
    cost_matrix = np.linalg.norm(
        agent_positions[:, None, :] - task_goals[None, :, :],
        axis=-1
    )  # shape: [num_agents, num_candidate_tasks]

    print("\n  [Hungarian] Cost matrix (agent × task):")
    print("  " + "     ".join([f"T{task_indices[j]}" for j in range(len(task_indices))]))
    for i, row in enumerate(cost_matrix):
        print(f"  Agent {i}: " + "  ".join([f"{v:.3f}" for v in row]))

    # scipy's linear_sum_assignment implements the Hungarian method.
    # Returns: agent_rows (always [0,1,2,...,N-1]) and task_cols (the optimal assignment).
    agent_rows, task_cols = linear_sum_assignment(cost_matrix)

    # Map the local column indices back to actual task indices in the full tasks dict
    assigned_task_indices = [task_indices[j] for j in task_cols]

    print("\n  [Hungarian] Optimal assignment:")
    for i, (agent_i, task_col) in enumerate(zip(agent_rows, task_cols)):
        print(f"    Agent {agent_i}  →  Task index {task_indices[task_col]}"
              f"  (distance = {cost_matrix[agent_i, task_col]:.3f} m)")

    return assigned_task_indices


def evaluate_ma_policy(env=None, env_kwargs={}, tasks=None, model_path=None, q_function=None, config=None, eval_split=False, timesteps=50, seed=0, task_indices=None):
    """
    Evaluate multi-agent policy with language-based tasks.
    Task-to-agent assignment is done via the Hungarian method:
    cost = Euclidean distance from each agent's spawned position to each task's goal.

    Args:
        env: MARLEnv instance (creates new if None)
        env_kwargs: kwargs for MARLEnv (scale, padding, headless, etc.)
        tasks: Task dict with embeddings and reward functions
        model_path: Path to trained Q-network weights
        q_function: Pre-initialized Q-network (loads if model_path provided)
        config: Config dict with network/training hyperparams
        eval_split: Use evaluation tasks (True) or training (False)
        timesteps: Rollout length
        seed: Random seed
        task_indices: list of task indices to consider for assignment.
                      Must have len >= num_agents. If None, num_agents
                      tasks are picked randomly and Hungarian is still applied
                      (trivially, since it's 1-to-1 already).
    """
    from tasks import make_language_navigation_tasks
    from modules import GeneralMAQNetwork

    key = jax.random.PRNGKey(seed)
    
    # Load default tasks if not provided
    if tasks is None:
        tasks = make_language_navigation_tasks(eval_split)
    
    # Default config
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
    
    # Create Q-network if not provided
    if q_function is None:
        q_function = GeneralMAQNetwork(
            obs_size=config["obs_size"], 
            task_size=config["task_size"], 
            act_size=config["act_size"], 
            config=config["q_config"], 
            key=key
        )
    
    # Load pretrained weights if path provided
    if model_path is not None:
        q_function = eqx.tree_deserialise_leaves(model_path, q_function)
        print(f"✓ Loaded model from: {model_path}")

    # Create environment if not provided
    if env is None:
        env = MARLEnv(**env_kwargs, num_agents=config["num_agents"])

    # ── Spawn agents (this populates agent_initial_state) ──────────────────────
    # agent_initial_state shape: [num_agents, 4]
    # Columns: [N_pos, E_pos, vel_N, vel_E]
    # Current position of agent i = agent_initial_state[i, 0:2]
    # Built inside env.reset() as:
    #   pos = jnp.stack([n_pos, e_pos], axis=-1)          # [num_agents, 2]
    #   vel = jnp.zeros((num_agents, 2))                   # [num_agents, 2]
    #   agent_state = jnp.concatenate([pos, vel], axis=-1) # [num_agents, 4]
    key, reset_key = jax.random.split(key)
    agent_initial_state = env.reset(reset_key)
    # ── End spawn ──────────────────────────────────────────────────────────────

    num_agents = config["num_agents"]

    # ── Build the candidate task pool ─────────────────────────────────────────
    if task_indices is None:
        # Pick num_agents random unique tasks as candidates
        key, choice_key = jax.random.split(key)
        task_indices = list(
            np.array(
                jax.random.choice(
                    choice_key,
                    tasks['task_embedding'].shape[0],
                    (num_agents,),
                    replace=False
                )
            )
        )

    assert len(task_indices) >= num_agents, \
        f"Need at least {num_agents} candidate tasks, got {len(task_indices)}"

    # Goal coordinates for each candidate task — shape [num_candidates, 2]
    # This is what gets compared against agent positions to build the cost matrix
    candidate_goals = np.array(tasks['reward_kwargs']['goal'][task_indices])  # [num_candidates, 2]

    # ── Hungarian assignment ───────────────────────────────────────────────────
    # Input:  agent current positions [num_agents, 2]  (from agent_initial_state[:, :2])
    #         candidate goal coordinates [num_candidates, 2]
    # Output: one task index per agent — the optimal 1-to-1 matching
    print(f"\n[Hungarian] Assigning {num_agents} agents to tasks...")
    print(f"  Agent initial positions (N, E):")
    for i in range(num_agents):
        pos = agent_initial_state[i, :2]
        print(f"    Agent {i}: ({float(pos[0]):+.3f}N, {float(pos[1]):+.3f}E)")

    assigned_task_indices = hungarian_task_assignment(
        agent_initial_state=agent_initial_state,
        task_goals=candidate_goals,
        task_indices=task_indices
    )
    agent_task_idx = np.array(assigned_task_indices)
    # ── End Hungarian ──────────────────────────────────────────────────────────

    agent_tasks = {
        "task_embedding": tasks['task_embedding'][agent_task_idx],
        "reward_function": tasks["reward_function"],
        "done_function": tasks["done_function"],
        "reward_kwargs": {"goal": tasks["reward_kwargs"]["goal"][agent_task_idx]}
    }

    print(f"\n✓ Assigned {num_agents} agents with tasks (Hungarian-optimal):")
    for i in range(num_agents):
        goal = agent_tasks['reward_kwargs']['goal'][i]
        print(f"  Agent {i}: {tasks['task_string'][agent_task_idx[i]]}")
        print(f"           goal=({float(goal[0]):+.3f}N, {float(goal[1]):+.3f}E)")

    # Rollout policy — note: we pass the already-spawned agent_initial_state
    # so the rollout starts from the same positions used in assignment
    print(f"\n✓ Running {timesteps}-step rollout...")

    embeds = jnp.array(agent_tasks["task_embedding"])

    def scan_fn(carry, _):
        rkey, agent_state, prev_action = carry
        rkey, step_key = jax.random.split(rkey)
        q_values = q_function(agent_state, embeds, jax.random.PRNGKey(0))
        action = q_values.argmax(-1)
        next_state = env.step(agent_state, action)
        return (step_key, next_state, action), (agent_state, action, next_state)

    _, (state, action, next_state) = jax.lax.scan(
        f=scan_fn,
        init=(reset_key, agent_initial_state, jnp.zeros((num_agents,), dtype=jnp.int32)),
        xs=(),
        length=timesteps
    )
    data = {
        "state": state,
        "next_state": next_state,
        "action": jnp.expand_dims(action, -1),
    }

    # Compute goals for all timesteps
    bgoals = jnp.repeat(
        jnp.expand_dims(agent_tasks['reward_kwargs']['goal'], 0),
        timesteps,
        axis=0
    )
    
    # Compute rewards
    rewards = jax.vmap(jax.vmap(point_navigation_reward))(data, goal=bgoals)
    rewards, dones = global_fn(
        data["state"], 
        data["next_state"], 
        rewards, 
        jnp.zeros_like(rewards, dtype=bool)
    )
    
    # Only consider the rewards of the task each agent is assigned to
    rewards = rewards[..., jnp.arange(env.num_agents)]
    dones   = dones[..., jnp.arange(env.num_agents)]

    # Visualize
    print(f"✓ Rendering {timesteps} frames...")
    frames = env.render_seq(data['state'], bgoals)
    
    print(f"✓ Evaluation complete!")
    print(f"  Total reward: {rewards.sum():.3f}")
    print(f"  Mean distance to goal: {jnp.linalg.norm(data['next_state'][...,:2] - bgoals, axis=-1).mean():.3f}")
    
    return data, bgoals, frames, rewards, dones


def evaluate_policy(env_kwargs={}, tasks=None, model_path=None, q_function=None, config=None, eval_split=False, timesteps=50):
    """Single-agent evaluation (deprecated, use evaluate_ma_policy)"""
    from tasks import make_language_navigation_tasks
    from modules import GeneralQNetwork

    key = jax.random.PRNGKey(0)
    if tasks is None:
        tasks = make_language_navigation_tasks(eval_split)
    if config is None:
        config = {
            "seed": 0,
            "lr": 0.0001,
            "loss": "meanq",
            "weight_decay": 0.0001,
            "gamma": 0.95,
            "batch_size": 32,
            "tau": 0.001,
            "epochs": 3000,
            "eval_interval": 50,
            "q_config": {
                "mlp_size": 384,
                "head_size": 384,
                "ensemble_size": 2,
                "dropout": 0.0,
                "ensemble_reduce": "min",
            },
            "task_size": 768,
            "obs_size": 4,
            "act_size": 9,
            "simulator_weights": "data/dynamics_model_weights.eqx",
        }
    if q_function is None:
        q_function = GeneralQNetwork(
            obs_size=config["obs_size"], 
            task_size=config["task_size"], 
            act_size=config["act_size"], 
            config=config["q_config"], 
            key=key
        )
    if model_path is not None:
        q_function = eqx.tree_deserialise_leaves(model_path, q_function)

    e = MARLEnv(**env_kwargs, num_agents=tasks['task_embedding'].shape[0])
    data = rollout_policy(e, q_function, tasks, tasks['task_embedding'].shape[0], key, timesteps)
    bgoals = jnp.repeat(jnp.expand_dims(tasks['reward_kwargs']['goal'], 0), timesteps, axis=0)
    rewards = jax.vmap(jax.vmap(point_navigation_reward))(data, goal=bgoals)
    frames = e.render_seq(data['state'], bgoals)
    return data, bgoals, frames, rewards


def create_video(output_path='video.mp4', fps=10, *args, **kwargs):
    """Create MP4 video from policy evaluation"""
    import imageio
    print(f"Creating video at {fps} fps...")
    data, bgoals, frames, rewards = evaluate_policy(*args, **kwargs)
    frames = np.array(frames, dtype=np.uint8)
    with imageio.get_writer(output_path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f"✓ Video saved to {output_path}")


def run_interactive(dt=10):
    """Interactive keyboard control with pygame visualization"""
    e = MARLEnv(headless=False, scale=512, num_agents=1)
    agent_state = e.reset(jax.random.PRNGKey(0))
    goal = jnp.array([0.0, 0.0])

    pygame.init()
    clock = pygame.time.Clock()
    running = True
    e.render(agent_state, jnp.expand_dims(goal, 0))

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()

        if (keys[pygame.K_w] and keys[pygame.K_a]) or keys[pygame.K_z]:
            command = ACTION_IDX["NW"]
        elif (keys[pygame.K_w] and keys[pygame.K_d]) or keys[pygame.K_c]:
            command = ACTION_IDX["NE"]
        elif (keys[pygame.K_s] and keys[pygame.K_a]) or keys[pygame.K_q]:
            command = ACTION_IDX["SW"]
        elif (keys[pygame.K_s] and keys[pygame.K_d]) or keys[pygame.K_e]:
            command = ACTION_IDX["SE"]
        elif keys[pygame.K_s]:
            command = ACTION_IDX["S"]
        elif keys[pygame.K_w]:
            command = ACTION_IDX["N"]
        elif keys[pygame.K_d]:
            command = ACTION_IDX["E"]
        elif keys[pygame.K_a]:
            command = ACTION_IDX["W"]
        else:
            command = ACTION_IDX["0"]

        action = jnp.array([command])
        agent_state = e.step(agent_state, action)
        e.render(agent_state, jnp.expand_dims(goal, 0))
        clock.tick(dt)

    pygame.quit()


if __name__ == '__main__':
    print("=" * 60)
    print("MULTI-AGENT POLICY EVALUATION  (Hungarian assignment)")
    print("=" * 60)

    # ── task_indices: the CANDIDATE pool of tasks to assign from.
    #    Must have len >= num_agents (5 here).
    #    The Hungarian method picks the optimal 1-to-1 match based on
    #    distance from each agent's spawned position to each task's goal.
    #
    #    You can pass more candidates than agents — e.g. 8 tasks for 5 agents —
    #    and Hungarian will pick the best 5 assignments from those 8.
    #
    #    Indices refer to rows in tasks['task_string'] / tasks['reward_kwargs']['goal'].
    #    Use available_tasks.py to browse them.
    data, goals, frames, rewards, dones = evaluate_ma_policy(
        env_kwargs={"scale": 512, "headless": True},
        model_path="models/best_model.eqx",
        timesteps=100,
        seed=42,
        task_indices=[0, 11, 22, 33, 123]   # 5 candidates for 5 agents — Hungarian assigns optimally
    )

    # Save video
    import imageio
    frames = np.array(frames, dtype=np.uint8)
    output_path = 'marl_navigation.mp4'
    print(f"\nSaving video to {output_path}...")
    with imageio.get_writer(output_path, fps=10) as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f"✓ Video saved!")
