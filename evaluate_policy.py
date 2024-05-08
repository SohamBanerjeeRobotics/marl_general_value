import copy
from constants import ACTION_IDX
from dynamics_model import StateTransitionModel
import equinox as eqx
import jax.numpy as jnp
import jax
import numpy as np
from constants import ARENA_BOUNDS_E, ARENA_BOUNDS_N, ACTION_VEL
from rewards2 import point_navigation_reward
from rewards import ma_collision_reward_and_done
import pygame


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
        self.screen = pygame.display.set_mode((self.scale + self.padding , self.scale + self.padding))
        self.border_vis = pygame.Rect(self.padding // 2, self.padding // 2, self.scale, self.scale)

    def reset(self, key, eps=0.1):
        keys = jax.random.split(key, 3)
        vel = jax.random.choice(keys[0], self.initial_velocities, shape=(self.num_agents, 2)) 
        vel = vel * jax.random.uniform(keys[1], shape=(self.num_agents, 2), minval=0.0, maxval=1.0)
        pos = jax.random.uniform(
            keys[2], shape=(self.num_agents, 2,), 
            minval=jnp.array([
                ARENA_BOUNDS_N[0] + eps, ARENA_BOUNDS_E[0] + eps
            ]), 
            maxval=jnp.array([
                ARENA_BOUNDS_N[1] - eps, ARENA_BOUNDS_E[1] - eps
            ])  
        )
        agent_state = jnp.concatenate([pos, vel], axis=-1)
        return agent_state

    def step(self, agent_state, action_idx):
        next_state = self.model(agent_state, action_idx)
        next_state = jnp.clip(
            next_state,
            a_min=jnp.array([
                ARENA_BOUNDS_N[0] - 0.01, ARENA_BOUNDS_E[0] - 0.01, -1, -1 
            ]),
            a_max=jnp.array([
                ARENA_BOUNDS_N[1] + 0.01, ARENA_BOUNDS_E[1] + 0.01, 1, 1
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

        #agent_color = (jnp.array([255, 0, 0]).reshape(1, -1) / jnp.arange(1, self.num_agents + 1).reshape(-1, 1))
        #goal_color = (jnp.array([0, 255, 0]).reshape(1, -1) / jnp.arange(1, self.num_agents + 1).reshape(-1, 1))
        color = [
            "red", "green", "blue", "yellow", "purple", "orange", "cyan", "magenta", "pink",
            "bisque", "brown", "burlywood", "cadetblue1", "darkgoldenrod1", "gold", 
            "light salmon", "light steel blue", "olive"
            ]
        cross_length = 0.05 * self.scale
        x_length = 0.033 * self.scale
        agent_radius = 0.15 * (self.scale) / (ARENA_BOUNDS_N[1] - ARENA_BOUNDS_N[0])

        self.screen.fill("gray")
        self.rect = pygame.draw.rect(self.screen, "white", self.border_vis)
        for i in range(len(agent_pos)):
            pygame.draw.circle(
                self.screen, 
                color[i],
                agent_pos[i].tolist(), 
                agent_radius
            )
        for i in range(len(agent_pos)):
            if i % 2 == 0:
                pygame.draw.line(
                    self.screen,
                    color[i],
                    (agent_goal[i] - jnp.array([x_length, x_length])).tolist(),
                    (agent_goal[i] + jnp.array([x_length, x_length])).tolist(),
                    2
                )
                pygame.draw.line(
                    self.screen,
                    color[i],
                    (agent_goal[i] - jnp.array([-x_length, x_length])).tolist(),
                    (agent_goal[i] + jnp.array([-x_length, x_length])).tolist(),
                    2
                )
            else:
                pygame.draw.line(
                    self.screen,
                    color[i],
                    (agent_goal[i] - jnp.array([cross_length, 0])).tolist(),
                    (agent_goal[i] + jnp.array([cross_length, 0])).tolist(),
                    2
                )
                pygame.draw.line(
                    self.screen,
                    color[i],
                    (agent_goal[i] - jnp.array([0, cross_length])).tolist(),
                    (agent_goal[i] + jnp.array([0, cross_length])).tolist(),
                    2
                )

        pygame.display.flip()
        return pygame.surfarray.array3d(self.screen)


def rollout_policy(env, q_function, tasks, num_agents, key, timesteps=50, initial_state=None):
    embeds = tasks["task_embedding"]

    def scan_fn(carry, _):
        key, agent_state, prev_action = carry
        key, reset_key = jax.random.split(key)
        action = q_function(agent_state, embeds, jax.random.PRNGKey(0)).argmax(-1)
        #action = jax.random.categorical(
        #    reset_key, q_function(agent_state, embeds, jax.random.PRNGKey(0)) * 50.0
        #)
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

def evaluate_ma_policy(env_kwargs={}, tasks=None, model_path=None, q_function=None, config=None, eval_split=True, timesteps=50, seed=0):
    from tasks import make_language_navigation_tasks
    from modules import GeneralMAQNetwork

    key = jax.random.PRNGKey(seed)
    if tasks is None:
        tasks = make_language_navigation_tasks(eval_split)
    if config is None:
        config = {
            "seed": 0,
            "lr": 0.0001,
            "loss": "meanq",
            "weight_decay": 0.0001,
            "gamma": jnp.array([0.95]),
            "batch_size": 32,
            "num_agents": 5,
            "tau": jnp.array([1/1000]),
            "epochs": 3000,
            "eval_interval": 50,
            "q_config": {
                "mlp_size": 384,
                "head_size": 384,
                "ensemble_size": 1,
                "dropout": 0.0,
                "ensemble_size": 2,
                "ensemble_reduce": "min",
            },
            "task_size": 768,
            "obs_size": 4,
            "act_size": 9,
            "simulator_weights": "data/dynamics_model_weights.eqx",
        }
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

    e = MARLEnv(**env_kwargs, num_agents=config["num_agents"])
    agent_task_idx = jax.random.choice(key, tasks['task_embedding'].shape[0], (config['num_agents'],), replace=False)
    agent_tasks = {
        "task_embedding": tasks['task_embedding'][agent_task_idx],
        "reward_function": tasks["reward_function"],
        "done_fuction": tasks["done_function"],
        "reward_kwargs": {"goal": tasks["reward_kwargs"]["goal"][agent_task_idx]}
    } 

    data = rollout_policy(e, q_function, agent_tasks, config['num_agents'], key, timesteps)
    bgoals = jnp.repeat(jnp.expand_dims(agent_tasks['reward_kwargs']['goal'], 0), timesteps, axis=0)
    rewards = jax.vmap(jax.vmap(point_navigation_reward))(data, goal=bgoals)
    rewards, _ = global_fn(data["state"], data["next_state"], rewards, jnp.zeros_like(rewards, dtype=bool))
    #global_rewards, _ = globa_fn(data["state"], data[
    # TODO: Collision rewards
    # Now visualize
    frames = e.render_seq(data['state'], bgoals)
    return data, bgoals, frames, rewards

def evaluate_policy(env_kwargs={}, tasks=None, model_path=None, q_function=None, config=None, eval_split=True, timesteps=50):
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
            "gamma": jnp.array([0.95]),
            "batch_size": 32,
            "tau": jnp.array([1/1000]),
            "epochs": 3000,
            "eval_interval": 50,
            "q_config": {
                "mlp_size": 384,
                "head_size": 384,
                "ensemble_size": 1,
                "dropout": 0.0,
                "ensemble_size": 2,
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
    # Now visualize
    frames = e.render_seq(data['state'], bgoals)
    return data, bgoals, frames, rewards


def create_video(*args, **kwargs):
    import imageio
    data, bgoals, frames, rewards = evaluate_policy(*args, **kwargs)
    # Define the codec and create a VideoWriter object
    #frames = np.array(jnp.transpose(frames, (0, 3, 1, 2)))
    frames = np.array(frames, dtype=np.uint8)
    with imageio.get_writer('video.mp4', fps=1) as writer:
        for frame in frames:
            writer.append_data(frame)

def run_interactive(dt=10):
    e = MARLEnv(headless=False)
    agent_state = e.reset(jax.random.PRNGKey(0))
    goal = jnp.zeros_like(agent_state)[..., :2]

    pygame.init()
    clock = pygame.time.Clock()
    running = True
    e.render(agent_state, goal)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            keys = pygame.key.get_pressed()

            if (keys[pygame.K_w] and keys[pygame.K_a]) or keys[pygame.K_z]:
                # TODO: Why is N/S inverted? 
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

        action = command.reshape(1)

        agent_state = e.step(agent_state, action)
        e.render(agent_state, goal)
        clock.tick(dt)

if __name__ == '__main__':
    run_interactive()
    #create_video(model_path=None, env_kwargs={"scale": 480})

