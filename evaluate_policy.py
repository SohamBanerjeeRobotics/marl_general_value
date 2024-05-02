import copy
from dataset import ACTION_IDX
from dynamics_model import StateTransitionModel
import equinox as eqx
import jax.numpy as jnp
import jax
import numpy as np
from dataset import ARENA_BOUNDS_E, ARENA_BOUNDS_N, ACTION_VEL, ACTION_MAPPING
from rewards2 import point_navigation_reward


class MARLEnv:
    def __init__(self, num_agents=1, scale=128, padding=0.1):
        self.initial_velocities = jnp.concatenate(list(ACTION_VEL.values()))
        model = StateTransitionModel(state_size=4, num_actions=9, dropout=0, key=jax.random.PRNGKey(0))
        model = eqx.tree_deserialise_leaves("data/dynamics_model_weights.eqx", model)
        self.model = eqx.filter_jit(eqx.filter_vmap(model))
        self.num_agents = num_agents
        self.scale = scale
        self.padding = scale * padding

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
                ARENA_BOUNDS_N[0], ARENA_BOUNDS_E[0], -1, -1 
            ]),
            a_max=jnp.array([
                ARENA_BOUNDS_N[1], ARENA_BOUNDS_E[1], 1, 1
            ])
        )
        return next_state

    def state_pos_to_screen_pos(self, pos):
        e_scale = (self.scale - self.padding) / (ARENA_BOUNDS_E[1] - ARENA_BOUNDS_E[0])
        n_scale = (self.scale - self.padding) / (ARENA_BOUNDS_N[1] - ARENA_BOUNDS_N[0])
        scaled_pos = pos * jnp.array([n_scale, e_scale]) + jnp.array([self.padding / 2, self.padding / 2])
        # Screen coords are left to right, top to bottom
        screen_pos = (scaled_pos * jnp.array([1, -1])).T
        return screen_pos

    def visualize_seq(self, agent_states, goal, dt=1e-4):
        frames = []
        for t in range(agent_states.shape[0]):
            frames.append(self.visualize(agent_states[t], goal, True, dt))
        video = np.stack(frames)
        return video


    def visualize(self, agent_state, goal, headless=False, dt=1e-4):
        agent_pos = eqx.filter_vmap(self.state_pos_to_screen_pos)(agent_state[:, :2])
        agent_goal = eqx.filter_vmap(self.state_pos_to_screen_pos)(goal)

        if headless:
            import os
            os.environ['SDL_VIDEODRIVER'] = 'dummy'

        import pygame
        if not pygame.get_init():
            pygame.init()
            self.screen = pygame.display.set_mode((self.scale + self.padding , self.scale + self.padding))
            self.border_vis = pygame.Rect(self.padding // 2, self.padding // 2, self.scale, self.scale)
            self.clock = pygame.time.Clock()

        self.screen.fill("gray")
        self.rect = pygame.draw.rect(self.screen, "white", self.border_vis)

        agent_color = (jnp.array([255, 0, 0]).reshape(1, -1) / jnp.arange(1, self.num_agents + 1).reshape(-1, 1))
        goal_color = (jnp.array([0, 255, 0]).reshape(1, -1) / jnp.arange(1, self.num_agents + 1).reshape(-1, 1))
        for i in range(len(agent_pos)):
            pygame.draw.circle(
                self.screen, 
                agent_color[i].tolist(),
                agent_pos[i].tolist(), 
                0.05 * self.scale,
            )
            pygame.draw.circle(
                self.screen, 
                goal_color[i].tolist(),
                agent_goal[i].tolist(), 
                0.05 * self.scale,
            )

        pygame.display.flip()
        return pygame.surfarray.array3d(self.screen)


def rollout_policy(env, q_function, tasks, num_agents, key, timesteps=50):
    from modules import greedy_policy

    embeds = tasks["task_embedding"]

    def scan_fn(carry, _):
        agent_state, prev_action = carry
        action = eqx.filter_vmap(greedy_policy, in_axes=(None, 0, 0, None))(
            q_function, agent_state, embeds, jax.random.PRNGKey(0)
        )
        next_state = env.step(agent_state, action)
        return (next_state, action), (agent_state, action, next_state)

    key, reset_key = jax.random.split(key)
    agent_states = env.reset(reset_key)
    _, (state, action, next_state) = jax.lax.scan(
        f=scan_fn, 
        init=(agent_states, jnp.zeros((agent_states.shape[0],), dtype=jnp.int32)),
        xs=(), 
        length=timesteps
    )
    return {
        "state": state,
        "next_state": next_state,
        "action": jnp.expand_dims(action, -1),
    }

def evaluate_policy(model_path=None, q_function=None, config=None, eval_split=True, timesteps=50):
    from tasks import make_language_navigation_tasks
    from modules import GeneralQNetwork

    key = jax.random.PRNGKey(0)
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

    e = MARLEnv(num_agents=tasks['task_embedding'].shape[0])
    data = rollout_policy(e, q_function, tasks, tasks['task_embedding'].shape[0], key, timesteps)
    bgoals = jnp.repeat(jnp.expand_dims(tasks['reward_kwargs']['goal'], 0), timesteps, axis=0)
    rewards = jax.vmap(jax.vmap(point_navigation_reward))(data, goal=bgoals)
    # Now visualize
    frames = e.visualize_seq(data['state'], bgoals[0])
    return data, bgoals, frames, rewards



if __name__ == '__main__':
    evaluate_policy()