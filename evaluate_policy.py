from dataset import ACTION_IDX
from dynamics_model import StateTransitionModel
import equinox as eqx
import jax.numpy as jnp
import jax
import numpy as np
from dataset import ARENA_BOUNDS_E, ARENA_BOUNDS_N, ACTION_VEL, ACTION_MAPPING
from rewards2 import point_navigation_reward


class MARLEnv:
    def __init__(self, num_agents=1, width=600, height=600, w_padding=100, h_padding=100):
        self.initial_velocities = jnp.concatenate(list(ACTION_VEL.values()))
        model = StateTransitionModel(state_size=4, num_actions=9, dropout=0, key=jax.random.PRNGKey(0))
        model = eqx.tree_deserialise_leaves("data/dynamics_model_weights.eqx", model)
        self.model = eqx.filter_jit(eqx.filter_vmap(model))
        self.num_agents = num_agents
        self.width = width
        self.height = height
        self.w_padding = w_padding
        self.h_padding = h_padding

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
        e_scale = self.width / (ARENA_BOUNDS_E[1] - ARENA_BOUNDS_E[0])
        n_scale = self.height / (ARENA_BOUNDS_N[1] - ARENA_BOUNDS_N[0])
        scaled_pos = pos * jnp.array([n_scale, e_scale]) + jnp.array([self.w_padding / 2, self.h_padding / 2])
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
            self.border_vis = pygame.Rect(self.w_padding // 2, self.h_padding // 2, self.width, self.height)
            self.screen = pygame.display.set_mode((self.width + self.w_padding, self.height + self.h_padding))
            self.clock = pygame.time.Clock()
            self.screen.fill((255, 255, 255))
            pygame.draw.rect(self.screen, "gray", self.border_vis)

        agent_color = (jnp.array([255, 0, 0]).reshape(1, -1) / jnp.arange(1, self.num_agents + 1).reshape(-1, 1))
        goal_color = (jnp.array([0, 255, 0]).reshape(1, -1) / jnp.arange(1, self.num_agents + 1).reshape(-1, 1))
        for i in range(len(agent_pos)):
            pygame.draw.circle(
                self.screen, 
                agent_color[i].tolist(),
                agent_pos[i].tolist(), 
                #(i + 4) * 5,
                15,
            )
            pygame.draw.circle(
                self.screen, 
                goal_color[i].tolist(),
                agent_goal[i].tolist(), 
                15.
                #(i + 4) * 5
            )
        if headless:
            return pygame.surfarray.array3d(self.screen)
        else:
            pygame.display.flip()


def rollout_policy(env, q_function, tasks, num_agents, key, timesteps=50):
    from modules import greedy_policy

    #key, goal_key = jax.random.split(key)
    #goal_idx = jax.random.choice(goal_key, jnp.arange(tasks["reward_kwargs"]["goal"].shape[0]), shape=(num_agents,), replace=False)
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

 
        
        


    #     ep_rewards = 0
    #     eval_q_function = eqx.nn.inference_mode(self.q_function)
    #     final_dists = []
    #     all_states = []
    #     for i in range(self.eval_episodes):
    #         agent_state = jnp.array([0.0, 0.0, 0, 0, 0])
    #         done = False
    #         eval_task = {
    #             "task_string": self.tasks["task_string"][i:i+1],
    #             "task_embedding": self.tasks["task_embedding"][i:i+1],
    #             "reward_function": self.tasks["reward_function"],
    #             "done_function": self.tasks["done_function"],
    #             "reward_kwargs": {"goal": self.tasks["reward_kwargs"]["goal"][i:i+1]},
    #         }
    #         ep_reward = 0
    #         num_steps = 0
    #         states = []
    #         while not done and num_steps < 50:
    #             action = greedy_policy(
    #                 eval_q_function, agent_state, self.tasks["task_embedding"][i], key=jax.random.PRNGKey(0)
    #             )
    #             next_state = self.simulator(agent_state, action)
    #             reward_fn_inputs = {
    #                 "state": agent_state.reshape(1, -1),
    #                 "action": action.reshape(1, -1),
    #                 "next_state": next_state.reshape(1, -1),
    #             }
    #             states.append(agent_state)
    #             result = add_rewards_to_dataset(reward_fn_inputs, eval_task)
    #             reward, done = result['next_reward'].reshape(1), result['next_done'].reshape(1)

    #             agent_state = next_state
    #             ep_reward += reward
    #             num_steps += 1
    #         ep_rewards += ep_reward
    #         all_states.append(jnp.stack(states, axis=0))
    #         final_dists.append(jnp.linalg.norm(agent_state[:2] - eval_task["reward_kwargs"]["goal"]).item())

    #     video = []
    #     for i, trajectory in enumerate(all_states):
    #         frames = jnp.zeros((trajectory.shape[0], 64, 64, 3), dtype=jnp.uint8)
    #         # boundaries roughly -2, 2
    #         agent_idx = ((

if __name__ == '__main__':
    from tasks import make_language_navigation_tasks
    from modules import GeneralQNetwork, greedy_policy
    from tasks import add_rewards_to_dataset

    key = jax.random.PRNGKey(0)
    tasks = make_language_navigation_tasks()
    eval_timesteps=50
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
    q_function = GeneralQNetwork(
        obs_size=config["obs_size"], 
        task_size=config["task_size"], 
        act_size=config["act_size"], 
        config=config["q_config"], 
        key=key
    )

    e = MARLEnv(num_agents=tasks['task_embedding'].shape[0])
    #agent_state = e.reset(key)
    data = rollout_policy(e, q_function, tasks, tasks['task_embedding'].shape[0], key, eval_timesteps)
    bgoals = jnp.repeat(jnp.expand_dims(tasks['reward_kwargs']['goal'], 0), eval_timesteps, axis=0)
    #rewards = tasks['reward_function'](data, goal=bgoals)
    rewards = jax.vmap(jax.vmap(point_navigation_reward))(data, goal=bgoals)
    # Reduce for single agent
    sa_rewards = rewards.sum(-1)
    # Now visualize
    frames = e.visualize_seq(data['state'], bgoals[0])

