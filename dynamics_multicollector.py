import jax.numpy as jnp
import jax
from jax import random
from dynamics_model import StateTransitionModel
from dataset import ARENA_BOUNDS_N, ARENA_BOUNDS_E  
import equinox as eqx

class DynamicsMultiCollector:
    def __init__(self, sim, reward_fn, done_fn, goals, embeddings, num_envs, random_epochs, config):
        self.obs_shape = config["obs_size"]
        self.num_acts = config["act_size"]
        self.num_envs = num_envs
        self.sim_step = eqx.filter_vmap(sim)
        self.sim_reset = eqx.filter_vmap(sim.initial_state)
        self.reward_fn = [eqx.filter_vmap(fn) for fn in reward_fn]
        self.done_fn = [eqx.filter_vmap(fn) for fn in done_fn]
        self.goals = goals
        self.embeddings = jnp.concatenate(embeddings, axis=0)
        self.random_epochs = random_epochs

        assert self.num_envs == sum([g.shape[0] for g in self.goals]) == self.embeddings.shape[0]

    def initial_state(self):
        return {
            "observation": jnp.zeros((self.num_envs, self.obs_shape)),
            "next_observation": jnp.zeros((self.num_envs, self.obs_shape)),
            "action": jnp.zeros((self.num_envs)),
            "reward": jnp.zeros((self.num_envs)),
            "done": jnp.ones((self.num_envs), dtype=bool),
            "episode_reward": jnp.zeros((self.num_envs)),
            "epoch": jnp.array(0),
            "best_return": -jnp.inf,
            "terminal_episode_return": jnp.array(0.0),
            "episode_length": jnp.zeros((self.num_envs))
        }

    def __call__(self, q_function, cstate, policy, epsilon, key):
        # Initial obs
        reset_key = random.split(key, self.num_envs)
        # Reset done envs
        reset_mask = cstate['done']
        cstate['observation'] = reset_mask.reshape(-1, 1) * self.sim_reset(reset_key) + ~reset_mask.reshape(-1, 1) * cstate['observation']
        cstate['done'] = jnp.zeros_like(cstate['done'])

        # Record metrics
        cstate['episode_reward'] = cstate['episode_reward'] * reset_mask
        cstate['episode_length'] = cstate['episode_length'] * reset_mask

        # Sample action
        key, action_key = random.split(key)
        cstate['action'] = jax.lax.cond(
            cstate['epoch'] < self.random_epochs,
            lambda: random.randint(action_key, (self.num_envs,), 0, 9),
            lambda: eqx.filter_vmap(policy, in_axes=(None, 0, 0, None, 0))(
                q_function, cstate['observation'], self.embeddings, epsilon, random.split(action_key, self.num_envs)
            )
        )

        # Step env
        cstate['next_observation'] = self.sim_step(cstate['observation'], cstate['action'])

        # Compute reward and done for each env/task
        reward = []
        done = []
        for (rfn, dfn, goal) in zip(self.reward_fn, self.done_fn, self.goals):
            reward.append(rfn(
                {"state": cstate['observation'], "action": cstate['action'], "next_state": cstate['next_observation']},
                goal
            ))
            done.append(dfn(
                {"state": cstate['observation'], "action": cstate['action'], "next_state": cstate['next_observation']},
                goal
            ))
        cstate['reward'] = jnp.concatenate(reward, axis=0)
        cstate['done'] = jnp.concatenate(done, axis=0)

        # Keep running totals of reward and episode length
        cstate['episode_reward'] += cstate['reward']
        cstate['episode_length'] += 1

        # Record the return
        terminal_episode_return = (cstate['episode_reward'] * cstate['done']).sum() / jnp.maximum(cstate['done'].sum(), 1)
        cstate['terminal_episode_return'] = jax.lax.select(
            jnp.isnan(terminal_episode_return),
            cstate['terminal_episode_return'],
            terminal_episode_return
        )

        transitions = {
            "state": cstate['observation'],
            "action": cstate['action'], 
            "next_reward": cstate['reward'], 
            "next_state": cstate['next_observation'],
            "next_done": cstate['done'],
            "task_embedding": self.embeddings,
        }
        cstate['observation'] = cstate['next_observation']

        cstate['epoch'] += 1
        cstate['best_return'] = jnp.maximum(cstate['best_return'], cstate['terminal_episode_return'])
        return transitions, cstate 
