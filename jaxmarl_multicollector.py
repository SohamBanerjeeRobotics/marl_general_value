from typing import Dict, List, NamedTuple
import numpy as np
import jax
import jax.numpy as jnp
from jax import random


# Done is true at the initial state of the following episode
# E.g.
# env.step() -> (observation, False)
# policy(observation, False)
# # Terminal state
# env.step() -> (observation, True)
# env.reset()
# Env is reset before the policy ever sees the "done" state

# However, the way we store, we will append 'done' to the end of the episode
# so 'done' will align with the last action the agent took
# it really means "done after this action"
# marking the last transition in the episode
# q(next_obs) * done
#
# Start will mark the first transition of the episode

## ISSUE: collector done is offset from training done

class JMMultiCollector:
    # Lifetime variables
    sampled_frames = 0
    sampled_epochs = 0
    best_reward = -np.inf

    # Discarded state between calls
    episode_reward = []
    seq_lens = []
    seq_len = 0

    def __init__(self, env, config):
        self.env = env
        self.config = config["collect"]
        self.obs_shape = jnp.stack(env.observation_space.sample(), axis=0).shape # Agent, *
        self.act_shape = jnp.stack(env.action_space.sample(), axis=0).shape # Agent, *
        self.num_envs = self.env.num_envs
        self.num_agents = self.env.n_agents

        self.observation = jnp.zeros((self.num_envs, *self.obs_shape))
        self.next_observation = jnp.zeros((self.num_envs, *self.obs_shape))
        self.action = jnp.zeros((self.num_envs, *self.act_shape))
        self.reward = jnp.zeros((self.num_envs, self.num_agents))
        self.done = jnp.ones((self.num_envs, self.num_agents), dtype=bool)
        self.episode_reward = jnp.full((self.num_envs, self.num_agents), 0)

    def __call__(self, policy, action_noise, key, need_reset=False):
        # Initial obs
        key, reset_key = random.split(key)
        self.env.seed(random.bits(reset_key).item())
        if jnp.all(self.done):
            self.env.reset()
        else:
            reset_idx = jnp.unique(jnp.where(self.done)[0])
            for idx in reset_idx:
                self.env.reset_at(idx.item())
        #self.done = self.done.at[reset_idx].set(False)
        self.episode_reward = self.episode_reward * ~self.done
        self.done = jnp.zeros_like(self.done)
        #self.episode_reward = self.episode_reward.at[reset_idx].set(0)
        # Return all observations without resetting
        self.observation = jnp.stack([t2j(o) for o in self.env.get_from_scenario(True, False, False, False)[0]], axis=1)

        # Next obs
        if self.sampled_epochs < self.config["random_epochs"]:
            self.random_sampling = True
            self.action = jnp.stack(
                [t2j(_get_random_action(a, True, self.env)) for a in self.env.agents], axis=1)
        else:
            self.random_sampling = False
            key, action_key = random.split(key)
            action_key = random.split(action_key, self.num_envs * self.num_agents).reshape(self.num_envs, self.num_agents, 2)
            # Policy must already be vmapped correctly
            self.action = policy(
                self.observation,
                action_noise,
                action_key,
            )

        # Equivalent to torch.unbind
        action = np.swapaxes(self.action, 0, 1)
        action = [j2t(s.squeeze(0)) for s in np.split(np.swapaxes(self.action, 0, 1), action.shape[0], axis=0)]

        #self.action = None
        (
            next_observation,
            reward,
            done,
            _,
        ) = self.env.step(action)
        if any(done):
            breakpoint()

        self.next_observation = jnp.stack([t2j(t) for t in next_observation], axis=1)
        self.reward = jnp.stack([t2j(r) for r in reward], axis=1)
        self.episode_reward += self.reward
        # dlpack doesnt work for torch/jax bools
        self.done = t2j(done.to(int)).reshape(self.num_envs, 1).repeat(self.num_agents, 1).astype(bool)

        transitions = {
            "observation": self.observation.copy(),
            "action": self.action.copy(), 
            "next_reward": self.reward.copy(), 
            "next_observation": self.next_observation.copy(),
            "next_done": self.done.copy(),
        }

        self.sampled_frames += 1 * self.num_envs
        self.sampled_epochs += 1
        if self.episode_reward.max() > self.best_reward:
            self.best_reward = self.episode_reward.max()
        return transitions, self.episode_reward.max(), self.best_reward

