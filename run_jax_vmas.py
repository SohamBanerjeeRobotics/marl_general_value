import vmas
import jax
import jax.numpy as jnp
import torch
import equinox as eqx
import optax
import flashbax as fbx
import tqdm

from modules import QNetwork, Policy
from losses import update_critic, update_actor
from vmas_multicollector import VMASMultiCollector

seed = 0
batch_size = 1024
noise_scale = jnp.array(0.2)
tau = jnp.array(1 / 200)
num_agents = 1
num_envs = 128
key = jax.random.PRNGKey(seed)
#env = vmas.make_env("football", num_envs=1, n_red_agents=0)
#env = vmas.make_env("football", num_envs=num_envs, n_red_agents=0, n_blue_agents=1)
env = vmas.make_env("sampling", num_envs=num_envs)
epochs = 10_000
random_epochs = 8_000
actor_delay = 4
gamma = jnp.array([0.99])


# opt setup
critic_lr_schedule = optax.constant_schedule(0.0001)

critic_opt = optax.chain(
    #optax.clip_by_global_norm(config["train"]["gradient_scale"]),
    optax.adamw(critic_lr_schedule),
    #optax.adamw(lr_schedule, weight_decay=config["train"]["weight_decay"], eps=config["train"]["adam_eps"])
)
actor_lr_schedule = optax.constant_schedule(0.00002)

actor_opt = optax.chain(
    #optax.clip_by_global_norm(config["train"]["gradient_scale"]),
    optax.adamw(critic_lr_schedule),
    #optax.adamw(lr_schedule, weight_decay=config["train"]["weight_decay"], eps=config["train"]["adam_eps"])
)
buffer = fbx.make_prioritised_flat_buffer(
    max_length=10_000,
    min_length=batch_size,
    sample_batch_size=batch_size,
    add_sequences=True,
)

# Q value
#critic = QNetwork(8, 2, key)
critic = QNetwork(24, 2, key)
marl_critic = eqx.filter_vmap(critic)
#actor = Policy(8, env.action_space[0].low, env.action_space[0].high, key=key)
actor = Policy(24, -jnp.ones(2), jnp.ones(2), key=key)
marl_actor = eqx.filter_vmap(actor, in_axes=(0, None, 0))
keys = jax.random.split(key, num_agents)

critic_opt_state = critic_opt.init(eqx.filter(critic, eqx.is_inexact_array))
actor_opt_state = actor_opt.init(eqx.filter(actor, eqx.is_inexact_array))

'''
obs = jnp.stack(env.observation_space.sample())
act = jnp.stack(env.action_space.sample())
marl_critic(obs, act)
marl_actor(obs, noise_scale, keys)

obs2 = jnp.stack(env.observation_space.sample())
act2 = jnp.stack(env.action_space.sample())
obs_batch = jnp.stack([obs, obs2])
act_batch = jnp.stack([act, act2])
rewards = jax.random.uniform(key, shape=(2, num_agents))
terminated = jnp.zeros((2, num_agents), dtype=bool)
truncated = jnp.zeros((2, num_agents), dtype=bool)

tape = {"observation": obs_batch, "next_observation": obs_batch, "next_reward": rewards, "action": act_batch, "next_terminated": terminated, "next_truncated": truncated}
'''

collector_config = {
    "collect": {
        "random_epochs": random_epochs
    }
}
collector = VMASMultiCollector(env, collector_config)
bufstate = None

# metrics
actor_loss = jnp.array([jnp.inf])
td_error = jnp.array([jnp.inf])

pbar = tqdm.tqdm(total=epochs)
for epoch in range(epochs):
    pbar.update()
    out_str = ""
    _, collect_key, buffer_key, critic_key, actor_key = jax.random.split(key, 5)
    transitions, running_reward, best_reward = collector(eqx.filter_vmap(marl_actor, in_axes=(0, None, 0)), noise_scale, collect_key)
    if bufstate == None:
        bufstate = buffer.init({k: v[0] for k, v in transitions.items()})
    bufstate = jax.jit(buffer.add, donate_argnums=0)(bufstate, transitions)
    out_str += f"rew: {transitions['next_reward'].mean():0.3f}, best: {best_reward:0.3f} "

    if not collector.random_sampling:
        batch = jax.jit(buffer.sample)(bufstate, buffer_key)
        critic, td_error, qvalue = eqx.filter_jit(update_critic)(critic, critic, actor, batch.experience[0], critic_opt, critic_opt_state, gamma, noise_scale, tau, critic_key)
        # Reduce td_error over agent dim since transitions contain all agents
        bufstate = jax.jit(buffer.set_priorities, donate_argnums=0)(bufstate, batch.indices, td_error.mean(-1))
        out_str += f"ql: {td_error.mean():0.4f} "
        out_str += f"qvm: {jnp.abs(qvalue).mean():0.2f} "
        out_str += f"actl: {actor_loss.mean():0.3f}"
        pbar.set_description(out_str)

        if epoch % actor_delay == 0:
            actor, actor_loss = eqx.filter_jit(update_actor)(actor, critic, batch.experience[0], actor_opt, actor_opt_state, noise_scale, actor_key)

    #print(out_str)
