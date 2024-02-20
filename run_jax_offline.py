import vmas
import jax
import jax.numpy as jnp
import torch
import equinox as eqx
import optax
import flashbax as fbx
import tqdm

from modules import EnsembleQNetwork, epsilon_greedy
from losses import update_critic, update_actor
from vmas_multicollector import VMASMultiCollector
from dataset import replay_buffer_from_csv

seed = 0
batch_size = 1024
tau = jnp.array(1 / 200)
num_agents = 1
num_envs = 128
key = jax.random.PRNGKey(seed)
#env = vmas.make_env("sampling", num_envs=num_envs)
epochs = 10_000
gamma = jnp.array([0.99])


# opt setup
lr_schedule = optax.constant_schedule(0.0001)
opt = optax.chain(
    #optax.clip_by_global_norm(config["train"]["gradient_scale"]),
    optax.adamw(lr_schedule),
)

# buffer = fbx.make_prioritised_flat_buffer(
#     max_length=10_000,
#     min_length=batch_size,
#     sample_batch_size=batch_size,
#     add_sequences=True,
# )

# Q value
#critic = QNetwork(8, 2, key)
critic = EnsembleQNetwork(24, 2, key)
marl_critic = eqx.filter_vmap(critic)
actor = epsilon_greedy
marl_actor = eqx.filter_vmap(actor, in_axes=(0, None, 0))
keys = jax.random.split(key, num_agents)

opt_state = opt.init(eqx.filter(critic, eqx.is_inexact_array))

# collector = VMASMultiCollector(env, collector_config)
buffer, bufstate = replay_buffer_from_csv("PATH", batch_size)

# metrics
actor_loss = jnp.array([jnp.inf])
td_error = jnp.array([jnp.inf])

pbar = tqdm.tqdm(total=epochs)
for epoch in range(epochs):
    pbar.update()
    out_str = ""
    _, collect_key, buffer_key, critic_key, actor_key = jax.random.split(key, 5)
    transitions, running_reward, best_reward = collector(eqx.filter_vmap(marl_actor, in_axes=(0, None, 0)), noise_scale, collect_key)
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
