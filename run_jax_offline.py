import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import tqdm

from modules import GeneralQNetwork, greedy_policy
from losses import update_qnet
from dataset import dataset_from_csv, replay_buffer_from_csv
import tasks

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
q_config = {
    "mlp_size": 256,
    "head_size": 256,
    "ensemble_size": 5,
    "dropout": 0.0,
}

q_function = GeneralQNetwork(obs_size=24, task_size=1024, act_size=2, config=q_config, key=key)
q_target = GeneralQNetwork(obs_size=24, task_size=1024, act_size=2, config=q_config, key=key)
opt_state = opt.init(eqx.filter(q_function, eqx.is_inexact_array))

buffer, bufstate = replay_buffer_from_csv(
    [
        "data/rand-1hz-sticky-1/robomaster_1/rl_statesactions_tuple/rl_tuples.csv",
        "data/rand-1hz-sticky-2/robomaster_1/rl_statesactions_tuple/rl_tuples.csv",
        "data/rand-1hz-sticky-3/robomaster_1/rl_statesactions_tuple/rl_tuples.csv"
    ],
    batch_size
)
datasets = [
  "data/rand-1hz-sticky-1/robomaster_1/rl_statesactions_tuple/rl_tuples.csv",
  "data/rand-1hz-sticky-2/robomaster_1/rl_statesactions_tuple/rl_tuples.csv",
  "data/rand-1hz-sticky-3/robomaster_1/rl_statesactions_tuple/rl_tuples.csv"
]
data, data_size = dataset_from_csv(datasets)

all_tasks = tasks.make_global_navigation_tasks(3)
# "task_string": task_strings,
# "task_embedding": task_embeddings,
# "reward_function": reward_fn,
# "reward_kwargs": reward_kwargs

reward_fn = all_tasks['reward_function']
data_with_rewards = tasks.compute_rewards(data, all_tasks)
# B, num_goals, S
breakpoint()


# metrics
actor_loss = jnp.array([jnp.inf])
td_error = jnp.array([jnp.inf])

pbar = tqdm.tqdm(total=epochs)
for epoch in range(epochs):
    pbar.update()
    out_str = ""
    key, buffer_key, train_key = jax.random.split(key, 3)

    batch = jax.jit(buffer.sample)(bufstate, buffer_key)
    critic, td_error, qvalue = eqx.filter_jit(update_qnet)(q_function, q_target, batch.experience[0], opt, opt_state, gamma, tau, train_key)
    # Reduce td_error over agent dim since transitions contain all agents
    bufstate = jax.jit(buffer.set_priorities, donate_argnums=0)(bufstate, batch.indices, td_error.mean(-1))
    out_str += f"ql: {td_error.mean():0.4f} "
    out_str += f"qvm: {jnp.abs(qvalue).mean():0.2f} "
    out_str += f"actl: {actor_loss.mean():0.3f}"
    pbar.set_description(out_str)
