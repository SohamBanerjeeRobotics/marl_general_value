import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import tqdm

from modules import GeneralQNetwork, greedy_policy
from losses import update_general_qnet, update_qnet
from dataset import dataset_from_csv, replay_buffer_from_csv
import tasks

seed = 0
batch_size = 128
tau = jnp.array(1 / 200)
num_agents = 1
num_envs = 128
key = jax.random.PRNGKey(seed)
#env = vmas.make_env("sampling", num_envs=num_envs)
epochs = 1_000
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

q_function = GeneralQNetwork(obs_size=5, task_size=1024, act_size=9, config=q_config, key=key)
q_target = GeneralQNetwork(obs_size=5, task_size=1024, act_size=9, config=q_config, key=key)
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
keys = jax.random.split(jax.random.PRNGKey(0), data_with_rewards['next_reward'].shape[:-1]).reshape(*data_with_rewards['next_reward'].shape[:-1], -1)
test = eqx.filter_vmap(eqx.filter_vmap(q_function))(data_with_rewards['state'], data_with_rewards['task_embedding'], keys)

test_data = {k: v[:10] for k, v in data_with_rewards.items()}
_, _, td_error, qvalue = update_general_qnet(q_function, q_target, test_data, opt, opt_state, gamma, tau, key)


# TODO: Utilize negative reward for leaving boundaries
# Apply/create dones
# Predict dones?

# metrics
td_error = jnp.array([jnp.inf])

num_batches = (data_size + batch_size - 1) // batch_size
pbar = tqdm.tqdm(total=num_batches)
for epoch in range(epochs):
    #key, buffer_key, train_key = jax.random.split(key, 3)

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, data_size)
        data_batch = {k: v[start_idx:end_idx] for k, v in data_with_rewards.items()}

        key, _ = jax.random.split(key)
        q_function, q_target, td_error, qvalue = eqx.filter_jit(update_general_qnet)(q_function, q_target, data_batch, opt, opt_state, gamma, tau, key)
        out_str = f"Epoch {epoch}/{epochs} ql: {td_error.mean():0.4f} "
        pbar.set_description(out_str)
        pbar.update()
