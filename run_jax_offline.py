import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import tqdm

from modules import EnsembleQNetwork, greedy_policy
from losses import update_qnet
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
q_config = {
    "mlp_size": 256,
    "head_size": 256,
    "ensemble_size": 5,
    "dropout": 0.0,
}

q_function = EnsembleQNetwork(24, 2, key)
q_target = EnsembleQNetwork(24, 2, key)
opt_state = opt.init(eqx.filter(q_function, eqx.is_inexact_array))

buffer, bufstate = replay_buffer_from_csv(
    "data/random-1hz-fixedspeedactions-1/robomaster_1/rl_statesactions_tuple/rl_tuples.csv", 
    batch_size
)

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
