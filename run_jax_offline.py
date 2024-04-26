from dynamics_model import StateTransitionModel
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import h5py
import tqdm

from modules import GeneralQNetwork, greedy_policy
from losses import update_general_qnet
from tasks import add_rewards_to_dataset, make_global_navigation_tasks

seed = 0
batch_size = 1
tau = jnp.array(1 / 200)
num_agents = 1
num_envs = 128
key = jax.random.PRNGKey(seed)
#env = vmas.make_env("sampling", num_envs=num_envs)
epochs = 10
gamma = jnp.array([0.99])
eval_episodes = 10


# opt setup
lr_schedule = optax.constant_schedule(0.0001)
opt = optax.chain(
    #optax.clip_by_global_norm(config["train"]["gradient_scale"]),
    optax.adamw(lr_schedule),
)
q_config = {
    "mlp_size": 256,
    "head_size": 256,
    "ensemble_size": 1,
    "dropout": 0.1,
}

q_function = GeneralQNetwork(obs_size=5, task_size=768, act_size=9, config=q_config, key=key)
q_target = GeneralQNetwork(obs_size=5, task_size=768, act_size=9, config=q_config, key=key)
opt_state = opt.init(eqx.filter(q_function, eqx.is_inexact_array))

dataset_with_str = h5py.File("dataset.h5", "r")
dataset = {k: v for k,v in dataset_with_str.items() if k != 'task_string'} 
data_size = dataset['next_reward'].shape[0]

simulator = StateTransitionModel(state_size=5, num_actions=9, dropout=0, key=jax.random.PRNGKey(0))
simulator = eqx.tree_deserialise_leaves("data/dynamics_model_weights.eqx", simulator)
eval_tasks = make_global_navigation_tasks(eval_episodes)

# B, num_goals, S
test_data = {k: v[:1] for k, v in dataset.items()}
update_general_qnet(q_function, q_target, test_data, opt, opt_state, gamma, tau, key)


# TODO: Utilize negative reward for leaving boundaries
# Apply/create dones
# Predict dones?

# metrics
td_error = jnp.array([jnp.inf])

num_batches = (data_size + batch_size - 1) // batch_size
for epoch in range(epochs):
    pbar = tqdm.tqdm(total=num_batches)
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, data_size)
        data_batch = {
            k: v[0:1] if k == "task_embedding" else v[start_idx:end_idx] 
            for k, v in dataset.items()
        }

        key, _ = jax.random.split(key)
        q_function, q_target, td_error, qvalue, qtarget_value = eqx.filter_jit(update_general_qnet)(q_function, q_target, data_batch, opt, opt_state, gamma, tau, key)
        out_str = f"Epoch {epoch}/{epochs} ql: {td_error.mean():0.4f} qv: {qvalue.mean():0.4f} qtv: {qtarget_value.mean():0.4f}"
        pbar.set_description(out_str)
        pbar.update()

    # Eval
    ep_rewards = 0
    eval_q_function = eqx.nn.inference_mode(q_function)
    for i in range(eval_episodes):
        agent_state = jnp.array([0.0, 0.0, 0, 0, 0])
        done = False
        eval_task = {
            "task_string": eval_tasks["task_string"][i:i+1],
            "task_embedding": eval_tasks["task_embedding"][i:i+1],
            "reward_function": eval_tasks["reward_function"],
            "done_function": eval_tasks["done_function"],
            "reward_kwargs": {"goal": eval_tasks["reward_kwargs"]["goal"][i:i+1]},
        }
        ep_reward = 0
        num_steps = 0
        while not done and num_steps < 1000:
            action = greedy_policy(eval_q_function, agent_state, eval_tasks["task_embedding"][i], key=jax.random.PRNGKey(0))
            next_state = simulator(agent_state, action)
            reward_fn_inputs = {
                "state": agent_state.reshape(1, -1),
                "action": action.reshape(1, -1),
                "next_state": next_state.reshape(1, -1),
            }
            result = add_rewards_to_dataset(reward_fn_inputs, eval_task)
            reward, done = result['next_reward'].reshape(1), result['next_done'].reshape(1)

            agent_state = next_state
            ep_reward += reward
            num_steps += 1
        ep_rewards += ep_reward
    print(f"Episode reward: {ep_rewards.item() / eval_episodes:.2f}")



