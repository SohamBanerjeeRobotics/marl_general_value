import argparse
from dynamics_model import StateTransitionModel
from evaluate_policy import evaluate_ma_policy, MARLEnv
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
import h5py
import tqdm
import wandb

from modules import GeneralMAQNetwork, GeneralQNetwork, greedy_policy
from losses import update_general_qnet, update_general_qnet_ma
from tasks import add_rewards_to_dataset, make_language_navigation_tasks
from rewards import ma_collision_reward_and_done


# TODO: We are reaching deadlocks because we cannot rely on the other agent making a speicifc move. This is a downside of the dataset

# opt setup
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("-w", "--wandb", action="store_true")
args = parser.parse_args()

# opt setup
num_agents = 3
config = {
    "seed": args.seed,
    "lr": 0.0001,
    "loss": "weighted",
    "weight_decay": 0.0001,
    "warmup_epochs": 100,
    "gamma": jnp.array([0.95]),
    "batch_size": 256,
    "num_agents": num_agents,
    "tau": jnp.array([1/2000]),
    "epochs": 100_000,
    "eval_interval": 1000,
    "eval_trials": 5,
    "q_config": {
        "mlp_size": 1024,
        "head_size": 1024,
        "dropout": 0.0,
        "ensemble_size": 2,
        "ensemble_reduce": "min",
    },
    "task_size": 768,
    "obs_size": 4,
    "act_size": 9,
    "simulator_weights": "data/dynamics_model_weights.eqx",
}
if args.wandb:
    wandb.init(project='morlmarl', config=config)

key = jax.random.PRNGKey(config["seed"])
global_fn = jax.jit(jax.vmap(ma_collision_reward_and_done), donate_argnums=(2,3))

lr_warmup = optax.linear_schedule(config["lr"] * 0.01, config["lr"], config["warmup_epochs"])
lr_train = optax.constant_schedule(config["lr"])
lr_schedule = optax.join_schedules([lr_warmup, lr_train], [config["warmup_epochs"]])
opt = optax.chain(
    #optax.clip_by_global_norm(config["train"]["gradient_scale"]),
    optax.adamw(lr_schedule, weight_decay=config["weight_decay"]),
)

q_function = GeneralMAQNetwork(
    obs_size=config["obs_size"], 
    task_size=config["task_size"], 
    act_size=config["act_size"], 
    config=config["q_config"], 
    key=key,
)
q_target = GeneralMAQNetwork(
    obs_size=config["obs_size"], 
    task_size=config["task_size"], 
    act_size=config["act_size"], 
    config=config["q_config"], 
    key=key,
)
opt_state = opt.init(eqx.filter(q_function, eqx.is_inexact_array))

dataset_with_str = h5py.File("dataset.h5", "r")
dataset = {k: jnp.array(v) for k,v in dataset_with_str.items() if k != 'task_string'} 
data_size = dataset['next_reward'].shape[0]
eval_tasks = make_language_navigation_tasks(True)

simulator = MARLEnv(num_agents=config["num_agents"])

# B, num_goals, S
#test_data = {k: v[:1] for k, v in dataset.items()}
#update_general_qnet(q_function, q_target, test_data, opt, opt_state, config["gamma"], config["tau"], key)


# TODO: Utilize negative reward for leaving boundaries
# Apply/create dones
# Predict dones?

# metrics
td_error = jnp.array([jnp.inf])

num_batches = (data_size + config["batch_size"] - 1) // config["batch_size"]
pbar = tqdm.tqdm(total=config["epochs"])
best_eval_return = -np.inf
best_eval_distance = np.inf
eval_return = -np.inf
for epoch in range(1, config["epochs"]):
    key, task_key = jax.random.split(key)
#    start_idx = i * config["batch_size"]
#    end_idx = min((i + 1) * config["batch_size"], data_size)
#    batch_size = end_idx - start_idx
    # 4000 C 5 is ~10^15 datapoints which is too much to materialize
    # Instead, let us just sample
    # But issue: HDF5 does not support random access, only sequential
    # Solution, first sample a slice, then build MA batch
    #batch_idx = jnp.repeat(jnp.arange(start_idx, end_idx), config["num_agents"])
    batch_idx = jax.random.randint(task_key, (config["batch_size"] * config["num_agents"],), 0, data_size)

    # Sample without replacement, but only along the agent axis
    # This way, each agent has a unique task, but the task can appear multiple times across a batch
    sample_keys = jax.random.split(task_key, config["batch_size"])
    sampled_task_idx = jax.vmap(jax.random.choice, in_axes=(0, None, None, None))(
        sample_keys, dataset["task_embedding"].shape[1], (config["num_agents"],), False
    ).reshape(-1)
    # Batch will be of shape [B, A]
    ma_data_batch = {
        "next_reward": dataset["next_reward"][batch_idx, sampled_task_idx],
        "next_done": dataset["next_done"][batch_idx, sampled_task_idx],
        "task_embedding": dataset["task_embedding"][0, sampled_task_idx],
        "state": dataset["state"][batch_idx],
        "next_state": dataset["next_state"][batch_idx],
        "action": dataset["action"][batch_idx],
    }
    ma_data_batch = {k: v.reshape(config["batch_size"], config["num_agents"], -1) for k, v in ma_data_batch.items()}
    # Tack on custom reward for collisions which requires global state
    r, d = global_fn(
        ma_data_batch['state'], ma_data_batch['next_state'], ma_data_batch['next_reward'], ma_data_batch['next_done']
    )
    ma_data_batch['next_reward'] = r
    ma_data_batch['next_done'] = d
    q_function, q_target, td_error, qvalue, qtarget_value = eqx.filter_jit(update_general_qnet_ma)(q_function, q_target, ma_data_batch, opt, opt_state, config["gamma"], config["tau"], config["loss"], key)

    out_str = f"Epoch {epoch}/{config['epochs']} ql: {td_error.mean():0.3f} qv: {qvalue.mean():0.3f} ret: {eval_return:.2f} best: {best_eval_return:.2f}"
    pbar.set_description(out_str)
    pbar.update()
    if args.wandb:
        wandb.log({
            "train/loss": td_error.mean(),
            "train/epoch": epoch,
            "train/q_value_mean": qvalue.mean(),
            "train/q_target_value_mean": qtarget_value.mean(),
            "train/done_density": ma_data_batch['next_done'].mean(),
            "train/reward": ma_data_batch['next_reward'].mean(),
        })

    if epoch % config["eval_interval"] == 0:
        # Eval
        eval_q_function = eqx.nn.inference_mode(q_function)
        mean_eval_distance = eval_return = eval_collisions = 0
        all_frames = []
        for i in range(config["eval_trials"]):
            data, goals, frames, rewards, dones = evaluate_ma_policy(env=simulator, tasks=eval_tasks, config=config, q_function=eval_q_function, seed=i)
            mean_eval_distance += jnp.linalg.norm(data['next_state'][...,:2] - goals, axis=-1).mean()
            eval_return += rewards.sum(0).mean()
            eval_collisions += dones.sum() / 2
            all_frames.append(frames)


        mean_eval_distance /= config["eval_trials"]
        eval_return /= config["eval_trials"]
        all_frames = jnp.concatenate(all_frames, axis=0)

        if eval_return > best_eval_return:
            best_eval_return = eval_return
        if mean_eval_distance < best_eval_distance:
            best_eval_distance = mean_eval_distance

        eqx.tree_serialise_leaves(f"models/ne-nagents-{config['num_agents']}-seed-{config['seed']}-epoch-{epoch}-return-{eval_return:0.2f}.eqx", q_function)
        video = jnp.transpose(all_frames, (0, 3, 1, 2))
        if args.wandb:
            video = wandb.Video(np.array(video), fps=10)
            wandb.log({
                "eval/mean_return": eval_return,
                "eval/mean_distance": mean_eval_distance,
                "eval/video": video,
                "eval/best_return": best_eval_return,
                "eval/best_distance": best_eval_distance,
                "eval/collisions": eval_collisions,
                "train/epoch": epoch,
            }, step=epoch)
        
        



