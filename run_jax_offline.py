from dynamics_model import StateTransitionModel
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
import h5py
import tqdm
import wandb

from modules import GeneralQNetwork, greedy_policy
from losses import update_general_qnet
from tasks import add_rewards_to_dataset, make_language_navigation_tasks
from evaluate_policy import evaluate_policy
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("-w", "--wandb", action="store_true")
args = parser.parse_args()

# opt setup
config = {
    "seed": args.seed,
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
if args.wandb:
    wandb.init('morlmarl', config=config)

key = jax.random.PRNGKey(config["seed"])

lr_schedule = optax.constant_schedule(config["lr"])
opt = optax.chain(
    #optax.clip_by_global_norm(config["train"]["gradient_scale"]),
    optax.adamw(lr_schedule, weight_decay=config["weight_decay"]),
)

q_function = GeneralQNetwork(
    obs_size=config["obs_size"], 
    task_size=config["task_size"], 
    act_size=config["act_size"], 
    config=config["q_config"], 
    key=key
)
q_target = GeneralQNetwork(
    obs_size=config["obs_size"], 
    task_size=config["task_size"], 
    act_size=config["act_size"], 
    config=config["q_config"], 
    key=key
)
opt_state = opt.init(eqx.filter(q_function, eqx.is_inexact_array))

dataset_with_str = h5py.File("dataset.h5", "r")
dataset = {k: jnp.array(v) for k,v in dataset_with_str.items() if k != 'task_string'} 
data_size = dataset['next_reward'].shape[0]

simulator = StateTransitionModel(
    state_size=config["obs_size"], 
    num_actions=config["act_size"], 
    dropout=0, 
    key=jax.random.PRNGKey(0)
)
simulator = eqx.tree_deserialise_leaves(config["simulator_weights"], simulator)
#eval_tasks = make_global_navigation_tasks(config["eval_episodes"])
eval_tasks = make_language_navigation_tasks(eval=True)

# B, num_goals, S
#test_data = {k: v[:1] for k, v in dataset.items()}
#update_general_qnet(q_function, q_target, test_data, opt, opt_state, config["gamma"], config["tau"], config["loss"], key)


# TODO: Utilize negative reward for leaving boundaries
# Apply/create dones
# Predict dones?

# metrics
td_error = jnp.array([jnp.inf])

num_batches = (data_size + config["batch_size"] - 1) // config["batch_size"]
pbar = tqdm.tqdm(total=config["epochs"])
best_eval_return = -np.inf
eval_return = -np.inf
for epoch in range(1, config["epochs"]):
    for i in range(num_batches):
        start_idx = i * config["batch_size"]
        end_idx = min((i + 1) * config["batch_size"], data_size)
        data_batch = {
            k: v[0:1] if k == "task_embedding" else v[start_idx:end_idx] 
            for k, v in dataset.items()
        }

        key, _ = jax.random.split(key)
        q_function, q_target, td_error, qvalue, qtarget_value = eqx.filter_jit(update_general_qnet)(q_function, q_target, data_batch, opt, opt_state, config["gamma"], config["tau"], config["loss"], key)
    out_str = f"Epoch {epoch}/{config['epochs']} ql: {td_error.mean():0.3f} qv: {qvalue.mean():0.3f} ret: {eval_return:.2f} best: {best_eval_return:.2f}"
    pbar.set_description(out_str)
    pbar.update()
    if args.wandb:
        wandb.log({
            "train/loss": td_error.mean(),
            "train/epoch": epoch,
            "train/q_value_mean": qvalue.mean(),
            "train/q_target_value_mean": qtarget_value.mean()
        })

    if epoch % config["eval_interval"] == 0 or epoch == 1:
        # Eval
        eval_q_function = eqx.nn.inference_mode(q_function)
        data, goals, frames, rewards = evaluate_policy(q_function=eval_q_function)
        mean_eval_dist = jnp.linalg.norm(data['next_state'][...,:2] - goals, axis=-1).mean()


        eval_return = rewards.sum(0).mean()
        if eval_return > best_eval_return:
            best_eval_return = eval_return
            eqx.tree_serialise_leaves(f"models/ne-{config['seed']}-{epoch}-{eval_return:0.2f}.eqx", q_function)

        video = jnp.transpose(frames, (0, 3, 1, 2))
        if args.wandb:
            video = wandb.Video(np.array(video), fps=10)
            wandb.log({
                "eval/mean_return": eval_return,
                "eval/mean_distance": mean_eval_dist,
                "eval/video": video,
                "eval/best_return": best_eval_return,
                "train/epoch": epoch,
            }, step=epoch)
        