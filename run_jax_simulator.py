from dynamics_model import StateTransitionModel
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import h5py
import tqdm
from rewards2 import point_navigation_done, point_navigation_reward
import wandb

from dynamics_multicollector import DynamicsMultiCollector
from modules import GeneralQNetwork, epsilon_greedy_policy
from losses import update_general_qnet_simple
from tasks import make_navigation_goals_and_embeddings



# opt setup
config = {
    "seed": 0,
    "lr": 0.0001,
    "gamma": jnp.array([0.99]),
    "batch_size": 1,
    "tau": jnp.array([1/200]),
    "epsilon": jnp.array([0.2]),
    "num_envs": 8,
    "random_epochs": 100,
    "eval_num_envs": 16,
    "eval_timesteps": 400,
    "eval_interval": 100,
    "epochs": 20_000,
    "eval_episodes": 10,
    "q_config": {
        "mlp_size": 256,
        "head_size": 256,
        "ensemble_size": 1,
        "dropout": 0.1,
    },
    "task_size": 768,
    "obs_size": 5,
    "act_size": 9,
    "simulator_weights": "data/dynamics_model_weights.eqx",
}
wandb.init('morlmarl', config=config)

key = jax.random.PRNGKey(config["seed"])
key, eval_key = jax.random.split(key)

lr_schedule = optax.constant_schedule(config["lr"])
opt = optax.chain(
    #optax.clip_by_global_norm(config["train"]["gradient_scale"]),
    optax.adamw(lr_schedule),
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

simulator = StateTransitionModel(
    state_size=config["obs_size"], 
    num_actions=config["act_size"], 
    dropout=0, 
    key=jax.random.PRNGKey(0)
)
simulator = eqx.tree_deserialise_leaves(config["simulator_weights"], simulator)
nav_goals_and_embs = make_navigation_goals_and_embeddings(config["num_envs"])
collector = DynamicsMultiCollector(
    sim=simulator, 
    reward_fn=[point_navigation_reward], 
    done_fn=[point_navigation_done], 
    goals=[nav_goals_and_embs["goal"]], 
    embeddings=[nav_goals_and_embs["task_embedding"]], 
    num_envs=config["num_envs"],
    random_epochs=config["random_epochs"],
    config=config
)
eval_nav_goals_and_embs = make_navigation_goals_and_embeddings(config["eval_num_envs"])
eval_collector = DynamicsMultiCollector(
    sim=simulator, 
    reward_fn=[point_navigation_reward], 
    done_fn=[point_navigation_done], 
    goals=[eval_nav_goals_and_embs["goal"]], 
    embeddings=[eval_nav_goals_and_embs["task_embedding"]], 
    num_envs=config["eval_num_envs"],
    random_epochs=0,
    config=config
)

td_error = jnp.array([jnp.inf])
pbar = tqdm.tqdm(total=config["epochs"])
cstate = collector.initial_state()
eval_mean_return = -jnp.inf
for epoch in range(config["epochs"]):
    key, collect_key = jax.random.split(key)
    # TODO: We should have a last_episode_reward matrix that we update so we have a
    # mean return for each timestep
    transitions, cstate = eqx.filter_jit(collector)(q_function, cstate, epsilon_greedy_policy, config["epsilon"], collect_key)
    print(cstate['action'])

    q_function, q_target, td_error, qvalue, qtarget_value = eqx.filter_jit(update_general_qnet_simple)(q_function, q_target, transitions, opt, opt_state, config["gamma"], config["tau"], key)
    out_str = (
        f"Epoch {epoch}/{config['epochs']} "
        f"ql: {td_error.mean():0.4f} "
        f"qv: {qvalue.mean():0.4f} "
        f"qtv: {qtarget_value.mean():0.4f} "
        f"train: {cstate['terminal_episode_return']: 0.2f} "
        f"eval: {eval_mean_return: 0.2f}"
    )
    pbar.set_description(out_str)
    pbar.update()
    wandb.log(
        {
            "train/loss": td_error.mean(),
            "train/epoch": epoch,
            "train/q_value_mean": qvalue.mean(),
            "train/q_target_value_mean": qtarget_value.mean(),
            "train/terminal_episode_return": cstate["terminal_episode_return"],
            "eval/mean_return": eval_mean_return,
        }
    )

    # Eval
    if epoch % config["eval_interval"] == 0:
        # Reset all envs
        ecstate = eval_collector.initial_state()
        ecstate["done"] = jnp.ones_like(ecstate["done"])

        completed_episodes = jnp.zeros_like(ecstate["done"])
        episode_rewards = jnp.zeros_like(ecstate["episode_reward"])
        step = 0
        eval_q_function = eqx.nn.inference_mode(q_function)
        while not jnp.all(completed_episodes):
            etransitions, ecstate = eqx.filter_jit(eval_collector)(eval_q_function, ecstate, epsilon_greedy_policy, 0, eval_key)
            # First terminal episode for env
            reward_mask = ecstate["done"] & ~completed_episodes
            episode_rewards = episode_rewards * ~reward_mask + ecstate["episode_reward"] * reward_mask
            completed_episodes = completed_episodes + ecstate["done"]
            step += 1

            if step >= config["eval_timesteps"]:
                episode_rewards = episode_rewards * reward_mask + ecstate["episode_reward"] * ~reward_mask
                break

        eval_mean_return = jnp.mean(episode_rewards)

    # # Eval
    # ep_rewards = 0
    # eval_q_function = eqx.nn.inference_mode(q_function)
    # final_dists = []
    # for i in range(config["eval_episodes"]):
    #     agent_state = jnp.array([0.0, 0.0, 0, 0, 0])
    #     done = False
    #     eval_task = {
    #         "task_string": eval_tasks["task_string"][i:i+1],
    #         "task_embedding": eval_tasks["task_embedding"][i:i+1],
    #         "reward_function": eval_tasks["reward_function"],
    #         "done_function": eval_tasks["done_function"],
    #         "reward_kwargs": {"goal": eval_tasks["reward_kwargs"]["goal"][i:i+1]},
    #     }
    #     ep_reward = 0
    #     num_steps = 0
    #     while not done and num_steps < 200:
    #         action = greedy_policy(eval_q_function, agent_state, eval_tasks["task_embedding"][i], key=jax.random.PRNGKey(0))
    #         next_state = simulator(agent_state, action)
    #         reward_fn_inputs = {
    #             "state": agent_state.reshape(1, -1),
    #             "action": action.reshape(1, -1),
    #             "next_state": next_state.reshape(1, -1),
    #         }
    #         result = add_rewards_to_dataset(reward_fn_inputs, eval_task)
    #         reward, done = result['next_reward'].reshape(1), result['next_done'].reshape(1)

    #         agent_state = next_state
    #         ep_reward += reward
    #         num_steps += 1
    #     ep_rewards += ep_reward
    #     final_dists.append(jnp.linalg.norm(agent_state[:2] - eval_tasks["reward_kwargs"]["goal"][i]).item())
    # wandb.log({
    #     "eval/mean_return": ep_rewards.item() / config['eval_episodes'],
    #     "eval/mean_dist2goal": jnp.mean(jnp.array(final_dists)),
    #     "train/loss": td_error.mean(),
    #     "train/epoch": epoch,
    #     "train/q_value_mean": qvalue.mean(),
    #     "train/q_target_value_mean": qtarget_value.mean()
    # })
    # print(f"Episode reward: {ep_rewards.item() / config['eval_episodes']:.2f}, final dist {jnp.mean(jnp.array(final_dists))}")



