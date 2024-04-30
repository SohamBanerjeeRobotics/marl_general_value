import equinox as eqx
from equinox import nn
import jax
import jax.numpy as jnp
from jax import random
import pickle
from typing import Dict, Any
from modules import GeneralQNetwork, greedy_policy



# Load embeddings and corresponding natural language commands
mappings = pickle.loads(open('robomaster_control.pkl', "rb").read())
# Convert action index to command
ACTION_IDX = {
    "0": jnp.array(0),
    "W": jnp.array(1),
    "SW": jnp.array(2),
    "N": jnp.array(3),
    "SE": jnp.array(4),
    "E": jnp.array(5),
    "NE": jnp.array(6),
    "S": jnp.array(7),
    "NW": jnp.array(8),
}
ACTION_VEL = {
    "0": jnp.array([0, 0]),
    "W": jnp.array([-1, 0]),
    "SW": jnp.array([-1, -1]),
    "S": jnp.array([0, -1]),
    "SE": jnp.array([1, -1]),
    "E": jnp.array([1, 0]),
    "NE": jnp.array([1, 1]),
    "N": jnp.array([0, 1]),
    "NW": jnp.array([-1, 1]),
}
ACTION_VEL = {k: 0.3 * (v / jnp.linalg.norm(v)) for k, v in ACTION_VEL.items()}
ACTION_VEL["0"] = jnp.array([0, 0])
ACTION_MAPPING = {ACTION_IDX[s].item(): ACTION_VEL[s] for s in ACTION_IDX}

# Model config and setup
config = {
    "seed": 0,
    "lr": 0.0001,
    "weight_decay": 0.001,
    "gamma": jnp.array([0.95]),
    "batch_size": 32,
    "tau": jnp.array([1/1000]),
    "epochs": 3000,
    "eval_interval": 20,
    "eval_episodes": 1,
    "q_config": {
        "mlp_size": 384,
        "head_size": 384,
        "ensemble_size": 1,
        "dropout": 0.0,
    },
    "task_size": 768,
    "obs_size": 5,
    "act_size": 9,
    "simulator_weights": "data/dynamics_model_weights.eqx",
}
q_function = GeneralQNetwork(
    obs_size=config["obs_size"], 
    task_size=config["task_size"], 
    act_size=config["act_size"], 
    config=config["q_config"], 
    key=jax.random.PRNGKey(0)
)
#q_function = eqx.tree_deserialise_leaves(f"models/ne-{config['seed']}-{epoch}-{eval_score:0.2f}.eqx", q_function)
q_function = eqx.tree_deserialise_leaves(f"models/ne-0-760-2.93.eqx", q_function)
def policy_wrapper(state, task_embedding):

    action_idx = eqx.filter_jit(greedy_policy)(q_function, state, task_embedding, key=jax.random.PRNGKey(0))
    action_vel = ACTION_MAPPING[action_idx.item()]
    return action_vel

# To map states and tasks to velocities, use policy wrapper
# e.g., policy_wrapper(state_numpy_vec, mappings['Agent 0, navigate to the west edge'])
# where state_numpy_vec is of shape 5: (e, n, ve, vn, yaw_angle)
# policy_wrapper(jnp.array([0, 0, 0, 0, 0]), mappings['Agent 0, navigate to the west edge'])
