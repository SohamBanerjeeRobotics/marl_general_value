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

from constants import ACTION_MAPPING

config = {
    "seed": 0,
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

# Model config and setup
q_function = GeneralQNetwork(
    obs_size=config["obs_size"], 
    task_size=config["task_size"], 
    act_size=config["act_size"], 
    config=config["q_config"], 
    key=jax.random.PRNGKey(0)
)
q_function = eqx.tree_deserialise_leaves(f"models/ne-0-950-1.94.eqx", q_function)

def policy_wrapper(q_function, state, task_embedding):
    action_idx = eqx.filter_jit(greedy_policy)(q_function, state, task_embedding, key=jax.random.PRNGKey(0))
    return action_idx

# To map states and tasks to velocities, use policy wrapper
# e.g., policy_wrapper(state_numpy_vec, mappings['Agent 0, navigate to the west edge'])
# where state_numpy_vec is of shape 5: (e, n, ve, vn, yaw_angle)
# policy_wrapper(jnp.array([0, 0, 0, 0, 0]), mappings['Agent 0, navigate to the west edge'])
