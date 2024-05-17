import equinox as eqx
from equinox import nn
import jax
import jax.numpy as jnp
from jax import random
import pickle
from typing import Dict, Any
from modules import GeneralMAQNetwork, GeneralQNetwork, greedy_policy
import yaml



# Load embeddings and corresponding natural language commands
train_mappings = pickle.loads(open('robomaster_control_train.pkl', "rb").read())
eval_mappings = pickle.loads(open('robomaster_control_eval.pkl', "rb").read())
# Convert action index to command

from constants import ACTION_MAPPING

cfg_path = "experiments/real_soft-2.0.yaml"
with open(cfg_path) as f:
    config = yaml.safe_load(f)

config['seed'] = 0
key = jax.random.PRNGKey(0)

# Model config and setup
ma_q_function = GeneralMAQNetwork(
    obs_size=config["obs_size"], 
    task_size=config["task_size"], 
    act_size=config["act_size"], 
    config=config["q_config"], 
    key=key,
)
model_weights = 'models/soft-2.0-126000.eqx'
ma_q_function = eqx.tree_deserialise_leaves(model_weights, ma_q_function)


def ma_policy_wrapper(ma_q_function, state, task_embedding, key):
    key, sample_key = jax.random.split(key)
    q_value = ma_q_function(state, task_embedding, jax.random.PRNGKey(0))
    return jax.random.categorical(sample_key, q_value * 100.0), key
    #return q_value.argmax(axis=-1)
