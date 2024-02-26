import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import tqdm
import pandas as pd
from modules import Block
from dataset import dataset_from_csv, split_dataset, ACTION_IDX


class StateTransitionModel(eqx.Module):
  mlp: eqx.nn.Linear
  num_actions: int

  def __init__(self, state_size, num_actions, dropout, key):
    keys = jax.random.split(key, 4)
    self.num_actions = num_actions
    hidden_size = 512
    self.mlp = eqx.nn.Sequential([
      eqx.nn.Linear(state_size + num_actions, hidden_size, key=keys[0]), eqx.nn.RMSNorm((hidden_size,)), eqx.nn.Lambda(jax.nn.gelu),
      eqx.nn.Linear(hidden_size, hidden_size, key=keys[1]), eqx.nn.RMSNorm((hidden_size,)), eqx.nn.Lambda(jax.nn.gelu),
      eqx.nn.Linear(hidden_size, hidden_size, key=keys[2]), eqx.nn.RMSNorm((hidden_size,)), eqx.nn.Lambda(jax.nn.gelu),
      eqx.nn.Linear(hidden_size, state_size, key=keys[3])
    ])

  def save(self, path: str):
    eqx.tree_serialise_leaves(path, self)

  def load(self, path: str):
    eqx.tree_deserialise_leaves(path, self)

  def __call__(self, state, action):
    one_hot_action = jax.nn.one_hot(action, self.num_actions)
    pred_next_state = state + self.mlp(jnp.concatenate([state, one_hot_action], axis=-1))
    return pred_next_state

if __name__ == '__main__':
  print(ACTION_IDX)
  model = StateTransitionModel(state_size=5, num_actions=9, dropout=0, key=jax.random.PRNGKey(0))
  model = eqx.tree_deserialise_leaves("data/dynamics_model_weights.eqx", model)
  for action, idx in ACTION_IDX.items():
    print(f"action {action}: {model(jnp.array([1.0, 1.0, 0, 0, 0]), idx)}")
  breakpoint()