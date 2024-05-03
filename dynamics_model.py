import jax
import numpy as np
import jax.numpy as jnp
import equinox as eqx
from constants import ACTION_IDX, ARENA_BOUNDS_E, ARENA_BOUNDS_N


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

  def initial_state(self, key):
    keys = jax.random.split(key, 5)
    return jnp.concatenate([
      jax.random.uniform(keys[0], shape=(1,), minval=ARENA_BOUNDS_E[0], maxval=ARENA_BOUNDS_E[1]),
      jax.random.uniform(keys[1], shape=(1,), minval=ARENA_BOUNDS_N[0], maxval=ARENA_BOUNDS_N[1]),
      jnp.zeros((3,))
    ])


if __name__ == '__main__':
  print(ACTION_IDX)
  model = StateTransitionModel(state_size=4, num_actions=9, dropout=0, key=jax.random.PRNGKey(0))
  model = eqx.tree_deserialise_leaves("data/dynamics_model_weights.eqx", model)
  deltas = {}
  for action, idx in ACTION_IDX.items():
    print(f"action {action}: {model(jnp.array([1.0, 1.0, 0, 0]), idx)}")
  for action, idx in ACTION_IDX.items():
    print(f"action {action} delta: {jnp.array([1.0, 1.0, 0, 0]) - model(jnp.array([1.0, 1.0, 0, 0]), idx)}")