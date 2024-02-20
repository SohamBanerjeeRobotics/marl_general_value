import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import pandas as pd
from ..modules import Block


class StateTransitionModel(eqx.Module):
  mlp: eqx.nn.Linear

  def __init__(self, state_size=6, action_size=3, dropout=0.05, key):
    keys = jax.random.split(key, 3)
    self.mlp = eqx.nn.Sequential([
      Block(state_size + action_size, 256, dropout, key=keys[0]),
      Block(256, 256, dropout, key=keys[1]),
      eqx.nn.Linear(256, state_size, key=keys[2])
    ])

  def __call__(self, state, action):
    pred_next_state = self.mlp(jnp.concatenate(state, action, axis=-1))
    return pred_next_state

def loss_fn(model, state, action, next_state):
  pred_next_state =  model(state, action)
  return 0.5 * (pred_next_state - next_state) ** 2


key = jax.random.PRNGKey(0)
model = StateTransitionModel()
dataset = pd.read_csv("../data/randwalk_img2/robomaster_0/current_state/pos.csv")
opt = optax.adamw() # Might need to initialize optax state too
for epoch in range(500):
  data = get_data_from_rosbags()
  grad = eqx.filter_grad(loss_fn)(model, data.state, data.action, data.next_state)
  updates, opt_state = opt.update(
    grad, opt_state, params=eqx.filter(model, eqx.is_inexact_array)
  )
  model = eqx.apply_updates(model, updates)