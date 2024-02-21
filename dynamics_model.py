import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import tqdm
import pandas as pd
from modules import Block
from dataset import dataset_from_csv, split_dataset


class StateTransitionModel(eqx.Module):
  mlp: eqx.nn.Linear
  num_actions: int

  def __init__(self, state_size, num_actions, dropout, key):
    keys = jax.random.split(key, 3)
    self.num_actions = num_actions
    self.mlp = eqx.nn.Sequential([
      Block(state_size + num_actions, 256, dropout, key=keys[0]),
      Block(256, 256, dropout, key=keys[1]),
      eqx.nn.Linear(256, state_size, key=keys[2])
    ])

  def __call__(self, state, action):
    one_hot_action = jax.nn.one_hot(action, self.num_actions)
    pred_next_state = self.mlp(jnp.concatenate([state, one_hot_action], axis=-1))
    return pred_next_state

def loss_fn(model, state, action, next_state):
  pred_next_state = eqx.filter_vmap(model)(state, action)
  mae = jnp.mean(jnp.abs(pred_next_state - next_state), axis=0)
  return jnp.mean(0.5 * (pred_next_state - next_state) ** 2), mae

batch_size = 64
epochs = 10_000
key = jax.random.PRNGKey(0)
key, model_key, data_key = jax.random.split(key, 3)
#  TODO: Should include velocity
model = StateTransitionModel(state_size=3, num_actions=9, dropout=0, key=model_key)
data, data_size = dataset_from_csv(
  "/local/scratch/sm2558/general_value/data/random-1hz-fixedspeedactions-1/robomaster_1/rl_statesactions_tuple/rl_tuples.csv"
)
train, test, val = split_dataset(data, data_size, key=data_key)
lr_schedule = optax.constant_schedule(0.0001)
opt = optax.chain(
    optax.adamw(lr_schedule),
)
opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
pbar = tqdm.tqdm(total=epochs)
for epoch in range(epochs):
  pbar.update()
  key, sample_key = jax.random.split(key)
  batch_idx = jax.random.randint(sample_key, (batch_size,), 0, data_size)
  state = train["state"][batch_idx]
  action = train["action"][batch_idx]
  next_state = train["next_state"][batch_idx]

  (loss, info), grad = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn, has_aux=True))(model, state, action, next_state)
  val_loss, val_mae = eqx.filter_jit(loss_fn)(model, val["state"], val["action"], val["next_state"])
  updates, opt_state = opt.update(
    grad, opt_state, params=eqx.filter(model, eqx.is_inexact_array)
  )
  model = eqx.apply_updates(model, updates)
  pbar.set_description(f"loss: {loss:0.4f}, val_loss: {val_loss:0.4f}, val_mae (px, py): {val_mae[0]:.4f}, {val_mae[1]:.4f}")