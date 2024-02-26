import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import tqdm
import pandas as pd
from modules import Block
from dataset import dataset_from_csv, split_dataset, ACTION_VEL
from dynamics_model import StateTransitionModel


def loss_fn(model, state, action, next_state):
  pred_next_state = eqx.filter_vmap(model)(state, action)
  mae = jnp.mean(jnp.abs(pred_next_state - next_state), axis=0)
  return jnp.mean(jnp.abs(pred_next_state - next_state)), mae

def multistep_loss_fn(model, state, action, next_state, length=10):
  for l in range(length):
    state = eqx.filter_vmap(model)(state, action)
    state = state[:-1]
    action = action[1:]

  #state = state[:length]
  #next_state = next_state[-length:]
  next_state = next_state[length:]
  mae = jnp.mean(0.5 * (state - next_state) ** 2, axis=0)
  return jnp.mean(0.5 * (state - next_state) ** 2), mae


def multistep_integrator(state, action, next_state, length=10):
  vel_map = list(ACTION_VEL.values()) 
  for l in range(length):
    vel = jnp.stack([vel_map[a] for a in action])
    state = jnp.concatenate([
      state[:, 0:2] + vel,
      vel,
      state[:, 4:5], # yaw
      #state[2:5]
    ], axis=-1)
    state = state[:-1]
    action = action[1:]
  next_state = next_state[length:]
  mae = jnp.mean(0.5 * (state - next_state) ** 2, axis=0)
  return jnp.mean(0.5 * (state - next_state) ** 2), mae

batch_size = 64
epochs = 10_000
key = jax.random.PRNGKey(0)
key, model_key, data_key = jax.random.split(key, 3)
#  TODO: Should include velocity
model = StateTransitionModel(state_size=5, num_actions=9, dropout=0, key=model_key)

datasets = [
  "data/random-1hz-fixspd-fulllog-1/robomaster_1/rl_statesactions_tuple/rl_tuples.csv",
  "data/random-1hz-fulllog-2/robomaster_1/rl_statesactions_tuple/rl_tuples.csv",
]
data, data_size = dataset_from_csv(datasets)
train, test, val = split_dataset(data, data_size, key=data_key)
lr_schedule = optax.constant_schedule(0.00001)
opt = optax.chain(
    optax.adamw(lr_schedule),
)
opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

def train_fn(model, train, val, opt_state, sample_key):
  batch_idx = jax.random.randint(sample_key, (batch_size,), 0, data_size)
  state = train["state"][batch_idx]
  action = train["action"][batch_idx]
  next_state = train["next_state"][batch_idx]

  (loss, _), grad = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model, state, action, next_state)
  updates, opt_state = opt.update(
    grad, opt_state, params=eqx.filter(model, eqx.is_inexact_array)
  )
  model = eqx.apply_updates(model, updates)
  val_loss, val_mae = loss_fn(model, val["state"], val["action"], val["next_state"])
  return model, loss, val_loss, val_mae, opt_state, jax.random.split(sample_key)[0]


best_model = None
best_val_loss = jnp.inf
pbar = tqdm.tqdm(total=epochs)
for epoch in range(epochs):
  model, loss, val_loss, val_mae, opt_state, key = eqx.filter_jit(train_fn)(model, train, val, opt_state, key)
  if val_loss < best_val_loss:
    best_model = model
    best_val_loss = val_loss
  pbar.update()
  pbar.set_description(f"loss: {loss:0.4f}, val_loss: {val_loss:0.4f}, best: {best_val_loss:0.4f}, val_mae (px, py, vx, vy): {val_mae[0]:.3f}, {val_mae[1]:.3f}, {val_mae[2]:.3f}, {val_mae[3]:.3f}")

test_loss, test_mae = eqx.filter_jit(loss_fn)(best_model, test["state"], test["action"], test["next_state"])
# Can't use val/train/test as they will be shuffled and these must be in order
# They must even be in the same dataset, so just take the first few
multistep_loss, multistep_mae = eqx.filter_jit(multistep_loss_fn)(best_model, data["state"][:50], data["action"][:50], data["next_state"][:50])
integrator_loss, integrator_mae = multistep_integrator(data["state"][:50], data["action"][:50], data["next_state"][:50])


print(f"Best model test loss: {test_loss}, MAE (x, y, vx, vy), test_mae (px, py, vx, vy): {test_mae[0]:.4f}, {test_mae[1]:.4f}, {test_mae[2]:.4f}, {test_mae[3]:.4f}")
print(f"Integrator 10-step integrator loss: {integrator_loss}, MAE (x, y, vx, vy), integrator_mae (px, py, vx, vy): {integrator_mae[0]:.4f}, {integrator_mae[1]:.4f}, {integrator_mae[2]:.4f}, {integrator_mae[3]:.4f}")
print(f"Best model 10-step multistep loss: {multistep_loss}, MAE (x, y, vx, vy), multistep_mae (px, py, vx, vy): {multistep_mae[0]:.4f}, {multistep_mae[1]:.4f}, {multistep_mae[2]:.4f}, {multistep_mae[3]:.4f}")
model.save("data/dynamics_model_weights.eqx")
eqx.tree_serialise_leaves("data/dynamics_model_weights.eqx", model)