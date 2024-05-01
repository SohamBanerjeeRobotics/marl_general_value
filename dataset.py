from typing import Dict, List
import enum
import pandas as pd
import jax
import jax.numpy as jnp
import flashbax as fbx
import numpy as np


# TODO: The robot coordinates should be (N, E) not (E, N)
STATE_IDX = {
    "n_pos": jnp.array([0]), 
    "e_pos": jnp.array([1]),
    "pos": jnp.array([0, 1]) ,
    "n_vel": jnp.array([2]), 
    "e_vel": jnp.array([3]),
    "vel": jnp.array([2, 3]) 
}
IDX_STATE = {
    0: "n_pos",
    1: "e_pos",
    2: "n_vel",
    3: "e_vel",
}

# TODO Why are S, N swapped in dataset?
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
    "W": jnp.array([0, -1]),
    "SW": jnp.array([-1, -1]),
    "S": jnp.array([-1, 0]),
    "SE": jnp.array([-1, 1]),
    "E": jnp.array([0, 1]),
    "NE": jnp.array([1, 1]),
    "N": jnp.array([1, 0]),
    "NW": jnp.array([1, -1]),
}
ACTION_VEL = {k: 0.3 * (v / jnp.linalg.norm(v)) for k, v in ACTION_VEL.items()}
ACTION_VEL["0"] = jnp.array([0, 0])
ACTION_MAPPING = {ACTION_IDX[s].item(): ACTION_VEL[s] for s in ACTION_IDX}

ARENA_BOUNDS_N = (-2.0, 2.0)
ARENA_BOUNDS_E = (-2.0, 2.0)


def action_to_discrete(actions, threshold=0.1):
    """
    Converts batched continuous action vectors in the x, y plane to discrete actions using fixed-size masks and summation.
    :param actions: A batch of actions as a JAX array with shape (N, 2), where N is the batch size.
    :param threshold: A small value to decide when to treat actions as no-ops.
    :return: An array of integers representing the discrete actions for the batch.
    """
    # Decompose actions into x and y components
    n, e = actions[0], actions[1]
    
    # Compute masks for each action based on the input conditions
    not_n = (jnp.abs(n) < threshold)
    not_e = (jnp.abs(e) < threshold)
    no_op_mask = not_n & not_e
    west_mask = (e < -threshold) & not_n
    sw_mask = (e < -threshold) & (n < -threshold)
    south_mask = not_e & (n < -threshold)
    se_mask = (e > threshold) & (n < -threshold)
    east_mask = (e > threshold) & not_n
    ne_mask = (e > threshold) & (n > threshold)
    north_mask = (n > threshold) & not_e
    nw_mask = (e < -threshold) & (n > threshold)

    # Create an action score matrix where each row represents an action and each column represents a condition
    action_scores = jnp.stack([
        no_op_mask, west_mask, sw_mask, south_mask, se_mask, east_mask, ne_mask, north_mask, nw_mask
    ]).astype(jnp.float32)
    
    # Determine the action for each vector by finding the index of the highest score
    discrete_actions = jnp.argmax(action_scores)
    
    return discrete_actions

# Vectorize the function over a batch of actions
# Note: With this approach, the explicit use of vmap is not necessary since the function is inherently vectorized.

def test_action_to_discrete():
    # Example usage with a batch of actions
    actions_batch = jnp.array([
        [1e-5, 1e-5], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1]
    ])
    discrete_actions = jax.vmap(action_to_discrete, in_axes=(0, None))(actions_batch, 0.1)
    assert discrete_actions == jnp.arange(9)

def augment_dataset(dataset, size, augment_size, key, eps=0.01):
    """Augment the dataset with additional zero-velocity data."""
    keys = jax.random.split(key, 8)
    e_state = jax.random.uniform(keys[0], shape=(augment_size,), minval=ARENA_BOUNDS_E[0] - 0.05, maxval=ARENA_BOUNDS_E[1] + 0.05)
    n_state = jax.random.uniform(keys[1], shape=(augment_size,), minval=ARENA_BOUNDS_N[0] - 0.05, maxval=ARENA_BOUNDS_N[1] + 0.05)
    augment = {
        "state": np.stack([
            n_state,
            e_state,
            jax.random.normal(keys[2], shape=(augment_size,),) * eps, 
            jax.random.normal(keys[3], shape=(augment_size,),) * eps, 
        ], axis=1),
        "next_state": np.stack([
            n_state,
            e_state,
            jax.random.normal(keys[5], shape=(augment_size,),) * eps, 
            jax.random.normal(keys[6], shape=(augment_size,),) * eps, 
        ], axis=1),
        "action": jnp.ones((augment_size, 2), dtype=np.int32) * ACTION_VEL["0"]
    }
    data = {k: jnp.concatenate([v, augment[k]], axis=0)for k, v in dataset.items()}
    return data, size + augment_size

def filter_out_of_bounds(df):
    """Remove transitions where previous state is out of bounds."""
    return df[
        df['prev_state.pn'].between(ARENA_BOUNDS_N[0], ARENA_BOUNDS_N[1]) &
        df['prev_state.pe'].between(ARENA_BOUNDS_E[0], ARENA_BOUNDS_E[1])
    ]

def dataset_from_csv(paths: List[str], relative_pose: bool = True) -> Dict[str, jax.Array]:
    """Load dataset from CSV."""
    datas = []
    size = 0
    key = jax.random.PRNGKey(0)
    for path in paths:
        df = pd.read_csv(path)
        data = {
            "state": np.stack([
                df['prev_state.pn'], 
                df['prev_state.pe'], 
                df['prev_state.vn'], 
                df['prev_state.ve'], 
            ], axis=-1),
            "next_state": np.stack([
                df['curr_state.pn'], 
                df['curr_state.pe'], 
                df['curr_state.vn'], 
                df['curr_state.ve'], 
            ], axis=-1),
            "action": np.stack([df['prev_action.n'], df['prev_action.e']], axis=-1),
        }
        # Previous state for zeroth entry is not valid
        data = {k: v[1:] for k, v in data.items()}

        key, _ = jax.random.split(key)
        # Augment with data where the agent is not moving
        # So that we cover the state space
        #augment_size = data['state'].shape[0] // 8# Double null action
        #data, size = augment_dataset(data, size, augment_size, key)
        size += len(df) - 1

        datas.append(data)

    data = {key: None for key in datas[0].keys()} 
    for key in data.keys():
        data[key] = jnp.concatenate([d[key] for d in datas], axis=0)
    
    data["action"] = jax.vmap(action_to_discrete, in_axes=(0, None))(data["action"], 0.1)
    data = {k: jnp.array(v, copy=False) for k, v in data.items()}
    # Shuffle
    p = jax.random.permutation(jax.random.PRNGKey(0), size)
    data = {k: v[p] for k, v in data.items()}
    #plot_dataset_distribution(data)
    return data, size

def plot_dataset_distribution(dataset: Dict[str, jax.Array]):
    """Plot the distribution of the dataset."""
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].hist2d(
        np.array(dataset["state"][:, 0]),
        np.array(dataset["state"][:, 1])
    )
    axs[0].set_title(f"E, N pos distribution")
    axs[1].hist2d(
        np.array(dataset["state"][:, 2]),
        np.array(dataset["state"][:, 3])
    )
    axs[1].set_title(f"E, N vel distribution")

    d = np.diff(np.unique(dataset["action"])).min()
    left_of_first_bin = dataset["action"].min() - float(d)/2
    right_of_last_bin = dataset["action"].max() + float(d)/2
    axs[2].hist(dataset["action"], np.arange(left_of_first_bin, right_of_last_bin + d, d))
    axs[2].set_title("Action distribution")
    plt.tight_layout()
    plt.show()



def add_next_state(dataset: Dict[str, jax.Array]) -> Dict[str, jax.Array]:

    """Add next state to the dataset.
    
    WARNING: This assumes the entire dataset is contiguous (we can get next_state by shifting state by one)
    Therefore each dataset should contain only a SINGLE run/experiment.

    We throw away the final transition rather than add new 'done' values to the dataset.
    As this will result in a difficult objective, as the Q function will be unable
    to learn where the dones occur (because they are effectively random).
    """
    dataset['next_state'] = dataset['state'][1:]
    dataset = {k: v[:-1] for k, v in dataset.items()}
    return dataset

def split_dataset(dataset, size, train_split=0.8, test_split=0.15, val_split=0.05, key=jax.random.PRNGKey(0)):
    """Split a dataset in train, test, val splits."""
    rand = jax.random.uniform(key, (size,))
    train_idx = (0 < rand) & (rand < train_split)
    test_idx = (train_split < rand) & (rand < (train_split + test_split))
    val_idx = ((train_split + test_split) < rand) & (rand < (train_split + test_split + val_split))
    assert jnp.all(train_idx ^ test_idx ^ val_idx) # Make sure all sets distinct
    train = {k: v[train_idx] for k, v in dataset.items()}
    test = {k: v[test_idx] for k, v in dataset.items()}
    val = {k: v[val_idx] for k, v in dataset.items()}
    return train, test, val

def replay_buffer_from_csv(paths: List[str], batch_size: int = 1024, key=jax.random.PRNGKey(0)):
    """Build a fbx replay buffer from the given dataset"""
    dataset, size = dataset_from_csv(paths)
    buffer = fbx.make_prioritised_flat_buffer(
        max_length=size,
        min_length=1,
        sample_batch_size=batch_size,
        add_sequences=True
    )
    bufstate = buffer.init({k: v[0] for k, v in dataset.items()})
    return buffer, bufstate
