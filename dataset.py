from typing import Dict
import enum
import pandas as pd
import jax
import jax.numpy as jnp
import flashbax as fbx

STATE_IDX = {
    "x_pos": jnp.array([0]),
    "y_pos": jnp.array([1]), 
    "pos": jnp.array([0, 1]) ,
    "x_vel": jnp.array([2]),
    "y_vel": jnp.array([3]), 
    "vel": jnp.array([2, 3]) 
}


def dataset_from_csv(path: str, normalize=False) -> Dict[str, jax.Array]:
    """Load dataset from CSV."""
    df = pd.read_csv(path)
    # TODO: Must do next_state before split
    #df['next_state'] = df['state']
    #df['state'] = df['state'].shift(-1)
    # Discard final transition which will have NaN state
    #data = data[:, :-1]
    data = df.to_dict(orient=list)
    data = {k: jnp.array(v) for k, v in data.items()}
    return data, len(df)

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
    train_idx = 0 < rand < train_split
    test_idx = train_split < rand < (train_split + test_split)
    val_idx = (train_split + test_split) < rand < (train_split + test_split + val_split)
    train = {k: v[train_idx] for k, v in dataset.items()}
    test = {k: v[test_idx] for k, v in dataset.items()}
    val = {k: v[val_idx] for k, v in dataset.items()}
    return train, test, val

def replay_buffer_from_csv(path: str, batch_size: int = 1024, key=jax.random.PRNGKey(0)):
    """Build a fbx replay buffer from the given dataset"""
    dataset, size = dataset_from_csv(path, key, train_split=1.0, test_split=0.1, val_split=0.0)
    buffer = fbx.make_prioritised_flat_buffer(
        max_length=size,
        min_length=1,
        sample_batch_size=batch_size,
        add_sequences=True
    )
    bufstate = buffer.init({k: v[0] for k, v in dataset.items()})
    return buffer, bufstate
