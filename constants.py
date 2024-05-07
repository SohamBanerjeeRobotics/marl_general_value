import jax.numpy as jnp


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
ACTION_VEL["0"] = jnp.array([0.0, 0.0])
ACTION_MAPPING = {ACTION_IDX[s].item(): ACTION_VEL[s] for s in ACTION_IDX}

ARENA_BOUNDS_N = (-1.95, 1.95)
ARENA_BOUNDS_E = (-1.95, 1.95)
