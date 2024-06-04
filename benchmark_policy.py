from evaluate_policy import MARLEnv
import jax.numpy as jnp
import numpy as np
import jax
import equinox as eqx
from modules import GeneralMAQNetwork
import time

num_agents = 3
config = {
    "seed": 0,
    "lr": 0.0001,
    "loss": "meanq",
    "weight_decay": 0.0001,
    "warmup_epochs": 100,
    "gamma": jnp.array([0.95]),
    "batch_size": 256,
    "num_agents": num_agents,
    "tau": jnp.array([1/2000]),
    "epochs": 200_000,
    "eval_interval": 1000,
    "eval_trials": 5,
    "q_config": {
        "mlp_size": 1024,
        "head_size": 1024,
        "dropout": 0.0,
        "ensemble_size": 2,
        "ensemble_reduce": "min",
    },
    "task_size": 768,
    "obs_size": 4,
    "act_size": 9,
    "simulator_weights": "data/dynamics_model_weights.eqx",
}

key = jax.random.PRNGKey(config["seed"])

q_function = eqx.filter_jit(GeneralMAQNetwork(
    obs_size=config["obs_size"], 
    task_size=config["task_size"], 
    act_size=config["act_size"], 
    config=config["q_config"], 
    key=key,
))

num_samples = 10
means = {}
stds = {}
for n in range(1, 6):
    simulator = MARLEnv(num_agents=n)
    state = simulator.reset(key)
    latent = jax.random.uniform(key, (n, config["task_size"]))
    # Warm cache
    q_function(state, latent, key).max(-1).block_until_ready()

    results = []
    for i in range(num_samples):
        start = time.time()
        q_function(state, latent, key).max(-1).block_until_ready()
        end = time.time()
        results.append(end - start)
    
    means[n] = np.mean(results) * 1000
    stds[n] = np.std(results) * 1000

print(means)
print(stds)
    
    




