from rewards import *
import random

from angle_emb import AnglE
import numpy as np
from dataset import ARENA_BOUNDS_E, ARENA_BOUNDS_N

random.seed(0)

#prompt = "You are a holonomic wheeled robot in a multirobot system,"
prompt = "You are holonomic robot in a multirobot system,"
llm = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()

def make_silly_tasks():
    task_strings = [
        "make like a banana and split.",
        "dance like a noodle in a pot of boiling water.",
        "tell your socks to go on an adventure in the dryer.",
        "act like a drum and beat it.",
        "chase rainbows until you find a pot of Wi-Fi.",
        "roll like a jellybean on a downhill adventure.",
        "sit like a mushroom and be a fun-guy.",
        "do what makes you happy."
    ]
    emb = jnp.concatenate([llm.encode(t, to_numpy=True) for t in task_strings])
    return {
        "task_string": task_strings,
        "task_embedding": emb,
        "reward_function": None,
        "reward_kwargs": {}
    }


def make_global_navigation_tasks(num_tasks=100):
    task_strings = []
    task_embeddings = []
    goals = []
    for i in range(num_tasks):
        x = random.uniform(-1.95, 1.95)
        y = random.uniform(-1.45, 1.45)

        command_strings = [
            f"navigate to ({x:0.1f}, {y:0.1f}) in the global coordinate frame.",
            f"find a way to the global coordinates ({x:0.3f}, {y:0.3f}).",
            f"get to ({x:0.1f}, {y:0.1f}).",
            f"you must find your way to ({x:0.3f}, {y:0.3f}) in a global coordinate system.",
            f"plan and execute a path to coordinates ({x:0.2f}, {y:0.2f}).",
            f"your objective is to arrive at global coordinates ({x:0.3f}, {y:0.3f}). Execute your objective.",
            f"proceed to the coordinates ({x:0.3f}, {y:0.3f}).",
            f"move towards location ({x:0.3f}, {y:0.3f}) in the worldwide coordinate system.",
            f"direct yourself to the coordinates ({x:0.3f}, {y:0.3f}) on the global map.",
            f"travel to global position ({x:0.3f}, {y:0.3f}).",
            f"advance to the position ({x:0.2f}, {y:0.2f}).",
        ]
        idx = random.randint(0, len(command_strings) - 1)
        task_str = f"{prompt} {command_strings[idx]}"
        task_strings.append(task_str)
        emb = llm.encode(task_str, to_numpy=True)
        task_embeddings.append(emb)
        goals.append((x, y))

    def reward_fn(dataset, goal):
        # Dataset shape: [B, 2]
        # Goal shape: [G, 2]
        # Output shape: [B, G]
        # TODO: Add boundary reward
        return goal_pos_reward(dataset, goal) + 0.01 * goal_vel_reward(dataset, 0) #+ boundary_reward(dataset, ARENA_BOUNDS_E, ARENA_BOUNDS_N)
    
    reward_kwargs = {"goal": jnp.array(goals)}
    assert len(task_strings) == len(task_embeddings) == reward_kwargs["goal"].shape[0] == num_tasks

    return {
        "task_string": task_strings,
        "task_embedding": jnp.concatenate(task_embeddings, axis=0),
        "reward_function": reward_fn,
        "reward_kwargs": reward_kwargs
    }

def compute_rewards(dataset, reward_dict):
    """Compute the cartesian product of transition tuples and rewards.
    
    In other words, compute each reward function for each (s, a, s') tuple.

    Takes a dataset of shape [B, ...] and a reward dict of shape [G, ...]

    This should return an updated dataset of shape [B, G, ...], with new "reward" and "task string" keys.
    """
    # Compute dims
    B = dataset['state'].shape[0]
    G = reward_dict["reward_kwargs"]["goal"].shape[0]

    stacked_dataset = {
        k: v.reshape(B, 1, -1)
        for k, v in dataset.items()
    }

    # TODO: Do not rely on goal, generalize
    stacked_goals = reward_dict["reward_kwargs"]["goal"].reshape(1, G, -1)
    stacked_rewards = reward_dict["reward_function"](stacked_dataset, stacked_goals).reshape(B, G, 1)
    #assert stacked_rewards.shape == (B, G)


    stacked_task_embeddings = jnp.repeat(jnp.array(reward_dict["task_embedding"]).reshape(1, G, -1), B, axis=0)

    # Repeat dataset over each reward/task
    stacked_dataset = {
        k: jnp.repeat(v, G, axis=1)
        for k, v in stacked_dataset.items()
    }

    stacked_dataset.update({
        "next_reward": stacked_rewards,
        "task_embedding": stacked_task_embeddings,
        "next_done": jnp.zeros(stacked_rewards.shape, dtype=bool)
    })
    return stacked_dataset
    




#ALL_TASKS = make_global_navigation_tasks()
#breakpoint()