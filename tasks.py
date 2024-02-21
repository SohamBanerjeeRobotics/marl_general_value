from rewards import *
import random

from angle_emb import AnglE
import numpy as np

random.seed(0)

prompt = "You are a holonomic wheeled robot in a multirobot system."
llm = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()

def make_silly_tasks():
    task_strings = [
        "Make like a banana and split.",
        "Dance like a noodle in a pot of boiling water.",
        "Tell your socks to go on an adventure in the dryer.",
        "Act like a drum and beat it.",
        "Chase rainbows until you find a pot of Wi-Fi.",
        "Roll like a jellybean on a downhill adventure.",
        "Sit like a mushroom and be a fun-guy.",
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
            f"Navigate to ({x:0.1f}, {y:0.1f}) in the global coordinate frame.",
            f"Find a way to the global coordinates ({x:0.3f}, {y:0.3f}).",
            f"Get to ({x:0.1f}, {y:0.1f}).",
            f"You must find your way to ({x:0.3f}, {y:0.3f}) in a global coordinate system.",
            f"Plan and execute a path to coordinates ({x:0.2f}, {y:0.2f}).",
            f"Your objective is to arrive at global coordinates ({x:0.3f}, {y:0.3f}). Execute your objective.",
            f"Proceed to the coordinates ({x:0.3f}, {y:0.3f}).",
            f"Move towards location ({x:0.3f}, {y:0.3f}) in the worldwide coordinate system.",
            f"Direct yourself to the coordinates ({x:0.3f}, {y:0.3f}) on the global map.",
            f"Travel to global position ({x:0.3f}, {y:0.3f}).",
            f"Advance to the position ({x:0.2f}, {y:0.2f}).",
        ]
        idx = random.randint(0, len(command_strings) - 1)
        task_str = f"{prompt} {command_strings[idx]}"
        task_strings.append(task_str)
        emb = llm.encode(task_str, to_numpy=True)
        task_embeddings.append(emb)
        goals.append((x, y))

    def reward_fn(dataset, goal):
        return goal_pos_reward(dataset, goal) + 0.01 * goal_vel_reward(dataset, 0)
    
    reward_kwargs = {"goal": jnp.array(goals)}
    assert len(task_strings) == len(task_embeddings) == reward_kwargs["goal"].shape[0] == num_tasks

    return {
        "task_string": task_strings,
        "task_embedding": task_embeddings,
        "reward_function": reward_fn,
        "reward_kwargs": reward_kwargs
    }

tasks = make_global_navigation_tasks()
breakpoint()