from rewards import *
import random

#from angle_emb import AnglE
from sentence_transformers import SentenceTransformer
import numpy as np
from constants import ARENA_BOUNDS_E, ARENA_BOUNDS_N
import h5py

random.seed(0)

#prompt = "You are a holonomic wheeled robot in a multirobot system,"
prompt = "Agent 0,"
#llm = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').to("cpu")
#llm = SentenceTransformer('paraphrase-MiniLM-L6-v2')
#llm = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)
llm = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')


def make_navigation_goals_and_embeddings(llm, num_tasks=1_000):
    task_strings = []
    task_embeddings = []
    goals = []
    eps = 0.3
    for i in range(num_tasks):
        x = random.uniform(ARENA_BOUNDS_E[0] + eps, ARENA_BOUNDS_E[1] - eps)
        y = random.uniform(ARENA_BOUNDS_N[0] + eps, ARENA_BOUNDS_N[1] - eps)

        command_strings = [
            f"navigate to ({x:0.2f}, {y:0.2f})",
        ]
        idx = random.randint(0, len(command_strings) - 1)
        task_str = f"{prompt} {command_strings[idx]}"
        task_strings.append(task_str)
        #task_embeddings.append(emb)
        goals.append((x, y))

    task_embeddings = llm.encode(task_strings)
    assert len(task_strings) == len(task_embeddings) == num_tasks

    return {
        "task_string": task_strings,
        "task_embedding": np.stack(task_embeddings, axis=0),
        "goal": np.array(goals)
    }


def make_language_navigation_tasks(eval=False):
    task_strings = []
    task_embeddings = []
    goals = []
    eps = 0.75

    # (N, E)
    command_locs = {
        "west edge": np.array([0, ARENA_BOUNDS_E[0] + eps]),
        "east edge": np.array([0, ARENA_BOUNDS_E[1] - eps]),
        "south edge": np.array([ARENA_BOUNDS_N[0] + eps, 0]),
        "north edge": np.array([ARENA_BOUNDS_N[1] - eps, 0]),
    }
    command_locs.update({
        "south west corner": command_locs["west edge"] + command_locs["south edge"],
        "west south corner": command_locs["west edge"] + command_locs["south edge"],
        "south east corner": command_locs["east edge"] + command_locs["south edge"],
        "east south corner": command_locs["east edge"] + command_locs["south edge"],
        "north west corner": command_locs["west edge"] + command_locs["north edge"],
        "west north corner": command_locs["west edge"] + command_locs["north edge"],
    })
    ne = {
            "north east corner": command_locs["east edge"] + command_locs["north edge"],
            #"east north corner": command_locs["east edge"] + command_locs["north edge"],
        }
    string_permutations = [
        "navigate to the {}",
        "pathfind to the {}",
        "find your way to the {}",
        "move to the {}",
        "your goal is the {}",
        "make your way to the {}",
        "head towards the {}",
        "travel to the {}",
        "reach the {}",
        "proceed to the {}",
        "{} is your target",
    ]
    eval_string_permutations = [
        "go to the {}",
    ]
    
    if eval:
        command_locs = {
            "west edge": np.array([0, ARENA_BOUNDS_E[0] + eps]),
            "east edge": np.array([0, ARENA_BOUNDS_E[1] - eps]),
            "south edge": np.array([ARENA_BOUNDS_N[0] + eps, 0]),
            "north edge": np.array([ARENA_BOUNDS_N[1] - eps, 0]),
        }
        command_locs.update({
            "south west corner": command_locs["west edge"] + command_locs["south edge"],
            "south east corner": command_locs["east edge"] + command_locs["south edge"],
            "north west corner": command_locs["west edge"] + command_locs["north edge"],
        })
        command_locs = {**command_locs, **ne} 
        string_permutations = eval_string_permutations
    for c, goal in command_locs.items():
        for s in string_permutations:
        # TODO: Show it works for coordinates, then train on N,S,E and show it works for west even if not trained?
        # TODO: Generate more data from simulator, can still be "offline"
        # Make sure we handle the boundaries by staying in for one frame then resetting

        #task_str = f"{prompt} navigate to the {c}"
            task_str = f"{prompt} {s.format(c)}"
            task_strings.append(task_str)
            goals.append(goal)
    task_embeddings = llm.encode(task_strings)
    def reward_fn(dataset, goal):
        # Dataset shape: [B, 2]
        # Goal shape: [G, 2]
        # Output shape: [B, G]
        # TODO: Add boundary reward
        return (
            relative_goal_pos_reward(dataset, goal) 
            #0.1 * speed_reward(dataset),
            #- 0.01 * goal_vel_reward(dataset, np.zeros_like(goal)) 
            #+ goal_pos_done(dataset, goal, 0.1)
            - 2 * boundary_reward(dataset, ARENA_BOUNDS_E, ARENA_BOUNDS_N).squeeze(-1)
        )

    def done_fn(dataset, goal):
        return (
            jnp.repeat(boundary_done(dataset, ARENA_BOUNDS_E, ARENA_BOUNDS_N).squeeze(-1), goal.shape[1], axis=-1)
            | goal_pos_done(dataset, goal, 0.1)
            #     #& goal_vel_done(dataset, np.zeros_like(goal), 0.1)
            # )
        )
    
    reward_kwargs = {"goal": np.array(goals)}
    assert len(task_strings) == len(task_embeddings) == reward_kwargs["goal"].shape[0]

    return {
        "task_string": task_strings,
        #"task_embedding": np.concatenate(task_embeddings, axis=0),
        "task_embedding": np.stack(task_embeddings, axis=0),
        "reward_function": reward_fn,
        "done_function": done_fn,
        "reward_kwargs": reward_kwargs
    }

def add_rewards_to_dataset(dataset, reward_dict):
    """Compute the cartesian product of transition tuples and rewards.
    
    In other words, compute each reward function for each (s, a, s') tuple.

    Takes a dataset of shape [B, ...] and a reward dict of shape [G, ...]

    This should return an updated dataset of shape [B, G, ...], with new "reward" and "task string" keys.
    Note that some values will be [B, 1, ...] or [1, G, ...] if they are constant across tasks or transitions.
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
    stacked_dones = reward_dict["done_function"](stacked_dataset, stacked_goals).reshape(B, G, 1)
    stacked_task_embeddings = reward_dict["task_embedding"].reshape(1, G, -1)
    #assert stacked_rewards.shape == (B, G)

    stacked_dataset.update({
        "next_reward": stacked_rewards,
        "task_embedding": stacked_task_embeddings,
        "next_done": stacked_dones,
    })

    return stacked_dataset

def merge_reward_datasets(datasets):
    for d in datasets:
        assert d['state'].shape[0] == datasets[0]['state'].shape[0]
    
    B = datasets[0]['state'].shape[0]
    G = datasets[0]['next_reward'].shape[1]

    # Merge rewards and dones
    rewards = np.concatenate([d['next_reward'] for d in datasets], axis=1)
    dones = np.concatenate([d['next_done'] for d in datasets], axis=1)
    embeds = np.concatenate([d['task_embedding'] for d in datasets], axis=1)
    datasets[0].update({
        "next_reward": rewards,
        "next_done": dones,
        "task_embedding": embeds
    })
    return datasets[0]
    

def make_dataset_ma(dataset):
    """Take the dataset and concatenate s, a, r, s' ...
    such that we have [s_1, s_2, ..., s_agents], [a_1, a_2, ..., a_agents]...

    Since our GNN is permutation invariant, we do not need every possible permutation of s.
    Rather, we just need one root s and the rest can be appended in any order. We will do
    a combination rather than permutation (permutation invariance!).
    """
    # Input shapes: [Batch, Task, F]
    # Output shapes: [Batch, Agent, Task, F]
    #data = {k: np.expand_dims(v, 1) for k,v in dataset.items()}
    B = dataset['state'].shape[0]
    
    # Tile instead of repeat for better "shuffling" of data
    root_data = {
        k: np.tile(v, (1, B, 1, 1)) 
        for k,v in dataset.items()
    }



