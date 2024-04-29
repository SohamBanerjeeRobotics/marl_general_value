from rewards import *
import random

#from angle_emb import AnglE
from sentence_transformers import SentenceTransformer
import numpy as np
from dataset import ARENA_BOUNDS_E, ARENA_BOUNDS_N, dataset_from_csv
import h5py

random.seed(0)

#prompt = "You are a holonomic wheeled robot in a multirobot system,"
prompt = "Agent 0,"
#llm = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').to("cpu")
#llm = SentenceTransformer('paraphrase-MiniLM-L6-v2')
#llm = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)
llm = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

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
    emb = np.concatenate([llm.encode(t, to_numpy=True) for t in task_strings])
    return {
        "task_string": task_strings,
        "task_embedding": emb,
        "reward_function": None,
        "reward_kwargs": {}
    }


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

    command_locs = {
        "west edge": np.array([ARENA_BOUNDS_E[0] + eps, 0]),
        "east edge": np.array([ARENA_BOUNDS_E[1] - eps, 0]),
        "south edge": np.array([0, ARENA_BOUNDS_N[0] + eps]),
        "north edge": np.array([0, ARENA_BOUNDS_N[1] - eps]),
    }
    command_locs.update({
        "south west edge": command_locs["west edge"] + command_locs["south edge"],
        "south east edge": command_locs["east edge"] + command_locs["south edge"],
        "north west edge": command_locs["west edge"] + command_locs["north edge"],
        #"north east edge": command_locs["east edge"] + command_locs["north edge"],
    })
    ne = {
            "north east edge": command_locs["east edge"] + command_locs["north edge"],
        }
    if eval:
        command_locs = ne + {

            #"north edge": np.array([0, ARENA_BOUNDS_N[1] - eps]),
            #"east edge": np.array([ARENA_BOUNDS_E[1] - eps, 0]),
        }
    else:
        command_locs.update(ne)
    for c, goal in command_locs.items():
        # TODO: Show it works for coordinates, then train on N,S,E and show it works for west even if not trained?
        # TODO: Generate more data from simulator, can still be "offline"
        # Make sure we handle the boundaries by staying in for one frame then resetting

        task_str = f"{prompt} navigate to the {c}"
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
            #- 0.01 * goal_vel_reward(dataset, np.zeros_like(goal)) 
            + goal_pos_done(dataset, goal, 0.1)
            - 2 * boundary_reward(dataset, ARENA_BOUNDS_E, ARENA_BOUNDS_N).squeeze(-1)
        )

    def done_fn(dataset, goal):
        return (
            boundary_done(dataset, ARENA_BOUNDS_E, ARENA_BOUNDS_N).squeeze(-1)
            | (
                goal_pos_done(dataset, goal, 0.1)
                #& goal_vel_done(dataset, np.zeros_like(goal), 0.1)
            )
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

def make_global_navigation_tasks(num_tasks=1_000):
    task_strings = []
    task_embeddings = []
    goals = []
    eps = 0.75
    for i in range(num_tasks):
        x = random.uniform(ARENA_BOUNDS_E[0] + eps, ARENA_BOUNDS_E[1] - eps)
        y = random.uniform(ARENA_BOUNDS_N[0] + eps, ARENA_BOUNDS_N[1] - eps)

        command_strings = [
            f"navigate to ({x:0.2f}, {y:0.2f})",
        ]
        # TODO: Show it works for coordinates, then train on N,S,E and show it works for west even if not trained?
        # TODO: Generate more data from simulator, can still be "offline"
        # Make sure we handle the boundaries by staying in for one frame then resetting
        command_locs = {
            "west edge": np.array([ARENA_BOUNDS_E[0] + eps, 0]),
            "south west corner": np.array([ARENA_BOUNDS_E[0] + eps, ARENA_BOUNDS_N[0] + eps]),
            "west edge": ARENA_BOUNDS_E[1] - eps,
            "south edge": ARENA_BOUNDS_N[0] + eps,
            "north edge": ARENA_BOUNDS_N[1] - eps,
        }
        idx = random.randint(0, len(command_strings) - 1)
        task_str = f"{prompt} {command_strings[idx]}"
        task_strings.append(task_str)
        goals.append((x, y))
    task_embeddings = llm.encode(task_strings)

    def reward_fn(dataset, goal):
        # Dataset shape: [B, 2]
        # Goal shape: [G, 2]
        # Output shape: [B, G]
        # TODO: Add boundary reward
        return (
            relative_goal_pos_reward(dataset, goal) 
            #- 0.01 * goal_vel_reward(dataset, np.zeros_like(goal)) 
            + goal_pos_done(dataset, goal, 0.1)
            - 2 * boundary_reward(dataset, ARENA_BOUNDS_E, ARENA_BOUNDS_N).squeeze(-1)
        )

    def done_fn(dataset, goal):
        return (
            boundary_done(dataset, ARENA_BOUNDS_E, ARENA_BOUNDS_N).squeeze(-1)
            | (
                goal_pos_done(dataset, goal, 0.1)
                #& goal_vel_done(dataset, np.zeros_like(goal), 0.1)
            )
        )
    
    reward_kwargs = {"goal": np.array(goals)}
    assert len(task_strings) == len(task_embeddings) == reward_kwargs["goal"].shape[0] == num_tasks

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
    


if __name__ == '__main__':
    datasets = [
        "data/rand-1hz-sticky-1/robomaster_1/rl_statesactions_tuple/rl_tuples.csv",
        "data/rand-1hz-sticky-2/robomaster_1/rl_statesactions_tuple/rl_tuples.csv",
        "data/rand-1hz-sticky-3/robomaster_1/rl_statesactions_tuple/rl_tuples.csv"
    ]
    data, data_size = dataset_from_csv(datasets)
    tasks = make_global_navigation_tasks(2_000)
    l_tasks = make_language_navigation_tasks()
    data_with_rewards = add_rewards_to_dataset(data, tasks)
    ldata_with_rewards = add_rewards_to_dataset(data, l_tasks)
    all_data_with_rewards = merge_reward_datasets([data_with_rewards, ldata_with_rewards])
    with h5py.File("dataset.h5", "w") as file:
        file.create_dataset("state", data=all_data_with_rewards["state"])
        file.create_dataset("action", data=all_data_with_rewards["action"])
        file.create_dataset("next_state", data=all_data_with_rewards["next_state"])
        file.create_dataset("next_reward", data=all_data_with_rewards["next_reward"])
        file.create_dataset("next_done", data=all_data_with_rewards["next_done"])
        file.create_dataset("task_embedding", data=all_data_with_rewards["task_embedding"])
        file.create_dataset("task_string", data=tasks["task_string"] + l_tasks["task_string"], dtype=h5py.special_dtype(vlen=str))
#file.create_dataset("task_strings", )

#ALL_TASKS = make_global_navigation_tasks()
#breakpoint()
