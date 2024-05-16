"""Make a single agent dataset"""

from dataset import dataset_from_csv
from tasks import add_rewards_to_dataset, make_language_navigation_tasks
import h5py


if __name__ == '__main__':
    datasets = [
        "data/robomaster_collect_1714734360.csv",
        "data/robomaster_collect_1714735377.csv",
        "data/robomaster_collect_1714736377.csv",
        "data/robomaster_collect_1714737295.csv",
        "data/robomaster_collect_1715178400.csv",
        "data/robomaster_collect_1715179339.csv",
    ]
    data, data_size = dataset_from_csv(datasets)
    #tasks = make_global_navigation_tasks(2_000)
    l_tasks = make_language_navigation_tasks()
    #data_with_rewards = add_rewards_to_dataset(data, tasks)
    all_data_with_rewards = add_rewards_to_dataset(data, l_tasks)
    # Subsample 
    data_amount = 1.0
    all_data_with_rewards = {k: v if k == 'task_embedding' else v[: int(data_amount * v.shape[0])] for k, v in all_data_with_rewards.items()}
    #all_data_with_rewards = merge_reward_datasets([data_with_rewards, ldata_with_rewards])
    with h5py.File("dataset-25.h5", "w") as file:
        file.create_dataset("state", data=all_data_with_rewards["state"])
        file.create_dataset("action", data=all_data_with_rewards["action"])
        file.create_dataset("next_state", data=all_data_with_rewards["next_state"])
        file.create_dataset("next_reward", data=all_data_with_rewards["next_reward"])
        file.create_dataset("next_done", data=all_data_with_rewards["next_done"])
        file.create_dataset("task_embedding", data=all_data_with_rewards["task_embedding"])
        file.create_dataset("task_string", data=l_tasks["task_string"], dtype=h5py.special_dtype(vlen=str))

#  Generate embeddings for evaluation
import robomaster_control_make_cmds