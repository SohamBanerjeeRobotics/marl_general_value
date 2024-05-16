from tasks import make_language_navigation_tasks
import pickle
import numpy as np


tasks = make_language_navigation_tasks()
eval_tasks = make_language_navigation_tasks(eval=True)

train_mapping = {}
for i in range(len(tasks['task_embedding'])):
    if tasks['reward_kwargs']['goal'][i].tobytes() not in train_mapping:
        train_mapping[tasks['reward_kwargs']['goal'][i].tobytes()] = []
    train_mapping[tasks['reward_kwargs']['goal'][i].tobytes()].append({
        "task_string": tasks['task_string'][i], 
        "embedding": tasks['task_embedding'][i], 
        "goal": tasks['reward_kwargs']['goal'][i]
    })

eval_mapping = {}
for i in range(len(eval_tasks['task_embedding'])):
    if eval_tasks['reward_kwargs']['goal'][i].tobytes() not in eval_mapping:
        eval_mapping[eval_tasks['reward_kwargs']['goal'][i].tobytes()] = []
    eval_mapping[eval_tasks['reward_kwargs']['goal'][i].tobytes()].append({
        "task_string": eval_tasks['task_string'][i], 
        "embedding": eval_tasks['task_embedding'][i], 
        "goal": eval_tasks['reward_kwargs']['goal'][i]
    })

# Keys are useless, drop them
train_mapping = list(train_mapping.values())
eval_mapping = list(eval_mapping.values())

with open("robomaster_control_train.pkl", 'wb') as f:
    pickle.dump(train_mapping, f)
with open("robomaster_control_eval.pkl", 'wb') as f:
    pickle.dump(eval_mapping, f)

# # Load and verify correctness
# with open("robomaster_control_train.pkl", 'rb') as f:
#     loaded_train = pickle.load(f)
# with open("robomaster_control_eval.pkl", 'rb') as f:
#     loaded_eval = pickle.load(f)

# for k, v in train_mapping.items():
#     assert k in loaded_train, f"{k} not in loaded_train"
#     for kk, vv in v.items():
#         assert np.allclose(vv, loaded_train[k][kk])


# for k, v in eval_mapping.items():
#     assert k in loaded_eval, f"{k} not in loaded_eval"
#     for kk, vv in v.items():
#         assert np.allclose(vv, loaded_eval[k][kk])
