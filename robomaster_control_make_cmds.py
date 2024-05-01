from tasks import make_language_navigation_tasks
import pickle
import numpy as np


tasks = make_language_navigation_tasks()
eval_tasks = make_language_navigation_tasks(eval=True)

strings = tasks['task_string'] + eval_tasks['task_string'][:1]
embeds = list(np.concatenate([tasks['task_embedding'], eval_tasks['task_embedding'][:1]], axis=0))
goals = list(tasks['reward_kwargs']['goal']) + list(eval_tasks['reward_kwargs']['goal'][:1])

mapping = {}
for i in range(len(embeds)):
    mapping[strings[i]] = {"embedding": embeds[i], "goal": goals[i]}

with open("robomaster_control.pkl", 'wb') as f:
    pickle.dump(mapping, f)

# Load and verify correctness
with open("robomaster_control.pkl", 'rb') as f:
    loaded = pickle.load(f)

for k, v in mapping.items():
    assert k in loaded
    for kk, vv in v.items():
        assert np.allclose(vv, loaded[k][kk])


