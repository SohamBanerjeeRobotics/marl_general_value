import numpy as np
import itertools

tasks = []
embs = []
for i in range(6):
    with open(f"task_embeddings/{i}.txt") as f:
        task = f.read()
    emb = np.loadtxt(f"task_embeddings/{i}.np.txt")
    tasks.append(task)
    embs.append(emb.reshape(-1))

#tasks = itertools.product(tasks, repeat=2)
#embs = itertools.product(embs, repeat=2)
tasks = itertools.combinations_with_replacement(tasks, r=2)
embs = itertools.combinations_with_replacement(embs, r=2)

for (t0, t1), (e0, e1) in zip(tasks, embs):
    cos_sim = np.dot(e0, e1)/(np.linalg.norm(e0) * np.linalg.norm(e1))
    print(f'"{t0}" -- "{t1}",  Latent Similarity: {cos_sim:0.2f}')


    
