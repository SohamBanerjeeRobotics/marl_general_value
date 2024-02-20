from angle_emb import AnglE
import numpy as np

prompt = "You are a holonomic wheeled robot in a multirobot system."

# TODO: Should we embed agent ID in the prompt?
task0 = "Navigate to the blue object."
task1 = "There is a blue object somewhere nearby. Find it."
task2 = "Enter a flocking formation with the other robots."
task3 = "You are agent 3. Swap positions with agent 5."
task4 = "You are agent 5. Swap positions with agent 3."
task5 = "You are agent 1. Swap positions with agent 4."

tasks = [task0, task1, task2, task3, task4, task5]

for i, task in enumerate(tasks):
    angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()
    #vec = angle.encode('You are a robot in a multirobot system.', to_numpy=True)
    vec = angle.encode(f"{prompt} {task}", to_numpy=True)
    with open(f"task_embeddings/{i}.txt", "w+") as f:
        f.write(f"{prompt} {task}")
    np.savetxt(f"task_embeddings/{i}.np.txt", vec)