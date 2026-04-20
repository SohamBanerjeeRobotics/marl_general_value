from tasks import make_language_navigation_tasks

# Get all available tasks
tasks = make_language_navigation_tasks(eval=False)

print("Available tasks:")
print("=" * 70)
for i, task_str in enumerate(tasks['task_string']):
    goal = tasks['reward_kwargs']['goal'][i]
    print(f"{i:2d}. {task_str}")
    print(f"    Goal: N={goal[0]:6.2f}, E={goal[1]:6.2f}")
    print()
