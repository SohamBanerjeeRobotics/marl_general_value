# main_static_assignment.py - USAGE EXAMPLE

import jax.numpy as jnp
import numpy as np
import imageio
from evaluate_policy_ma import evaluate_ma_policy_with_assignment

if __name__ == "__main__":
    print("=" * 70)
    print("MULTI-AGENT POLICY WITH STATIC TASK ASSIGNMENT")
    print("=" * 70)
    
    config = {
        "seed": 0,
        "num_agents": 5,
        "obs_size": 4,
        "act_size": 9,
        "task_size": 768,
        "q_config": {
            "mlp_size": 1024,
            "head_size": 1024,
            "ensemble_size": 2,
            "dropout": 0.0,
            "ensemble_reduce": "min",
        },
    }
    
    # ========== RUN WITH HUNGARIAN ALGORITHM ==========
    print("\n[Configuration]")
    print("  Algorithm: Hungarian (optimal assignment)")
    print("  Use all tasks: False (select random subset)")
    print("  Timesteps: 150")
    print("  Agents: 5")
    
    data, goals, frames, rewards, dones, assignments, task_info = evaluate_ma_policy_with_assignment(
        env_kwargs={
            "scale": 800,
            "num_agents": 5,
            "headless": True
        },
        model_path="models/morlmarl/meanq/0/epoch-2500.eqx",  # YOUR MODEL PATH
        config=config,
        timesteps=150,
        seed=42,
        use_hungarian=True,     # ✅ HUNGARIAN ALGORITHM
        use_all_tasks=False,    # Random subset of tasks
        num_agents_override=5   # 5 agents
    )
    
    # ========== SAVE VIDEO ==========
    frames_np = np.array(frames, dtype=np.uint8)
    output_path = "static_assignment_hungarian.mp4"
    print(f"\n💾 Saving video to {output_path}...")
    with imageio.get_writer(output_path, fps=10) as writer:
        for frame in frames_np:
            writer.append_data(frame)
    print(f"✓ Video saved!")
    
    # ========== PRINT SUMMARY ==========
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nAlgorithm: {task_info['algorithm']}")
    print(f"Total assignment cost: {task_info['total_cost']:.3f}")
    print(f"\nAssignments:")
    for i, task_str in enumerate(task_info['task_strings']):
        print(f"  Agent {i}: {task_str}")
    print(f"\nPerformance:")
    print(f"  Mean reward: {rewards.mean():.3f}")
    print(f"  Total reward: {rewards.sum():.3f}")
    print(f"  Final distance to goals: {jnp.linalg.norm(data['next_state'][-1, :, :2] - goals[-1], axis=-1).mean():.3f}")
    
    
    # ========== COMPARE WITH GREEDY ==========
    print("\n" + "=" * 70)
    print("COMPARING WITH GREEDY NEAREST-NEIGHBOR")
    print("=" * 70)
    
    data_greedy, goals_greedy, frames_greedy, rewards_greedy, dones_greedy, assignments_greedy, task_info_greedy = evaluate_ma_policy_with_assignment(
        env_kwargs={"scale": 800, "num_agents": 5, "headless": True},
        model_path="models/morlmarl/meanq/0/epoch-2500.eqx",
        config=config,
        timesteps=150,
        seed=42,
        use_hungarian=False,    # ✅ GREEDY ALGORITHM
        use_all_tasks=False,
        num_agents_override=5
    )
    
    frames_greedy_np = np.array(frames_greedy, dtype=np.uint8)
    output_path_greedy = "static_assignment_greedy.mp4"
    print(f"\n💾 Saving video to {output_path_greedy}...")
    with imageio.get_writer(output_path_greedy, fps=10) as writer:
        for frame in frames_greedy_np:
            writer.append_data(frame)
    print(f"✓ Video saved!")
    
    # ========== COMPARISON ==========
    print("\n" + "=" * 70)
    print("ALGORITHM COMPARISON")
    print("=" * 70)
    print(f"\n{'Metric':<30} {'Hungarian':<20} {'Greedy':<20}")
    print("-" * 70)
    print(f"{'Assignment cost':<30} {task_info['total_cost']:<20.3f} {task_info_greedy['total_cost']:<20.3f}")
    print(f"{'Mean reward':<30} {rewards.mean():<20.3f} {rewards_greedy.mean():<20.3f}")
    print(f"{'Total reward':<30} {rewards.sum():<20.3f} {rewards_greedy.sum():<20.3f}")
    print(f"{'Final dist to goal':<30} {jnp.linalg.norm(data['next_state'][-1, :, :2] - goals[-1], axis=-1).mean():<20.3f} {jnp.linalg.norm(data_greedy['next_state'][-1, :, :2] - goals_greedy[-1], axis=-1).mean():<20.3f}")
