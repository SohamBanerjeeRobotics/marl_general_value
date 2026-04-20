"""
nlp_sim.py
==========
Simple, clean NLP simulator matching the author's exact pipeline.

Flow:
  1. User types command
  2. GTE encodes it → find closest training task
  3. Run exactly N steps using the RL policy
  4. Stop. Show final positions. Wait for next command.

Run:
    python nlp_sim.py --model models/best_model.eqx --config experiments/soft-5.0.yaml
"""

import argparse
import os
import threading
import queue

import yaml
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import pygame
from sentence_transformers import SentenceTransformer

from evaluate_policy import MARLEnv
from modules import GeneralMAQNetwork
from tasks import make_language_navigation_tasks
from constants import ROBOT_DIAMETER

PROMPT = "Agent,"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  required=True)
    parser.add_argument("--config", default="experiments/soft-5.0.yaml")
    parser.add_argument("--steps",  type=int, default=60)
    parser.add_argument("--fps",    type=int, default=10)
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    NUM_AGENTS = config["num_agents"]
    TASK_SIZE  = config["task_size"]
    print(f"\n[config] num_agents={NUM_AGENTS}  mlp={config['q_config']['mlp_size']}")

    # Load Q-network
    key   = jax.random.PRNGKey(0)
    q_net = GeneralMAQNetwork(
        obs_size=config["obs_size"],
        task_size=TASK_SIZE,
        act_size=config["act_size"],
        config=config["q_config"],
        key=key,
    )
    q_net = eqx.tree_deserialise_leaves(args.model, q_net)
    q_net = eqx.nn.inference_mode(q_net)
    print(f"[OK] Q-network loaded.")

    # Load GTE — same as tasks.py
    print("[...] Loading GTE (thenlper/gte-base)...")
    gte = SentenceTransformer('thenlper/gte-base')
    print("[OK] GTE ready.")

    # Load all training tasks for nearest-neighbour lookup
    print("[...] Loading training tasks...")
    train_tasks = make_language_navigation_tasks(eval=False)
    eval_tasks  = make_language_navigation_tasks(eval=True)

    all_strings = train_tasks["task_string"] + eval_tasks["task_string"]
    all_embs    = np.concatenate([
        train_tasks["task_embedding"],
        eval_tasks["task_embedding"]
    ], axis=0)
    all_goals   = np.concatenate([
        train_tasks["reward_kwargs"]["goal"],
        eval_tasks["reward_kwargs"]["goal"]
    ], axis=0)
    all_embs_norm = all_embs / (np.linalg.norm(all_embs, axis=1, keepdims=True) + 1e-9)
    print(f"[OK] {len(all_strings)} tasks loaded.")

    # MARLEnv
    env = MARLEnv(num_agents=NUM_AGENTS, scale=500, headless=False)
    pygame.display.set_caption("NLP MARL Simulator")

    # Initial state
    key, rk     = jax.random.split(key)
    agent_state = env.reset(rk)

    ZERO_EMB  = np.zeros(TASK_SIZE, dtype=np.float32)
    ZERO_GOAL = np.zeros(2, dtype=np.float32)

    # Per-robot task state
    robot_embs   = [None] * NUM_AGENTS
    robot_goals  = [ZERO_GOAL.copy()] * NUM_AGENTS
    robot_labels = ["idle"] * NUM_AGENTS

    cmd_q = queue.Queue()
    clock = pygame.time.Clock()

    # ── Helper: find closest training task ───────────────────
    def find_closest_task(user_text):
        full          = f"{PROMPT} {user_text}"
        user_emb      = gte.encode([full])[0]
        user_emb_norm = user_emb / (np.linalg.norm(user_emb) + 1e-9)
        sims          = all_embs_norm @ user_emb_norm
        best_i        = int(np.argmax(sims))
        return (
            all_embs[best_i],
            all_goals[best_i],
            all_strings[best_i],
            float(sims[best_i]),
            full,
        )

    # ── Helper: parse robot specifier from tokens ─────────────
    def get_target_and_task(text):
        """Returns (target_robot_indices, task_text_string)."""
        tokens = text.strip().split()
        target    = list(range(NUM_AGENTS))  # default: all
        task_text = text.strip()

        for i, tok in enumerate(tokens):
            tl = tok.lower()

            if tl in ("all", "everyone"):
                target    = list(range(NUM_AGENTS))
                task_text = " ".join(tokens[i+1:]).strip()
                break

            if tl in ("robot", "r") and i + 1 < len(tokens):
                try:
                    rid = int(tokens[i+1]) - 1
                    if 0 <= rid < NUM_AGENTS:
                        target    = [rid]
                        task_text = " ".join(tokens[i+2:]).strip()
                        break
                except ValueError:
                    pass

            if tl.startswith("r") and len(tl) > 1:
                try:
                    rid = int(tl[1:]) - 1
                    if 0 <= rid < NUM_AGENTS:
                        target    = [rid]
                        task_text = " ".join(tokens[i+1:]).strip()
                        break
                except ValueError:
                    pass

            try:
                rid = int(tl) - 1
                if 0 <= rid < NUM_AGENTS:
                    target    = [rid]
                    task_text = " ".join(tokens[i+1:]).strip()
                    break
            except ValueError:
                pass

        return target, task_text or text.strip()

    # ── Helper: run exactly N steps then stop ─────────────────
    def run_steps(n_steps):
        """
        Run n_steps of the RL policy.
        ALL robots go through the GNN together every step.
        The GNN sees all robot states → handles collision avoidance.
        After n_steps, robots STOP and wait for next command.
        """
        nonlocal agent_state

        emb_matrix  = jnp.array(np.stack([
            robot_embs[i] if robot_embs[i] is not None else ZERO_EMB
            for i in range(NUM_AGENTS)
        ]))  # (N, 768) — each robot's own task embedding

        goal_matrix = jnp.array(np.stack(robot_goals))  # (N, 2)

        for step in range(n_steps):

            # ── Policy inference ──────────────────────────────
            # Exactly as author's rollout_policy scan_fn:
            #   action = q_function(agent_state, embeds, PRNGKey(0)).argmax(-1)
            # GNN processes ALL robots together — collision avoidance is implicit
            q_vals  = q_net(agent_state, emb_matrix, jax.random.PRNGKey(0))
            actions = jnp.argmax(q_vals, axis=-1).astype(jnp.int32)

            # Idle robots (no task) get action 0 = stop
            for i in range(NUM_AGENTS):
                if robot_embs[i] is None:
                    actions = actions.at[i].set(0)

            # ── Physics step ──────────────────────────────────
            agent_state = env.step(agent_state, actions)

            # ── Collision check ───────────────────────────────
            snp = np.array(agent_state)
            for a in range(NUM_AGENTS):
                for b in range(a + 1, NUM_AGENTS):
                    dist = float(np.linalg.norm(snp[a, :2] - snp[b, :2]))
                    if dist < ROBOT_DIAMETER:
                        print(f"  [COLLISION] step {step+1}: "
                              f"Robot {a+1} & Robot {b+1}  dist={dist:.3f}m")

            # ── Render ────────────────────────────────────────
            env.render(agent_state, goal_matrix)
            pygame.display.set_caption(
                f"NLP MARL  step {step+1}/{n_steps}")

            # Handle quit during simulation
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False

            clock.tick(args.fps)

        # Simulation done — print final positions
        snp = np.array(agent_state)
        print(f"\n  [result] After {n_steps} steps:")
        for i in range(NUM_AGENTS):
            d = np.linalg.norm(snp[i, :2] - robot_goals[i])
            print(f"    Robot {i+1}: "
                  f"({snp[i,0]:+.3f}N, {snp[i,1]:+.3f}E)  "
                  f"dist_to_goal={d:.3f}m  "
                  f"task='{robot_labels[i]}'")

        # Render final state and STOP — wait for next command
        env.render(agent_state, goal_matrix)
        pygame.display.set_caption("NLP MARL — done, waiting for command...")
        pygame.display.flip()
        return True

    # ── Input thread ──────────────────────────────────────────
    HELP = f"""
=== NLP MARL Simulator ({NUM_AGENTS} robots) ===
Type a command. Robots run {args.steps} steps then STOP and wait.

  robot 1 go to the north edge
  robot 2 move to the south west corner
  robot 3 reach the east side
  all navigate to the NE corner
  r1 head towards the west wall
  2 go south

  stop robot 1    clear robot 1 task
  stop all        clear all tasks
  reset           respawn robots randomly
  status          show current positions
  tasks           list all training tasks
  quit
"""

    def input_thread():
        print(HELP)
        while True:
            try:
                text = input(">>> ").strip()
            except EOFError:
                cmd_q.put("quit")
                break
            if text:
                cmd_q.put(text)

    threading.Thread(target=input_thread, daemon=True).start()

    # ── Initial render ────────────────────────────────────────
    env.render(agent_state, jnp.zeros((NUM_AGENTS, 2)))
    pygame.display.set_caption("NLP MARL — waiting for command...")
    pygame.display.flip()

    # ── Main loop ─────────────────────────────────────────────
    running = True
    while running:

        # Handle pygame quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

        # Block until we get a command (no background simulation)
        try:
            text = cmd_q.get(timeout=0.05)
        except queue.Empty:
            clock.tick(args.fps)
            continue

        lower = text.strip().lower()

        # ── Quit ──────────────────────────────────────────────
        if lower in ("quit", "exit", "q"):
            running = False

        # ── Reset ─────────────────────────────────────────────
        elif lower == "reset":
            key, rk     = jax.random.split(key)
            agent_state = env.reset(rk)
            robot_embs   = [None] * NUM_AGENTS
            robot_goals  = [ZERO_GOAL.copy()] * NUM_AGENTS
            robot_labels = ["idle"] * NUM_AGENTS
            env.render(agent_state, jnp.zeros((NUM_AGENTS, 2)))
            pygame.display.set_caption("NLP MARL — reset, waiting for command...")
            pygame.display.flip()
            print("  [reset] Robots respawned. Type a command.")

        # ── Stop all ──────────────────────────────────────────
        elif lower.startswith("stop"):
            rest = lower[4:].strip()
            if not rest or rest in ("all", "everyone"):
                robot_embs   = [None] * NUM_AGENTS
                robot_goals  = [ZERO_GOAL.copy()] * NUM_AGENTS
                robot_labels = ["idle"] * NUM_AGENTS
                print("  [stop] All robots cleared.")
            else:
                for w in rest.split():
                    try:
                        rid = int(w) - 1
                        if 0 <= rid < NUM_AGENTS:
                            robot_embs[rid]   = None
                            robot_goals[rid]  = ZERO_GOAL.copy()
                            robot_labels[rid] = "idle"
                            print(f"  [stop] Robot {rid+1} cleared.")
                    except ValueError:
                        pass

        # ── Status ────────────────────────────────────────────
        elif lower == "status":
            snp = np.array(agent_state)
            print(f"\n  Status:")
            for i in range(NUM_AGENTS):
                d     = np.linalg.norm(snp[i, :2] - robot_goals[i])
                state = "has task" if robot_embs[i] is not None else "idle"
                print(f"    Robot {i+1} [{state}]: "
                      f"({snp[i,0]:+.2f}N, {snp[i,1]:+.2f}E)  "
                      f"dist={d:.2f}m  '{robot_labels[i]}'")
            print()

        # ── Tasks list ────────────────────────────────────────
        elif lower in ("tasks", "list"):
            print(f"\n  All {len(all_strings)} training tasks:")
            for i, s in enumerate(all_strings):
                g = all_goals[i]
                print(f"  {i+1:3d}. {s:<55s}  goal=({g[0]:+.2f}N,{g[1]:+.2f}E)")
            print()

        # ── Help ──────────────────────────────────────────────
        elif lower in ("help", "?"):
            print(HELP)

        # ── NLP task command ──────────────────────────────────
        else:
            target, task_text = get_target_and_task(text)

            if not task_text:
                print("  ! No task text found.")
                continue

            # GTE encode + find closest training task
            emb, goal, matched, sim, gte_in = find_closest_task(task_text)

            print(f"\n  [you typed]  '{task_text}'")
            print(f"  [GTE input]  '{gte_in}'")
            print(f"  [matched]    '{matched}'  (sim={sim:.3f})")
            print(f"  [goal]       N={goal[0]:.2f}  E={goal[1]:.2f}")
            print(f"  [robots]     {[r+1 for r in target]}")

            # Assign task to target robots
            for i in target:
                robot_embs[i]   = emb
                robot_goals[i]  = goal.copy()
                robot_labels[i] = matched

            # Run exactly args.steps steps then STOP
            print(f"  [running]    {args.steps} steps...\n")
            ok = run_steps(args.steps)
            if not ok:
                running = False

            print("\n  Robots stopped. Type next command.")

    pygame.quit()
    print("Simulator closed.")


if __name__ == "__main__":
    main()
