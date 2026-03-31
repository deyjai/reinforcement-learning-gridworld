# -*- coding: utf-8 -*-
"""

Code generated with Claude Code Agent.

Q-Learning with ε-Greedy Exploration for Warehouse Robot Navigation

This module implements Q-Learning to train a robot to:
  1. Navigate from the charging dock to the pickup station
  2. Collect the package
  3. Deliver it to the packing station
  4. Return to the charging dock

State representation: (row, col, has_package, delivered)
Actions: "up", "down", "left", "right"

Usage:
    python q_learning_warehouse.py
"""

import numpy as np
from warehouse_gridworld_domain_random import WarehouseGridWorld, ACTIONS

# ──────────────────────────────────────────────
# Hyperparameters (change these to experiment)
# ──────────────────────────────────────────────
ALPHA = 0.1            # Learning rate
GAMMA = 0.99           # Discount factor
EPSILON_START = 1.0    # Initial exploration rate
EPSILON_MIN = 0.01     # Minimum exploration rate
EPSILON_DECAY = 0.995  # Multiplicative decay per episode
NUM_EPISODES = 2000    # Training episodes
MAX_STEPS = 120        # Max steps per episode (matches environment default)
DEFAULT_SEED = 99      # Default environment seed
TEST_SEEDS = [99, 42, 7, 123, 256]  # Seeds to test robustness


# ──────────────────────────────────────────────
# Q-Learning Agent
# ──────────────────────────────────────────────
class QLearningAgent:
    """Tabular Q-Learning agent with ε-greedy exploration."""

    def __init__(self, state_space_size: int, action_space_size: int,
                 alpha: float = ALPHA, gamma: float = GAMMA,
                 epsilon: float = EPSILON_START, epsilon_min: float = EPSILON_MIN,
                 epsilon_decay: float = EPSILON_DECAY):
        self.q_table = np.zeros((state_space_size, action_space_size))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng()

    def select_action(self, state_idx: int, greedy: bool = False) -> int:
        """ε-greedy action selection. If greedy=True, always exploit."""
        if not greedy and self.rng.random() < self.epsilon:
            return self.rng.integers(len(ACTIONS))
        return int(np.argmax(self.q_table[state_idx]))

    def update(self, s_idx: int, a_idx: int, reward: float,
               s_next_idx: int, done: bool) -> None:
        """Standard Q-Learning (off-policy TD(0)) update."""
        best_next = 0.0 if done else np.max(self.q_table[s_next_idx])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[s_idx, a_idx]
        self.q_table[s_idx, a_idx] += self.alpha * td_error

    def decay_epsilon(self) -> None:
        """Multiplicative epsilon decay, floored at epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ──────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────
def train(env: WarehouseGridWorld, agent: QLearningAgent,
          num_episodes: int = NUM_EPISODES) -> list:
    """
    Train the agent and return a list of cumulative rewards per episode.
    """
    rewards_per_episode = []

    for ep in range(num_episodes):
        state = env.reset()
        s_idx = env.state_to_index(state)
        total_reward = 0.0

        while not env.is_terminal():
            # Select action
            a_idx = agent.select_action(s_idx)
            action = ACTIONS[a_idx]

            # Take step
            result = env.step(action)
            s_next_idx = env.state_to_index(result.state)

            # Q-Learning update
            agent.update(s_idx, a_idx, result.reward, s_next_idx, result.done)

            s_idx = s_next_idx
            total_reward += result.reward

        agent.decay_epsilon()
        rewards_per_episode.append(total_reward)

        # Progress logging
        if (ep + 1) % 500 == 0:
            avg = np.mean(rewards_per_episode[-100:])
            print(f"  Episode {ep+1:>5}/{num_episodes}  |  "
                  f"ε = {agent.epsilon:.4f}  |  "
                  f"Avg reward (last 100) = {avg:.1f}")

    return rewards_per_episode


# ──────────────────────────────────────────────
# Evaluation (greedy policy rollout)
# ──────────────────────────────────────────────
def evaluate(env: WarehouseGridWorld, agent: QLearningAgent,
             verbose: bool = True) -> float:
    """
    Run one episode using the greedy (learned) policy.
    Returns total reward. Optionally prints each step.
    """
    state = env.reset()
    s_idx = env.state_to_index(state)
    total_reward = 0.0
    step = 0

    if verbose:
        print("\n" + "=" * 60)
        print("EVALUATION RUN (greedy policy)")
        print("=" * 60)
        print(f"  Pickup: {env.pickup_pos}  |  Packing: {env.packing_pos}  |  Dock: {env.dock_pos}")
        print(f"  Start state: {state}")
        print("-" * 60)

    while not env.is_terminal():
        a_idx = agent.select_action(s_idx, greedy=True)
        action = ACTIONS[a_idx]
        result = env.step(action)
        s_next_idx = env.state_to_index(result.state)
        total_reward += result.reward
        step += 1

        if verbose:
            event = result.info.get("event", "")
            marker = ""
            if event == "pickup":
                marker = "  *** PICKED UP PACKAGE ***"
            elif event == "delivery":
                marker = "  *** DELIVERED PACKAGE ***"
            elif event == "return_to_dock":
                marker = "  *** RETURNED TO DOCK — MISSION COMPLETE ***"
            elif event == "invalid_move":
                marker = "  (wall/boundary)"
            print(f"  Step {step:>3}: {action:<6} -> pos=({result.state[0]},{result.state[1]})  "
                  f"reward={result.reward:>6.1f}  "
                  f"has_pkg={bool(result.state[2])}  "
                  f"delivered={bool(result.state[3])}{marker}")

        s_idx = s_next_idx

    if verbose:
        print("-" * 60)
        print(f"  Total reward: {total_reward:.1f}  |  Steps: {step}")
        print("=" * 60)

    return total_reward



# ──────────────────────────────────────────────
# Extract & display learned policy
# ──────────────────────────────────────────────
def print_policy_grid(env: WarehouseGridWorld, agent: QLearningAgent,
                      has_package: int, delivered: int) -> None:
    """
    Print the greedy policy as directional arrows on the grid.
    Shows policy for a specific (has_package, delivered) phase.
    """
    arrow = {"up": "↑", "down": "↓", "left": "←", "right": "→"}
    phase_label = f"has_package={bool(has_package)}, delivered={bool(delivered)}"
    print(f"\n  Learned Policy Grid  [{phase_label}]")
    print("  " + "----" * env.size)

    for r in range(env.size):
        row_str = "  "
        for c in range(env.size):
            cell = int(env.grid[r, c])
            if cell == 1:  # shelf
                row_str += " ## "
                continue
            state = (r, c, has_package, delivered)
            s_idx = env.state_to_index(state)
            best_a = int(np.argmax(agent.q_table[s_idx]))
            row_str += f" {arrow[ACTIONS[best_a]]}  "
        print(row_str)
    print("  " + "----" * env.size)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║  Q-Learning with ε-Greedy — Warehouse Grid World     ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    # ── 1. Train on default seed ──
    print(f"[1] Training on default seed = {DEFAULT_SEED} ...")
    env = WarehouseGridWorld(seed=DEFAULT_SEED, max_steps=MAX_STEPS)
    print(f"    Layout: Pickup={env.pickup_pos}, Packing={env.packing_pos}, Dock={env.dock_pos}\n")

    agent = QLearningAgent(
        state_space_size=env.get_state_space_size(),
        action_space_size=env.get_action_space_size(),
    )
    rewards = train(env, agent, NUM_EPISODES)

    # ── 2. Print learned policy for each task phase ──
    print("\n[2] Learned policies (greedy):")
    print_policy_grid(env, agent, has_package=0, delivered=0)  # Phase 1: go to pickup
    print_policy_grid(env, agent, has_package=1, delivered=0)  # Phase 2: go to packing
    print_policy_grid(env, agent, has_package=1, delivered=1)  # Phase 3: return to dock

    # ── 4. Evaluation run ──
    print("\n[3] Evaluation run on default seed:")
    eval_reward = evaluate(env, agent, verbose=True)

    # ── 5. Robustness: test across multiple seeds ──
    print("\n[4] Robustness test across multiple seeds ...")
    for seed in TEST_SEEDS:
        print(f"\n  --- Seed {seed} ---")
        env_s = WarehouseGridWorld(seed=seed, max_steps=MAX_STEPS)
        print(f"    Pickup={env_s.pickup_pos}, Packing={env_s.packing_pos}, Dock={env_s.dock_pos}")

        agent_s = QLearningAgent(
            state_space_size=env_s.get_state_space_size(),
            action_space_size=env_s.get_action_space_size(),
        )
        rews = train(env_s, agent_s, NUM_EPISODES)

        # Quick greedy evaluation
        eval_r = evaluate(env_s, agent_s, verbose=False)

	# We can use this to determine the best policy
        print(f"    Greedy evaluation reward: {eval_r:.1f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
