# -*- coding: utf-8 -*-
"""
q_learning_epsilon.py - Q-Learning Agent with ε-Greedy Exploration

This module implements a tabular Q-Learning agent for a warehouse robot 
navigation task. It provides functionality to:

1. Train the agent in the WarehouseGridWorld environment.
   - Tracks rewards per episode.
   - Tracks task success per episode (1 if the agent completes pickup, delivery, 
     and returns to dock; 0 otherwise).
   - Supports ε-greedy exploration with epsilon decay over episodes.

2. Evaluate a trained agent in greedy mode, printing detailed step-by-step
   actions, rewards, and task events (pickup, delivery, return to dock).

3. Configure hyperparameters such as learning rate (alpha), discount factor (gamma),
   initial epsilon, epsilon decay, and maximum episodes/steps.

4. Perform robustness testing using multiple random seeds.

Initial code template for this Q-Learning workflow was generated with ChatGPT.
"""

import numpy as np
from src.environment.warehouse_gridworld_domain_random import WarehouseGridWorld, ACTIONS

# ──────────────────────────────────────────────
# Hyperparameters
# ──────────────────────────────────────────────
ALPHA = 0.1
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
NUM_EPISODES = 2000
MAX_STEPS = 120
DEFAULT_SEED = 99
TEST_SEEDS = [99, 42, 7, 123, 256]


class QLearningAgent:
    """Tabular Q-Learning agent with ε-greedy exploration."""

    def __init__(self, state_space_size, action_space_size,
                 alpha=ALPHA, gamma=GAMMA,
                 epsilon=EPSILON_START, epsilon_min=EPSILON_MIN,
                 epsilon_decay=EPSILON_DECAY):
        self.q_table = np.zeros((state_space_size, action_space_size))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng()

    def select_action(self, state_idx, greedy=False):
        if not greedy and self.rng.random() < self.epsilon:
            return self.rng.integers(len(ACTIONS))
        return int(np.argmax(self.q_table[state_idx]))

    def update(self, s_idx, a_idx, reward, s_next_idx, done):
        best_next = 0.0 if done else np.max(self.q_table[s_next_idx])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[s_idx, a_idx]
        self.q_table[s_idx, a_idx] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train(env: WarehouseGridWorld, agent: QLearningAgent, num_episodes=NUM_EPISODES):
    """
    Train the agent and return rewards + task success (1 if completed task, else 0).
    """
    rewards_per_episode = []
    success_per_episode = []

    for ep in range(num_episodes):
        state = env.reset()
        s_idx = env.state_to_index(state)
        total_reward = 0.0
        task_completed = 0

        while not env.is_terminal():
            a_idx = agent.select_action(s_idx)
            action = ACTIONS[a_idx]

            result = env.step(action)
            s_next_idx = env.state_to_index(result.state)

            agent.update(s_idx, a_idx, result.reward, s_next_idx, result.done)
            s_idx = s_next_idx
            total_reward += result.reward

            # Mark task success if agent returned to dock after delivery
            if result.info.get("event") == "return_to_dock":
                task_completed = 1

        agent.decay_epsilon()
        rewards_per_episode.append(total_reward)
        success_per_episode.append(task_completed)

        mean_reward = np.mean(rewards_per_episode[:ep+1])
        if (ep + 1) % 500 == 0:
            print(f"Episode {ep+1}/{num_episodes} | Average reward = {mean_reward:.1f}")

    return rewards_per_episode, success_per_episode


def evaluate(env: WarehouseGridWorld, agent: QLearningAgent, verbose=True):
    state = env.reset()
    s_idx = env.state_to_index(state)
    total_reward = 0.0
    step = 0
    success = 0

    while not env.is_terminal():
        a_idx = agent.select_action(s_idx, greedy=True)
        action = ACTIONS[a_idx]
        result = env.step(action)
        s_idx = env.state_to_index(result.state)
        total_reward += result.reward
        step += 1

        if result.info.get("event") == "return_to_dock":
            success = 1

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
            print(f"Step {step:>3}: {action:<6} -> pos=({result.state[0]},{result.state[1]})  "
                  f"reward={result.reward:>6.1f}  "
                  f"has_pkg={bool(result.state[2])}  "
                  f"delivered={bool(result.state[3])}{marker}")

    if verbose:
        print(f"Total reward: {total_reward:.1f} | Task success: {success}")
    return total_reward, success