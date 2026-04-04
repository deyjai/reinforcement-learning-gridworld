# -*- coding: utf-8 -*-
"""
main.py - Warehouse Robot Q-Learning Experiments

This script orchestrates experiments for a warehouse robot navigation task
using Q-Learning. It performs the following:

1. Runs standard Q-Learning with ε-greedy exploration.
2. Runs Q-Learning enhanced with an exploration bonus.
3. Computes average rewards, cumulative task success, and approximate
   convergence episodes for each method.
4. Visualizes policies, rewards, and success metrics.
5. Performs parameter sweeps over alpha, gamma, epsilon decay, and
   exploration bonus to study their effect on performance.
6. Conducts robustness checks across multiple random seeds to evaluate
   stability of the learned policies.

Results, plots, and logs are saved under the `results/` directory.

Note: The initial code template for main.py was generated with ChatGPT.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.environment.warehouse_gridworld_domain_random import WarehouseGridWorld
from src.algorithms.q_learning_epsilon import QLearningAgent, train as train_q_learning_epsilon
from src.algorithms.q_learning_bonus import train as train_q_learning_bonus
from src.visualization import visualize_policy

# -----------------------------
# Setup
# -----------------------------
def ensure_dirs():
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)

# -----------------------------
# Approximate convergence
# -----------------------------
def find_convergence_episode(data, window=50, tol=0.01):
    data = np.array(data)
    max_val = np.max(data)
    for i in range(window, len(data)-window):
        prev_avg = np.mean(data[i-window:i])
        next_avg = np.mean(data[i:i+window])
        if abs(next_avg - prev_avg) < tol * max_val:
            return i
    return len(data) - 1  # never converged

# -----------------------------
# Plot with convergence markers
# -----------------------------
def plot_with_convergence(data1, data2, conv1, conv2,
                          save_path, label1, label2, title, xlabel, ylabel):
    plt.figure()
    plt.plot(data1, label=label1)
    plt.plot(data2, label=label2)
    plt.axvline(conv1, color="blue", linestyle="--")
    plt.axvline(conv2, color="orange", linestyle="--")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()

# -----------------------------
# Bar plot helper
# -----------------------------
def plot_bar(values_dict, xlabel, ylabel, title, save_path):
    keys = list(values_dict.keys())
    values = [values_dict[k] for k in keys]
    plt.figure()
    plt.bar([str(k) for k in keys], values, color='skyblue')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(save_path)
    plt.show()
    plt.close()

# -----------------------------
# Parameter sweeps (average reward)
# -----------------------------
def sweep_epsilon_greedy_avg(env):
    alpha_default = 0.1
    gamma_default = 0.9
    epsilon_default = 0.99

    # --- Sweep alpha ---
    alpha_vals = [0.1, 0.3, 0.5]
    avg_rewards_alpha = {}
    for alpha in alpha_vals:
        agent = QLearningAgent(
            state_space_size=env.get_state_space_size(),
            action_space_size=env.get_action_space_size(),
            alpha=alpha, gamma=gamma_default, epsilon_decay=epsilon_default
        )
        rewards, _ = train_q_learning_epsilon(env, agent)
        avg_rewards_alpha[alpha] = np.mean(rewards)

    print("\nAverage Rewards vs Alpha:")
    print("Alpha | Average Reward")
    print("----------------------")
    for alpha, reward in avg_rewards_alpha.items():
        print(f"{alpha:.2f}  | {reward:.2f}")
    plot_bar(avg_rewards_alpha, "Alpha", "Average Reward", "Average Reward vs Alpha", "results/plots/alpha_bar.png")

    # --- Sweep gamma ---
    gamma_vals = [0.7, 0.9, 0.99]
    avg_rewards_gamma = {}
    for gamma in gamma_vals:
        agent = QLearningAgent(
            state_space_size=env.get_state_space_size(),
            action_space_size=env.get_action_space_size(),
            alpha=alpha_default, gamma=gamma, epsilon_decay=epsilon_default
        )
        rewards, _ = train_q_learning_epsilon(env, agent)
        avg_rewards_gamma[gamma] = np.mean(rewards)

    print("\nAverage Rewards vs Gamma:")
    print("Gamma | Average Reward")
    print("----------------------")
    for gamma, reward in avg_rewards_gamma.items():
        print(f"{gamma:.2f}  | {reward:.2f}")
    plot_bar(avg_rewards_gamma, "Gamma", "Average Reward", "Average Reward vs Gamma", "results/plots/gamma_bar.png")

    # --- Sweep epsilon decay ---
    epsilon_vals = [0.9, 0.95, 0.99]
    avg_rewards_epsilon = {}
    for eps in epsilon_vals:
        agent = QLearningAgent(
            state_space_size=env.get_state_space_size(),
            action_space_size=env.get_action_space_size(),
            alpha=alpha_default, gamma=gamma_default, epsilon_decay=eps
        )
        rewards, _ = train_q_learning_epsilon(env, agent)
        avg_rewards_epsilon[eps] = np.mean(rewards)

    print("\nAverage Rewards vs Epsilon Decay:")
    print("Epsilon Decay | Average Reward")
    print("-----------------------------")
    for eps, reward in avg_rewards_epsilon.items():
        print(f"{eps:.2f}         | {reward:.2f}")
    plot_bar(avg_rewards_epsilon, "Epsilon Decay", "Average Reward", "Average Reward vs Epsilon Decay", "results/plots/epsilon_bar.png")


def sweep_bonus_avg(env):
    k_vals = [0.1, 0.5, 1.0]  # exploration bonus values
    avg_rewards_k = {}
    for k in k_vals:
        _, rewards, _ = train_q_learning_bonus(env, k=k)
        avg_rewards_k[k] = np.mean(rewards)

    print("\nAverage Rewards vs Exploration Bonus (k):")
    print("k     | Average Reward")
    print("--------------------")
    for k, reward in avg_rewards_k.items():
        print(f"{k:.2f}  | {reward:.2f}")
    plot_bar(avg_rewards_k, "Exploration Bonus k", "Average Reward", "Average Reward vs Exploration Bonus", "results/plots/bonus_bar.png")

# -----------------------------
# Main
# -----------------------------
def main():
    ensure_dirs()

    # Choose a random seed for the main workflow
    seed = random.randint(0, 10000)
    np.random.seed(seed)
    random.seed(seed)

    k = 5.0  # bonus constant

    print("======================================")
    print(f" Running Experiments (seed={seed})")
    print("======================================")

    # ----- Q-Learning (ε-greedy) -----
    env1 = WarehouseGridWorld(seed=seed)
    agent1 = QLearningAgent(
        state_space_size=env1.get_state_space_size(),
        action_space_size=env1.get_action_space_size(),
    )
    rewards1, success1 = train_q_learning_epsilon(env1, agent1)

    # ----- Q-Learning + Bonus -----
    env2 = WarehouseGridWorld(seed=seed)
    Q2_dict, rewards2, success2 = train_q_learning_bonus(env2, k=k)

    # ----- Save raw data -----
    np.save("results/logs/rewards_q.npy", rewards1)
    np.save("results/logs/rewards_bonus.npy", rewards2)
    np.save("results/logs/success_q.npy", success1)
    np.save("results/logs/success_bonus.npy", success2)

    # ----- Compute approximate convergence -----
    conv_rewards_q = find_convergence_episode(rewards1)
    conv_rewards_bonus = find_convergence_episode(rewards2)

    print("\nApproximate convergence episodes (rewards):")
    print(f"  Q-Learning (ε-greedy): {conv_rewards_q}")
    print(f"  Q-Learning + Bonus  : {conv_rewards_bonus}")

    # ----- Compute overall metrics -----
    avg_reward_q = np.mean(rewards1)
    avg_reward_bonus = np.mean(rewards2)
    success_rate_q = np.mean(success1)
    success_rate_bonus = np.mean(success2)

    print("\nOverall metrics across all episodes:")
    print(f"  Q-Learning (ε-greedy): Avg reward = {avg_reward_q:.2f}, Success rate = {success_rate_q*100:.2f}%")
    print(f"  Q-Learning + Bonus  : Avg reward = {avg_reward_bonus:.2f}, Success rate = {success_rate_bonus*100:.2f}%")

    # ----- Plot rewards with convergence markers -----
    plot_with_convergence(
        rewards1, rewards2,
        conv_rewards_q, conv_rewards_bonus,
        save_path="results/plots/rewards_comparison.png",
        label1="Q-Learning (ε-greedy)",
        label2="Q-Learning + Bonus",
        title="Rewards Comparison",
        xlabel="Episode",
        ylabel="Reward"
    )

    # ----- Plot cumulative task success -----
    cumulative_success1 = np.cumsum(success1)
    cumulative_success2 = np.cumsum(success2)
    plt.figure()
    plt.plot(cumulative_success1, label="Q-Learning (ε-greedy)")
    plt.plot(cumulative_success2, label="Q-Learning + Bonus")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Successes")
    plt.title("Cumulative Task Success")
    plt.legend()
    plt.savefig("results/plots/success_comparison.png")
    plt.show()
    plt.close()

    # ----- Visualize policies -----
    visualize_policy(env1, agent1.q_table, title="Q-Learning")
    visualize_policy(env2, Q2_dict, title="Q-Learning + Bonus")

    print("\nRunning parameter sweeps (averaged rewards)...")

    # ----- Parameter sweeps (bar graphs + tables) -----
    env_q = WarehouseGridWorld(seed=seed)
    sweep_epsilon_greedy_avg(env_q)

    env_bonus = WarehouseGridWorld(seed=seed)
    sweep_bonus_avg(env_bonus)

    # -----------------------------
    # Robustness check (silent, at the end)
    # -----------------------------
    print("\nRunning robustness check across multiple seeds...")
    robustness_seeds = [42, 123, 999]  # seeds for robustness
    robustness_metrics_q = []
    robustness_metrics_bonus = []

    for s in robustness_seeds:
        np.random.seed(s)
        random.seed(s)

        # Q-Learning
        env_rq = WarehouseGridWorld(seed=s)
        agent_rq = QLearningAgent(
            state_space_size=env_rq.get_state_space_size(),
            action_space_size=env_rq.get_action_space_size(),
        )
        rewards_rq, success_rq = train_q_learning_epsilon(env_rq, agent_rq)
        robustness_metrics_q.append((np.mean(rewards_rq), np.mean(success_rq)))

        # Q-Learning + Bonus
        env_rb = WarehouseGridWorld(seed=s)
        _, rewards_rb, success_rb = train_q_learning_bonus(env_rb, k=k)
        robustness_metrics_bonus.append((np.mean(rewards_rb), np.mean(success_rb)))

    # Print summary
    print("\nRobustness Across Seeds:")
    print("Q-Learning (ε-greedy):")
    for idx, s in enumerate(robustness_seeds):
        r, sr = robustness_metrics_q[idx]
        print(f"  Seed {s}: Avg reward = {r:.2f}, Success rate = {sr*100:.2f}%")
    print("Q-Learning + Bonus:")
    for idx, s in enumerate(robustness_seeds):
        r, sr = robustness_metrics_bonus[idx]
        print(f"  Seed {s}: Avg reward = {r:.2f}, Success rate = {sr*100:.2f}%")

    print("\nAll tests completed. Bar graphs saved in results/plots.")


if __name__ == "__main__":
    main()