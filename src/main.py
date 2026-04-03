import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from src.environment.warehouse_gridworld_domain_random import WarehouseGridWorld
from src.algorithms.q_learning_epsilon import QLearningAgent, train as train_q_learning_epsilon
from src.algorithms.q_learning_bonus import train as train_q_learning_bonus
from src.visualization import plot_comparison, visualize_policy


def ensure_dirs():
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)


def main():
    ensure_dirs()

    seed = 42
    k = 5.0

    print("======================================")
    print(" Running Experiments")
    print("======================================")

    # ----- Q-Learning (ε-greedy) -----
    env1 = WarehouseGridWorld(seed=seed)
    agent1 = QLearningAgent(
        state_space_size=env1.get_state_space_size(),
        action_space_size=env1.get_action_space_size(),
    )
    rewards1 = train_q_learning_epsilon(env1, agent1)

    # ----- Q-Learning + Bonus -----
    env2 = WarehouseGridWorld(seed=seed)
    Q2_dict, rewards2 = train_q_learning_bonus(env2, k=k)
    # Q2_dict is already a dict keyed by full state tuples

    # ----- Save raw data -----
    np.save("results/logs/rewards_q.npy", rewards1)
    np.save("results/logs/rewards_bonus.npy", rewards2)

    # ----- Plot comparison -----
    plot_comparison(
        rewards1,
        rewards2,
        save_path="results/plots/rewards_comparison.png"
    )

    # ----- Visualize policies -----
    # For ε-greedy Q-learning, pass the Q-table + environment
    visualize_policy(env1, agent1.q_table, title="Q-Learning")

    # For bonus Q-learning, pass the dict directly
    visualize_policy(env2, Q2_dict, title="Q-Learning + Bonus")

    print("\nDone.")


if __name__ == "__main__":
    main()