import numpy as np
from src.environment.warehouse_gridworld_domain_random import WarehouseGridWorld, ACTIONS
from collections import defaultdict

# Hyperparameters
ALPHA = 0.1
GAMMA = 0.99
EPISODES = 2000
MAX_STEPS = 120  # matches environment default

# Exploration bonus constant
K = 5.0


def choose_action_with_bonus(Q, N, state, k):
    best_action = None
    best_value = float("-inf")

    for a in ACTIONS:
        q_val = Q[state][a]
        n_val = N[state][a]
        bonus = k / np.sqrt(n_val + 1)

        value = q_val + bonus

        if value > best_value:
            best_value = value
            best_action = a

    return best_action


def train(env, k):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    N = defaultdict(lambda: {a: 0 for a in ACTIONS})

    rewards_per_episode = []

    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0

        for step in range(MAX_STEPS):
            action = choose_action_with_bonus(Q, N, state, k)

            # Track visits
            N[state][action] += 1

            result = env.step(action)

            next_state = result.state
            reward = result.reward
            done = result.done

            # Q-learning update
            max_next_q = max(Q[next_state].values())
            Q[state][action] += ALPHA * (
                reward + GAMMA * max_next_q - Q[state][action]
            )

            state = next_state
            total_reward += reward

            if done:
                break

        rewards_per_episode.append(total_reward)

        if (episode + 1) % 500 == 0:
            avg = np.mean(rewards_per_episode[-100:])
            print(f"Episode {episode+1}/{EPISODES} | Avg reward (last 100): {avg:.2f}")

    return Q, rewards_per_episode


def evaluate(env, Q):
    state = env.reset()
    total_reward = 0

    print("\nEVALUATION RUN (greedy policy)")
    print("------------------------------------------------------------")

    for step in range(MAX_STEPS):
        action = max(Q[state], key=Q[state].get)

        result = env.step(action)

        next_state = result.state
        reward = result.reward
        done = result.done

        print(f"Step {step+1:3d}: {action:6s} -> reward={reward:6.1f}")

        total_reward += reward
        state = next_state

        if done:
            break

    print("------------------------------------------------------------")
    print(f"Total reward: {total_reward}")
    print("============================================================\n")


def run_single_seed(seed, k):
    print(f"\n=== Running seed {seed}, k={k} ===")

    env = WarehouseGridWorld(seed=seed)

    print(f"Layout: Pickup={env.pickup_pos}, Packing={env.packing_pos}, Dock={env.dock_pos}")

    Q, rewards = train(env, k)

    evaluate(env, Q)

    return rewards


def main():
    print("==============================================")
    print(" Q-Learning with Exploration Bonus")
    print("==============================================")

    seeds = [99, 42, 7, 123, 256]
    k_values = [1.0, 5.0]

    for k in k_values:
        print(f"\n########## Testing k = {k} ##########")

        for seed in seeds:
            run_single_seed(seed, k)


if __name__ == "__main__":
    main()