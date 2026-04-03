import numpy as np
import matplotlib.pyplot as plt
import pygame
import time


def smooth(data, window=50):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode='valid')


def plot_comparison(rewards_q, rewards_bonus, save_path=None):
    plt.figure()

    rewards_q_smooth = smooth(rewards_q)
    rewards_bonus_smooth = smooth(rewards_bonus)

    plt.plot(rewards_q_smooth, label="Q-Learning")
    plt.plot(rewards_bonus_smooth, label="Q-Learning + Bonus")

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Learning Curve Comparison")
    plt.legend()

    if save_path:
        plt.savefig(save_path)

    plt.show()
    plt.close()


def visualize_policy(env, Q, title=""):
    """
    Visualize a learned policy with Pygame.
    Q can be either:
      - a NumPy array (2D tabular Q-table)
      - a dict keyed by state tuples
    """
    from src.environment.warehouse_gridworld_domain_random import setup_pygame, draw_grid, ACTIONS

    screen, clock, font, small_font = setup_pygame()
    state = env.reset()
    running = True

    print(f"\n=== {title} ===")

    while running:
        pygame.time.delay(200)

        # Determine action from Q
        if isinstance(Q, np.ndarray):
            s_idx = env.state_to_index(state)
            best_a_idx = int(np.argmax(Q[s_idx]))
            action = ACTIONS[best_a_idx]
        else:  # dict
            if state not in Q:
                best_a_idx = 0  # default action if state missing
                action = ACTIONS[best_a_idx]
            else:
                # Q[state] is a dict {action_name: value}
                action = max(Q[state], key=Q[state].get)

        # Step in environment
        result = env.step(action)
        state = result.state

        # Handle quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Draw grid and update display
        draw_grid(env, screen, font, small_font)
        pygame.display.flip()
        clock.tick(5)

        if result.done:
            time.sleep(1)
            running = False

    pygame.quit()