"""
visualization.py - Visualization Utilities for Warehouse Robot Q-Learning

This module provides helper functions for visualizing experiment results and
learned policies in the WarehouseGridWorld environment:

1. `smooth(data, window=50)`:
   - Smooths a data sequence using a moving average.

2. `plot_comparison(data1, data2, ...)`:
   - Plots two sequences (e.g., rewards or success counts) for comparison.
   - Supports optional smoothing, labels, titles, and saving to a file.

3. `visualize_policy(env, Q, title="")`:
   - Visualizes a learned policy in Pygame.
   - Supports Q-tables as either a NumPy array (2D tabular) or a dict keyed
     by state tuples.
   - Animates the agent's behavior step-by-step according to the greedy policy.

Initial code template for this visualization was generated with ChatGPT.
"""

import numpy as np
import matplotlib.pyplot as plt
import pygame
import time


def smooth(data, window=50):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode='valid')


def plot_comparison(data1, data2, save_path=None,
                    label1="Method 1", label2="Method 2",
                    title=None, xlabel=None, ylabel=None,
                    smooth_fn=None):
    """
    Plot two sequences for comparison.
    
    Parameters
    ----------
    data1 : list or np.array
        First data sequence (e.g., rewards or success counts)
    data2 : list or np.array
        Second data sequence
    save_path : str, optional
        File path to save the plot
    label1, label2 : str
        Legend labels for the two sequences
    title : str, optional
        Plot title
    xlabel, ylabel : str, optional
        Axis labels
    smooth_fn : callable, optional
        Function to smooth data before plotting
    """
    plt.figure(figsize=(8, 5))

    if smooth_fn:
        y1 = smooth_fn(data1)
        y2 = smooth_fn(data2)
    else:
        y1 = data1
        y2 = data2

    plt.plot(y1, label=label1)
    plt.plot(y2, label=label2)
    plt.legend()

    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)

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