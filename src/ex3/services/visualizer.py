"""
Visualizer Module.

Generates plots for learning curves and training metrics.
"""
import matplotlib

matplotlib.use('Agg')
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_learning_curves(reinforce_rewards: list[float], a2c_rewards: list[float], save_path: str = "results/learning_curves.png") -> None:
    """Plots and saves the learning curves comparing REINFORCE and A2C."""
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    plt.plot(reinforce_rewards, label="REINFORCE", color="blue", alpha=0.7)
    plt.plot(a2c_rewards, label="A2C", color="orange", alpha=0.7)

    # Calculate moving averages for smoother curves
    window = 10
    if len(reinforce_rewards) >= window:
        r_ma = [sum(reinforce_rewards[i:i+window])/window for i in range(len(reinforce_rewards)-window)]
        a_ma = [sum(a2c_rewards[i:i+window])/window for i in range(len(a2c_rewards)-window)]

        plt.plot(range(window, len(reinforce_rewards)), r_ma, color="darkblue", linewidth=2, label="REINFORCE (MA)")
        plt.plot(range(window, len(a2c_rewards)), a_ma, color="darkorange", linewidth=2, label="A2C (MA)")

    plt.title("Learning Curves: REINFORCE vs A2C")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)

    plt.savefig("results/learning_curves.png")
    plt.close()
    print("Plot saved to results/learning_curves.png")

    # Diagnostic Graph (Variance over epochs)
    # We calculate a rolling standard deviation to show instability
    if len(reinforce_rewards) >= window:
        plt.figure(figsize=(10, 6))
        plt.plot(pd.Series(reinforce_rewards).rolling(window).std(), label="REINFORCE Reward Std Dev", color="blue", alpha=0.7)
        plt.plot(pd.Series(a2c_rewards).rolling(window).std(), label="A2C Reward Std Dev", color="orange", alpha=0.7)
        plt.title("Diagnostic Graph: Reward Variance (Instability)")
        plt.xlabel("Episode")
        plt.ylabel("Rolling Standard Deviation")
        plt.legend()
        plt.grid(True)
        plt.savefig("results/diagnostic_variance.png")
        plt.close()
        print("Diagnostic plot saved to results/diagnostic_variance.png")
