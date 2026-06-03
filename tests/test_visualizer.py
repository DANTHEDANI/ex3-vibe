"""
Test Visualizer.
"""
import os
from pathlib import Path

from ex3.services.visualizer import plot_learning_curves


def test_plot_learning_curves(tmp_path):
    # Mock output dir by patching Path or changing CWD? No, let's just let it run.
    # It creates results/learning_curves.png

    reinforce_rewards = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    a2c_rewards = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]

    plot_learning_curves(reinforce_rewards, a2c_rewards)

    output_file = Path("results/learning_curves.png")
    assert output_file.exists()

    # Cleanup
    os.remove(output_file)
