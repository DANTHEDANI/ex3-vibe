"""
SDK Module.

Single Entry Point for all logic as per Architecture Guidelines.
API Gatekeeper routing requests to internal services.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch

from ex3.services.a2c_agent import train_a2c
from ex3.services.data_pipeline import download_dataset
from ex3.services.feature_eng import extract_trajectory
from ex3.services.reinforce_agent import train_reinforce
from ex3.services.visualizer import plot_learning_curves
from ex3.services.world_model import train_world_model


class Ex3SDK:
    """The SDK API Gatekeeper for Ex3 Pipeline."""

    def __init__(self):
        self.world_model = None
        self.reinforce_policy = None
        self.a2c_policy = None

        self.states_df = None
        self.actions_df = None

    def run_pipeline(self):
        """Runs the entire pipeline end-to-end."""
        print("--- Step 1: Downloading Dataset ---")
        try:
            download_dataset()
        except Exception as e:
            print(f"Warning: Data download skipped or failed: {e}")

        print("\n--- Step 2: Feature Engineering ---")
        try:
            self.states_df, self.actions_df = extract_trajectory()
            print(f"Extracted {len(self.states_df)} days of trajectory.")
        except Exception as e:
            print(f"Feature engineering failed: {e}")
            return

        print("\n--- Step 3: Training World Model (LSTM) ---")
        self.world_model = train_world_model(self.states_df, self.actions_df)

        # Initial state for RL agents
        init_state = torch.tensor(self.states_df.iloc[0].values, dtype=torch.float32)

        print("\n--- Step 4: Training REINFORCE Agent ---")
        self.reinforce_policy, reinforce_rewards = train_reinforce(self.world_model, init_state)

        print("\n--- Step 5: Training A2C Agent ---")
        self.a2c_policy, a2c_rewards = train_a2c(self.world_model, init_state)

        print("\n--- Step 6: Generating Visualizations ---")
        plot_learning_curves(reinforce_rewards, a2c_rewards)

        print("\nPipeline Complete.")


if __name__ == "__main__":
    sdk = Ex3SDK()
    sdk.run_pipeline()
