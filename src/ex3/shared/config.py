"""
Configuration File.

Contains mathematical/physical constants, default parameter values,
and configurations as specified in the guidelines. Do not hardcode these in logic files.
"""
import os
from dataclasses import dataclass


@dataclass
class Config:
    # Dataset
    kaggle_dataset: str = "adnanelouardi/600k-fitness-exercise-and-workout-program-dataset"
    data_dir: str = "data"

    # LSTM World Model parameters
    sequence_length: int = 7
    lstm_hidden_size: int = 64
    lstm_epochs: int = 10
    lstm_lr: float = 0.001
    lstm_batch_size: int = 32

    # RL Agents parameters
    episode_length: int = 28
    rl_epochs: int = 100
    gamma: float = 0.99
    actor_lr: float = 0.001
    critic_lr: float = 0.005

    # Reward multipliers
    lambda_1: float = 0.5  # overload penalty multiplier
    lambda_2: float = 0.5  # imbalance penalty multiplier

    def get_kaggle_username(self) -> str:
        return os.environ.get("KAGGLE_USERNAME", "")

    def get_kaggle_key(self) -> str:
        return os.environ.get("KAGGLE_KEY", "")

cfg = Config()
