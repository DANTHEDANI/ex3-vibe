"""
REINFORCE Agent Module.

Implements the REINFORCE algorithm (Policy Gradient).
"""

import torch
import torch.nn as nn
import torch.optim as optim

from ex3.services.world_model import LSTMWorldModel
from ex3.shared.config import cfg
from ex3.shared.constants import N_ACTION_CLUSTERS


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state: torch.Tensor):
        return self.fc(state)

def compute_reward(action: int, state: torch.Tensor) -> float:
    """
    Computes reward based on the action and guidelines.
    rt = gaint - lambda1*overload - lambda2*imbalance
    """
    # state is [rolling_load, duration, week, day, count, chest, back, legs, core, arms]
    rolling_load = state[0].item()
    muscle_dist = state[5:10]

    gain = float(action) * 2.0

    # Overload if rolling_load is too high
    overload = max(0.0, rolling_load + action*10.0 - 500.0) / 50.0

    # Imbalance if muscle distribution is heavily skewed
    imbalance = torch.var(muscle_dist).item() * 10.0

    return gain - cfg.lambda_1 * overload - cfg.lambda_2 * imbalance
def train_reinforce(env_model: LSTMWorldModel, init_state: torch.Tensor) -> tuple[PolicyNetwork, list[float]]:
    """Trains the REINFORCE agent using the LSTM as the environment."""
    state_dim = init_state.shape[-1]
    policy = PolicyNetwork(state_dim, N_ACTION_CLUSTERS)
    optimizer = optim.Adam(policy.parameters(), lr=cfg.actor_lr)

    rewards_history = []

    print("Training REINFORCE...")
    for epoch in range(cfg.rl_epochs):
        log_probs = []
        rewards = []

        current_state = init_state.clone()
        hidden = None

        # Play an episode
        for _t in range(cfg.episode_length):
            action_probs = policy(current_state)
            m = torch.distributions.Categorical(action_probs)
            action = m.sample()

            log_probs.append(m.log_prob(action))
            reward = compute_reward(action.item(), current_state)
            rewards.append(reward)

            # Step in environment (LSTM)
            # Reshape state and action for LSTM
            st_lstm = current_state.unsqueeze(0).unsqueeze(0)
            at_lstm = action.unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                next_state, hidden = env_model(st_lstm, at_lstm, hidden)

            current_state = next_state.squeeze(0).squeeze(0)

        # Calculate returns
        returns = []
        g_val = 0
        for r in reversed(rewards):
            g_val = r + cfg.gamma * g_val
            returns.insert(0, g_val)
        returns = torch.tensor(returns)

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        # Policy gradient update
        loss = 0
        for log_prob, g_t in zip(log_probs, returns, strict=False):
            loss -= log_prob * g_t

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_reward = sum(rewards)
        rewards_history.append(epoch_reward)

        if (epoch + 1) % 20 == 0:
            print(f"REINFORCE Epoch {epoch+1}/{cfg.rl_epochs}, Total Reward: {epoch_reward:.2f}")

    return policy, rewards_history
