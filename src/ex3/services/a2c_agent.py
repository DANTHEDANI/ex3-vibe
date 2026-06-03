"""
A2C Agent Module.

Implements the Advantage Actor-Critic algorithm.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from ex3.services.reinforce_agent import compute_reward
from ex3.services.world_model import LSTMWorldModel
from ex3.shared.config import cfg
from ex3.shared.constants import N_ACTION_CLUSTERS


class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state: torch.Tensor):
        return self.actor(state), self.critic(state)

def train_a2c(env_model: LSTMWorldModel, init_state: torch.Tensor) -> tuple[ActorCriticNetwork, list[float]]:
    """Trains the A2C agent using the LSTM as the environment."""
    state_dim = init_state.shape[-1]
    model = ActorCriticNetwork(state_dim, N_ACTION_CLUSTERS)

    actor_optim = optim.Adam(model.actor.parameters(), lr=cfg.actor_lr)
    critic_optim = optim.Adam(model.critic.parameters(), lr=cfg.critic_lr)

    rewards_history = []

    print("Training A2C...")
    for epoch in range(cfg.rl_epochs):
        log_probs = []
        values = []
        rewards = []

        current_state = init_state.clone()
        hidden = None

        # Play an episode
        for _t in range(cfg.episode_length):
            action_probs, value = model(current_state)
            m = torch.distributions.Categorical(action_probs)
            action = m.sample()

            log_probs.append(m.log_prob(action))
            values.append(value)

            reward = compute_reward(action.item(), current_state)
            rewards.append(reward)

            st_lstm = current_state.unsqueeze(0).unsqueeze(0)
            at_lstm = action.unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                next_state, hidden = env_model(st_lstm, at_lstm, hidden)

            current_state = next_state.squeeze(0).squeeze(0)

        # Add next state value for bootstrapping
        _, next_value = model(current_state)
        values.append(next_value.detach())

        actor_loss = 0
        critic_loss = 0

        for i in range(len(rewards)):
            v_t = values[i]
            v_t_next = values[i+1].item() if i < len(rewards) - 1 else next_value.item()

            # TD Advantage: a_t = r_t + gamma * V(s_{t+1}) - V(s_t)
            a_t = rewards[i] + cfg.gamma * v_t_next - v_t.item()

            actor_loss -= log_probs[i] * a_t

            # Critic loss: MSE between TD target and V(s_t)
            td_target = torch.tensor(rewards[i] + cfg.gamma * v_t_next)
            critic_loss += (td_target - v_t) ** 2

        # Update
        actor_optim.zero_grad()
        critic_optim.zero_grad()

        actor_loss.backward()
        critic_loss.backward()

        actor_optim.step()
        critic_optim.step()

        epoch_reward = sum(rewards)
        rewards_history.append(epoch_reward)

        if (epoch + 1) % 20 == 0:
            print(f"A2C Epoch {epoch+1}/{cfg.rl_epochs}, Total Reward: {epoch_reward:.2f}")

    return model, rewards_history
