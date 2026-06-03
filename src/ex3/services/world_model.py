"""
World Model Module.

Implements the LSTM environment model (Transition Model).
Predicts s_{t+1} given s_t and a_t.
"""
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ex3.shared.config import cfg
from ex3.shared.constants import N_ACTION_CLUSTERS


class LSTMWorldModel(nn.Module):
    """LSTM model to simulate the environment transition."""

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = cfg.lstm_hidden_size):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # We concatenate state and action embeddings
        self.lstm = nn.LSTM(input_size=state_dim + action_dim, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, state_dim)

    def forward(self, state: torch.Tensor, action: torch.Tensor, hidden: tuple = None):
        """
        Forward pass.
        state shape: (batch, seq_len, state_dim)
        action shape: (batch, seq_len) - discrete actions
        """
        batch_size, seq_len = action.shape

        # One-hot encode action
        action_one_hot = torch.nn.functional.one_hot(action, num_classes=self.action_dim).float()

        # Concatenate state and action
        x = torch.cat([state, action_one_hot], dim=-1)

        # LSTM pass
        out, hidden = self.lstm(x, hidden)

        # Predict next state
        next_state_pred = self.fc(out)

        return next_state_pred, hidden

def train_world_model(states_df: pd.DataFrame, actions_df: pd.DataFrame) -> LSTMWorldModel:
    """Trains the LSTM world model on the trajectory."""
    # Convert to tensors
    states = torch.tensor(states_df.values, dtype=torch.float32)
    actions = torch.tensor(actions_df["action"].values, dtype=torch.long)

    state_dim = states.shape[1]

    # We create sequences of length cfg.sequence_length
    # For simplicity, we just use the whole trajectory as one batch if it's small,
    # or chunk it. Here we chunk it.
    seq_len = cfg.sequence_length

    if len(states) <= seq_len:
        # Fallback if trajectory is too short
        seq_len = len(states) - 1

    x_states, x_actions, y_states = [], [], []
    for i in range(len(states) - seq_len):
        x_states.append(states[i:i+seq_len])
        x_actions.append(actions[i:i+seq_len])
        y_states.append(states[i+1:i+seq_len+1])

    x_states = torch.stack(x_states)
    x_actions = torch.stack(x_actions)
    y_states = torch.stack(y_states)

    split_idx = max(1, int(len(x_states) * 0.8))
    train_dataset = TensorDataset(x_states[:split_idx], x_actions[:split_idx], y_states[:split_idx])
    val_dataset = TensorDataset(x_states[split_idx:], x_actions[split_idx:], y_states[split_idx:])

    train_loader = DataLoader(train_dataset, batch_size=cfg.lstm_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.lstm_batch_size, shuffle=False)

    model = LSTMWorldModel(state_dim=state_dim, action_dim=N_ACTION_CLUSTERS)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lstm_lr)

    train_losses, val_losses = [], []
    print("Training LSTM World Model...")
    for epoch in range(cfg.lstm_epochs):
        model.train()
        epoch_loss = 0.0
        for b_states, b_actions, b_next_states in train_loader:
            optimizer.zero_grad()
            preds, _ = model(b_states, b_actions)
            loss = criterion(preds, b_next_states)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for b_states, b_actions, b_next_states in val_loader:
                preds, _ = model(b_states, b_actions)
                loss = criterion(preds, b_next_states)
                val_loss += loss.item()

        val_losses.append(val_loss / len(val_loader) if len(val_loader) > 0 else val_loss)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{cfg.lstm_epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

    # Plot loss curves
    from pathlib import Path

    import matplotlib.pyplot as plt
    Path("results").mkdir(exist_ok=True)
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    if len(val_loader) > 0:
        plt.plot(val_losses, label="Val Loss")
    plt.title("LSTM World Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/lstm_loss_curve.png")
    plt.close()

    return model
