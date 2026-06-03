"""
Test models.
"""
import torch

from ex3.services.a2c_agent import ActorCriticNetwork, train_a2c
from ex3.services.reinforce_agent import PolicyNetwork, compute_reward, train_reinforce
from ex3.services.world_model import LSTMWorldModel, train_world_model
from ex3.shared.config import cfg


def test_world_model_forward():
    model = LSTMWorldModel(state_dim=10, action_dim=6, hidden_size=10)
    state = torch.randn(1, 5, 10)
    action = torch.randint(0, 6, (1, 5))

    next_state, hidden = model(state, action)
    assert next_state.shape == (1, 5, 10)

def test_train_world_model(mock_states_df, mock_actions_df):
    cfg.lstm_epochs = 1
    cfg.sequence_length = 2
    model = train_world_model(mock_states_df, mock_actions_df)
    assert isinstance(model, LSTMWorldModel)

def test_reinforce_agent(mock_init_state):
    cfg.rl_epochs = 1
    cfg.episode_length = 5
    model = LSTMWorldModel(state_dim=10, action_dim=6, hidden_size=10)

    policy, rewards = train_reinforce(model, mock_init_state)
    assert isinstance(policy, PolicyNetwork)
    assert len(rewards) == 1

def test_a2c_agent(mock_init_state):
    cfg.rl_epochs = 1
    cfg.episode_length = 5
    model = LSTMWorldModel(state_dim=10, action_dim=6, hidden_size=10)

    policy, rewards = train_a2c(model, mock_init_state)
    assert isinstance(policy, ActorCriticNetwork)
    assert len(rewards) == 1

def test_compute_reward():
    state = torch.zeros(10)
    r1 = compute_reward(0, state)
    r2 = compute_reward(5, state)
    assert r1 != r2
