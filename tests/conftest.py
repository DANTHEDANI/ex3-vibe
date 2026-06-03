"""
Fixtures for tests.
"""
import pandas as pd
import pytest
import torch


@pytest.fixture
def mock_states_df():
    # 10 columns matching the new feature_eng output
    return pd.DataFrame([
        {"rolling_load": 100, "session_duration_t": 60, "week_index_t": 1, "day_in_cycle_t": 1, "exercise_count": 3, "chest": 0.5, "back": 0.5, "legs": 0, "core": 0, "arms": 0},
        {"rolling_load": 150, "session_duration_t": 90, "week_index_t": 1, "day_in_cycle_t": 2, "exercise_count": 4, "chest": 0, "back": 0, "legs": 1.0, "core": 0, "arms": 0},
        {"rolling_load": 200, "session_duration_t": 120, "week_index_t": 1, "day_in_cycle_t": 3, "exercise_count": 5, "chest": 0, "back": 0, "legs": 0, "core": 1.0, "arms": 0},
        {"rolling_load": 0, "session_duration_t": 0, "week_index_t": 1, "day_in_cycle_t": 4, "exercise_count": 0, "chest": 0, "back": 0, "legs": 0, "core": 0, "arms": 0},
        {"rolling_load": 100, "session_duration_t": 60, "week_index_t": 1, "day_in_cycle_t": 5, "exercise_count": 3, "chest": 0.5, "back": 0.5, "legs": 0, "core": 0, "arms": 0},
        {"rolling_load": 150, "session_duration_t": 90, "week_index_t": 1, "day_in_cycle_t": 6, "exercise_count": 4, "chest": 0, "back": 0, "legs": 1.0, "core": 0, "arms": 0},
        {"rolling_load": 200, "session_duration_t": 120, "week_index_t": 1, "day_in_cycle_t": 7, "exercise_count": 5, "chest": 0, "back": 0, "legs": 0, "core": 1.0, "arms": 0},
        {"rolling_load": 0, "session_duration_t": 0, "week_index_t": 2, "day_in_cycle_t": 1, "exercise_count": 0, "chest": 0, "back": 0, "legs": 0, "core": 0, "arms": 0},
        {"rolling_load": 100, "session_duration_t": 60, "week_index_t": 2, "day_in_cycle_t": 2, "exercise_count": 3, "chest": 0.5, "back": 0.5, "legs": 0, "core": 0, "arms": 0},
    ])

@pytest.fixture
def mock_actions_df():
    return pd.DataFrame([
        {"action": 1},
        {"action": 2},
        {"action": 3},
        {"action": 0},
        {"action": 1},
        {"action": 2},
        {"action": 3},
        {"action": 0},
        {"action": 1},
    ])

@pytest.fixture
def mock_init_state():
    return torch.tensor([100.0, 60.0, 1.0, 1.0, 3.0, 0.5, 0.5, 0.0, 0.0, 0.0], dtype=torch.float32)
