"""
Test SDK pipeline.
"""
from unittest.mock import patch

from ex3.sdk.ex3_sdk import Ex3SDK


@patch("ex3.sdk.ex3_sdk.download_dataset")
@patch("ex3.sdk.ex3_sdk.extract_trajectory")
@patch("ex3.sdk.ex3_sdk.plot_learning_curves")
def test_run_pipeline(mock_plot, mock_extract, mock_download, mock_states_df, mock_actions_df):
    mock_extract.return_value = (mock_states_df, mock_actions_df)

    sdk = Ex3SDK()

    # Temporarily lower epochs for fast test
    from ex3.shared.config import cfg
    cfg.lstm_epochs = 1
    cfg.rl_epochs = 1
    cfg.episode_length = 2
    cfg.sequence_length = 2

    sdk.run_pipeline()

    assert sdk.world_model is not None
    assert sdk.reinforce_policy is not None
    assert sdk.a2c_policy is not None
    assert mock_plot.called
