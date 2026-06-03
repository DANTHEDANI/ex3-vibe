"""
Test Data Pipeline and Feature Engineering.
"""
from unittest.mock import MagicMock, patch

import pandas as pd

from ex3.services.data_pipeline import download_dataset
from ex3.services.feature_eng import extract_trajectory
from ex3.shared.config import cfg


@patch("subprocess.run")
@patch("zipfile.ZipFile")
@patch("os.remove")
def test_download_dataset_success(mock_remove, mock_zip, mock_subprocess, tmp_path):
    # Mock data dir
    cfg.data_dir = str(tmp_path)

    # We pretend the files don't exist initially
    # But after extraction, they should be simulated or we just test the flow
    mock_zip_instance = MagicMock()
    mock_zip.return_value.__enter__.return_value = mock_zip_instance

    # Create the mock zip file so it tries to extract
    zip_path = tmp_path / f"{cfg.kaggle_dataset.split('/')[-1]}.zip"
    zip_path.touch()

    download_dataset()

    assert mock_subprocess.called
    assert mock_zip.called
    assert mock_zip_instance.extractall.called
    assert mock_remove.called

@patch("ex3.services.data_pipeline.Path.exists")
def test_download_dataset_already_exists(mock_exists):
    # Return true for summary_path and detailed_path
    mock_exists.side_effect = [True, True, True]

    download_dataset()
    # It should return early without doing subprocess

@patch("pandas.read_csv")
def test_extract_trajectory(mock_read_csv, tmp_path):
    cfg.data_dir = str(tmp_path)
    # Touch the paths so it thinks they exist
    (tmp_path / "program_summary.csv").touch()
    (tmp_path / "programs_detailed_boostcamp_kaggle.csv").touch()

    # Mock dataframes
    summary_df = pd.DataFrame([
        {"title": "P1", "equipment": "Full Gym", "program_length": 8, "time_per_workout": "45 - 60 minutes"},
        {"title": "P2", "equipment": "Dumbbells", "program_length": 4, "time_per_workout": "30 mins"},
    ])
    detailed_df = pd.DataFrame([
        {"title": "P1", "week": 1, "day": 1, "sets": 3, "reps": 10, "exercise_name": "Bench Press"},
        {"title": "P1", "week": 1, "day": 2, "sets": 4, "reps": 8, "exercise_name": "Squat"},
    ])

    mock_read_csv.side_effect = [summary_df, detailed_df]

    states, actions = extract_trajectory()

    assert len(states) == 7
    assert len(actions) == 7
    assert "rolling_load" in states.columns
    assert "action" in actions.columns
