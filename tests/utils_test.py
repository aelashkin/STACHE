import pytest
import os
from unittest.mock import patch, mock_open, MagicMock
from src.utils import load_model
from stable_baselines3 import PPO, A2C

def test_load_model_file_not_found(tmp_path):
    """
    Test that load_model raises FileNotFoundError if the model file does not exist.
    """
    dummy_model_file = "non_existent_model.zip"
    with pytest.raises(FileNotFoundError):
        load_model(dummy_model_file, logs_dir=str(tmp_path), models_dir=str(tmp_path))

def test_load_model_log_not_found(tmp_path):
    """
    Test that load_model raises FileNotFoundError if the log file does not exist.
    """
    model_file_path = tmp_path / "test_model.zip"
    # Create an empty model file to pass the first check
    model_file_path.touch()

    with pytest.raises(FileNotFoundError):
        load_model(str(model_file_path), logs_dir=str(tmp_path), models_dir=str(tmp_path))

@patch("stable_baselines3.PPO.load")
def test_load_model_success(mock_ppo_load, tmp_path):
    """
    Test that load_model successfully loads the model when both model and log files exist,
    and extracts env_name and model_type from logs.
    """
    # Create a dummy model file
    model_file_path = tmp_path / "test_model.zip"
    model_file_path.touch()

    # Create a matching dummy log file
    log_file_path = tmp_path / f"logs_{model_file_path.stem}.txt"
    log_contents = """env_name: TestEnv
model_type: PPO
some_other_info: 123
"""
    with open(log_file_path, "w") as f:
        f.write(log_contents)

    # Mock the load call from stable_baselines3 to return a dummy PPO object
    dummy_loaded_model = MagicMock(spec=PPO)
    mock_ppo_load.return_value = dummy_loaded_model

    loaded_model, env_name = load_model(
        str(model_file_path),
        logs_dir=str(tmp_path),
        models_dir=str(tmp_path)
    )
    # Ensure load was called
    mock_ppo_load.assert_called_once_with(str(model_file_path))
    # Check values returned by load_model
    assert loaded_model == dummy_loaded_model
    assert env_name == "TestEnv"

@patch("stable_baselines3.A2C.load")
def test_load_model_success_a2c(mock_a2c_load, tmp_path):
    """
    Test that load_model successfully loads A2C when logs specify model_type: A2C.
    """
    model_file_path = tmp_path / "another_model.zip"
    model_file_path.touch()

    log_file_path = tmp_path / f"logs_{model_file_path.stem}.txt"
    log_contents = """env_name: SomeOtherEnv
model_type: A2C
extra_info: 456
"""
    with open(log_file_path, "w") as f:
        f.write(log_contents)

    dummy_loaded_model = MagicMock(spec=A2C)
    mock_a2c_load.return_value = dummy_loaded_model

    loaded_model, env_name = load_model(
        str(model_file_path),
        logs_dir=str(tmp_path),
        models_dir=str(tmp_path)
    )
    mock_a2c_load.assert_called_once_with(str(model_file_path))
    assert loaded_model == dummy_loaded_model
    assert env_name == "SomeOtherEnv"

def test_load_model_missing_info(tmp_path):
    """
    Test that load_model raises ValueError if the env_name or model_type is missing in the log file.
    """
    # Create a dummy model file
    model_file_path = tmp_path / "model_missing_info.zip"
    model_file_path.touch()

    # Create a matching dummy log file missing 'env_name' or 'model_type'
    log_file_path = tmp_path / f"logs_{model_file_path.stem}.txt"
    log_contents = "mean_reward: 0.123\nstd_reward: 0.456\n"
    with open(log_file_path, "w") as f:
        f.write(log_contents)

    with pytest.raises(ValueError):
        load_model(str(model_file_path), logs_dir=str(tmp_path), models_dir=str(tmp_path))

def test_load_model_unsupported_model_type(tmp_path):
    """
    Test that load_model raises ValueError if the model_type is unsupported.
    """
    model_file_path = tmp_path / "unsupported_model.zip"
    model_file_path.touch()

    log_file_path = tmp_path / f"logs_{model_file_path.stem}.txt"
    log_contents = """env_name: UnsupportedEnv
model_type: INVALID_MODEL
"""
    with open(log_file_path, "w") as f:
        f.write(log_contents)

    with pytest.raises(ValueError):
        load_model(str(model_file_path), logs_dir=str(tmp_path), models_dir=str(tmp_path))

cd /Users/eam/Documents/GIT/STACHE
pytest tests/utils_test.py