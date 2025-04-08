import pytest
from unittest.mock import patch, MagicMock
import yaml
from stable_baselines3 import PPO, A2C

from utils.experiment_io import load_experiment


def test_load_experiment_missing_config(tmp_path):
    """
    Test that load_experiment raises FileNotFoundError if config.yaml does not exist.
    """
    # Do not create config.yaml
    with pytest.raises(FileNotFoundError):
        load_experiment(str(tmp_path))


def test_load_experiment_missing_model(tmp_path):
    """
    Test that load_experiment raises FileNotFoundError if model.zip does not exist even if config.yaml exists.
    """
    # Create a valid config.yaml
    config_data = {
        "env_config": {"env_name": "TestEnv"},
        "model_config": {"model_type": "PPO"}
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
    # Do not create model.zip
    with pytest.raises(FileNotFoundError):
        load_experiment(str(tmp_path))


@patch("stable_baselines3.PPO.load")
def test_load_experiment_success(mock_ppo_load, tmp_path):
    """
    Test load_experiment successfully loads the model when both config.yaml and model.zip exist.
    """
    # Create a dummy model.zip file
    model_path = tmp_path / "model.zip"
    model_path.touch()

    # Create a valid config.yaml
    config_data = {
        "env_config": {"env_name": "TestEnv"},
        "model_config": {"model_type": "PPO", "some_other_info": 123}
    }
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    dummy_loaded_model = MagicMock(spec=PPO)
    mock_ppo_load.return_value = dummy_loaded_model

    loaded_model, config = load_experiment(str(tmp_path))
    mock_ppo_load.assert_called_once_with(str(model_path))
    assert loaded_model == dummy_loaded_model
    assert config["env_config"]["env_name"] == "TestEnv"


@patch("stable_baselines3.A2C.load")
def test_load_experiment_success_a2c(mock_a2c_load, tmp_path):
    """
    Test that load_experiment successfully loads A2C when config.yaml specifies model_type 'A2C'.
    """
    # Create a dummy model.zip file
    model_path = tmp_path / "model.zip"
    model_path.touch()

    # Create a valid config.yaml
    config_data = {
        "env_config": {"env_name": "SomeOtherEnv"},
        "model_config": {"model_type": "A2C", "extra_info": 456}
    }
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    dummy_loaded_model = MagicMock(spec=A2C)
    mock_a2c_load.return_value = dummy_loaded_model

    loaded_model, config = load_experiment(str(tmp_path))
    mock_a2c_load.assert_called_once_with(str(model_path))
    assert loaded_model == dummy_loaded_model
    assert config["env_config"]["env_name"] == "SomeOtherEnv"


def test_load_experiment_missing_info(tmp_path):
    """
    Test that load_experiment raises ValueError if 'model_type' is missing in config.yaml.
    """
    # Create a dummy model.zip file
    (tmp_path / "model.zip").touch()

    # Create an invalid config.yaml missing 'model_type'
    config_data = {
        "env_config": {"env_name": "TestEnv"},
        "model_config": {}  # Missing model_type
    }
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    with pytest.raises(ValueError):
        load_experiment(str(tmp_path))


def test_load_experiment_unsupported_model_type(tmp_path):
    """
    Test that load_experiment raises ValueError if the model_type is unsupported.
    """
    # Create a dummy model.zip file
    (tmp_path / "model.zip").touch()

    # Create config.yaml with an unsupported model_type
    config_data = {
        "env_config": {"env_name": "UnsupportedEnv"},
        "model_config": {"model_type": "INVALID_MODEL"}
    }
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    with pytest.raises(ValueError):
        load_experiment(str(tmp_path))

# cd /Users/eam/Documents/GIT/STACHE
# pytest tests/test_utils.py