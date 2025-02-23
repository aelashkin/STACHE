import os
import yaml
from datetime import datetime
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
import torch
from enum import Enum


class ModelType(Enum):
    A2C = "A2C"
    PPO = "PPO"

def save_model(model, experiment_dir):
    """
    Save the model as model.zip inside the experiment directory.
    """
    model_path = os.path.join(experiment_dir, "model.zip")
    model.save(model_path)
    print(f"Model saved at: {model_path}")
    return model_path


def save_config(env_config, model_config, experiment_dir):
    """
    Save environment and model configurations in config.yaml inside the experiment directory.
    """
    config_path = os.path.join(experiment_dir, "config.yaml")
    config_data = {
        "env_config": env_config,
        "model_config": model_config,
    }
    with open(config_path, "w") as file:
        yaml.dump(config_data, file)
    print(f"Configuration saved at: {config_path}")
    return config_path


def save_training_log(training_log, experiment_dir):
    """
    Save training logs into training.log inside the experiment directory.
    """
    log_path = os.path.join(experiment_dir, "training.log")
    with open(log_path, "w") as file:
        file.write(training_log)
    print(f"Training log saved at: {log_path}")
    return log_path


def save_experiment(model, env_config, model_config, training_log, experiments_base_dir="data/experiments/models"):
    """
    Create an experiment folder and save the model, configuration, and training log.
    
    The experiment folder is named as:
    {env_name}_{model_type}_model_{timestamp}

    Files created:
        - model.zip : the saved model.
        - config.yaml : merged environment and model configurations.
        - training.log : a summary log of the training and evaluation.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_name = env_config.get("env_name", "unknown_env")
    model_type = model_config.get("model_type", "unknown_model")
    experiment_folder_name = f"{env_name}_{model_type}_model_{timestamp}"
    experiment_dir = os.path.join(experiments_base_dir, experiment_folder_name)
    os.makedirs(experiment_dir, exist_ok=True)

    model_path = save_model(model, experiment_dir)
    config_path = save_config(env_config, model_config, experiment_dir)
    log_path = save_training_log(training_log, experiment_dir)

    return {
        "experiment_dir": experiment_dir,
        "model_path": model_path,
        "config_path": config_path,
        "log_path": log_path,
    }


def load_experiment(experiment_dir):
    """
    Load an experiment from the given folder. It will read the configuration and load the model.

    Expects:
      - {experiment_dir}/config.yaml
      - {experiment_dir}/model.zip

    Returns:
      A tuple (model, config_data) where:
          model      : the loaded model.
          config_data: the dictionary with 'env_config' and 'model_config'.
    """
    # Load configuration
    config_path = os.path.join(experiment_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as file:
        config_data = yaml.safe_load(file)

    # Load model
    model_path = os.path.join(experiment_dir, "model.zip")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model_type = config_data.get("model_config", {}).get("model_type")
    model_map = {
        "PPO": PPO,
        "A2C": A2C,
    }
    if model_type not in model_map:
        raise ValueError(f"Unsupported model type: {model_type}")

    model = model_map[model_type].load(model_path)
    print(f"Loaded model from: {model_path}")
    return model, config_data

def evaluate_agent(model, env, n_eval_episodes=50):
    """
    Evaluate the trained agent and return mean and standard deviation of rewards.
    """
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, deterministic=True)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")
    return mean_reward, std_reward


def load_config(config_path):
    """
    Load a configuration file and return its contents as a dictionary.
    
    Parameters:
        config_path (str): Path to the configuration file.
        
    Returns:
        dict: Parsed configuration data.
        
    Raises:
        FileNotFoundError: If the config file does not exist.
        PermissionError: If the config file is not readable.
        yaml.YAMLError: If the config file contains invalid YAML.
    """
    # Check if the file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    
    # Check if the file is readable
    if not os.access(config_path, os.R_OK):
        raise PermissionError(f"Configuration file is not readable: {config_path}")
    
    # Attempt to load the YAML file
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            if config is None:
                raise ValueError("Configuration file is empty or has invalid content.")
            return config
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML configuration file at {config_path}: {e}")


def get_device(config_device=None):
    """
    Determine the device to use for training.

    Parameters:
        config_device (str or None): Desired device ('cpu', 'cuda', or 'mps') from the configuration.
                                     If None, the device is chosen automatically based on availability.

    Returns:
        torch.device: The device to be used for training.
    
    Raises:
        ValueError: If the specified device is invalid or unavailable.
    """
    # Automatically determine the best available device if not specified
    if config_device is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    # Validate user-specified device
    config_device = config_device.lower()
    if config_device == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            raise ValueError("CUDA is not available on this system.")
    elif config_device == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            raise ValueError("MPS (Metal Performance Shaders) is not available on this system.")
    elif config_device == "cpu":
        return torch.device("cpu")
    else:
        raise ValueError(f"Unsupported device specified: {config_device}. Choose from 'cpu', 'cuda', or 'mps'.")