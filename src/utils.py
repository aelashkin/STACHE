import os
from datetime import datetime
from stable_baselines3.common.evaluation import evaluate_policy
from tqdm import tqdm
import yaml
import torch
from enum import Enum


class ModelType(Enum):
    A2C = "A2C"
    PPO = "PPO"

def save_model(model, model_dir="data/models/", env_name=None, model_type=None, timestamp=None):
    """
    Save the model with a timestamped filename, including the environment name.
    """
    os.makedirs(model_dir, exist_ok=True)
    if not timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(model_dir, f"{env_name}_{model_type}_model_{timestamp}.zip")
    model.save(model_path)
    print(f"Model saved at: {model_path}")
    return model_path

def save_logs(logs, log_dir="data/logs/", env_name=None, model_type=None, timestamp=None):
    """
    Save logs into a timestamped file, with the same name as the model but prefixed with 'logs_'.
    """
    os.makedirs(log_dir, exist_ok=True)
    if not timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"model_type: {model_type}")
    log_path = os.path.join(log_dir, f"logs_{env_name}_{model_type}_model_{timestamp}.txt")
    with open(log_path, "w") as file:
        for key, value in logs.items():
            if key == "training_config":
                file.write("Training Configuration:\n")
                for config_key, config_value in value.items():
                    file.write(f"  {config_key}: {config_value}\n")
            else:
                file.write(f"{key}: {value}\n")
    print(f"Logs saved at: {log_path}")
    return log_path

def evaluate_agent(model, env, n_eval_episodes=50):
    """
    Evaluate the trained agent and return mean and standard deviation of rewards.
    """
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, deterministic=True)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")
    return mean_reward, std_reward

def tqdm_monitor_training(model, total_timesteps, progress_bar=True):
    """
    Train the model with a tqdm-style progress bar.
    """
    if progress_bar:
        with tqdm(total=total_timesteps, desc="Training Progress") as pbar:
            # Patch the `callback` to update tqdm
            def tqdm_callback(local, global_):
                pbar.update(local["n_steps"])
                return True
            
            # Train the model
            model.learn(total_timesteps=total_timesteps, callback=tqdm_callback)
    else:
        model.learn(total_timesteps=total_timesteps)


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
