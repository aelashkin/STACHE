import os
from datetime import datetime
from stable_baselines3.common.evaluation import evaluate_policy
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
        file.write("Environment Configuration:\n")
        for key, val in logs["environment_config"].items():
            file.write(f"  {key}: {val}\n")
        file.write("\n---\n\n")

        file.write("Model Configuration:\n")
        for key, val in logs["model_config"].items():
            file.write(f"  {key}: {val}\n")
        file.write("\n---\n\n")

        file.write("Evaluation Results:\n")
        for key, val in logs["evaluation_results"].items():
            file.write(f"  {key}: {val}\n")

    print(f"Logs saved at: {log_path}")
    return log_path

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


def load_model(model_name, logs_dir="data/logs/", models_dir="data/models/"):
    import os
    from stable_baselines3 import PPO, A2C

    # Build absolute path for the model
    if not model_name.startswith(models_dir):
        model_path = os.path.join(models_dir, model_name)
    else:
        model_path = model_name

    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Derive log file name from model base name
    base_name = os.path.splitext(os.path.basename(model_path))[0]
    log_file_name = f"logs_{base_name}.txt"
    log_path = os.path.join(logs_dir, log_file_name)

    # Check if log file exists
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")

    model_type = None
    env_config = {}
    in_env_section = False

    # Parse log file for environment configuration and model type
    with open(log_path, "r") as lf:
        for line in lf:
            line = line.rstrip()
            if "Environment Configuration:" in line:
                in_env_section = True
                continue
            if "---" in line and in_env_section:
                in_env_section = False

            if in_env_section and ":" in line:
                key, value = line.split(":", 1)
                env_config[key.strip()] = value.strip()

            if line.startswith("model_type:"):
                model_type = line.split(":", 1)[1].strip()

    # Check for missing env_name
    if "env_name" not in env_config:
        raise ValueError("Missing 'env_name' in the logs.")

    # Confirm model_type was found
    if not model_type:
        raise ValueError("Missing 'model_type' in the logs.")

    # Map model type to actual classes
    model_map = {
        "PPO": PPO,
        "A2C": A2C
    }
    if model_type not in model_map:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Load and return the model plus env_config
    loaded_model = model_map[model_type].load(model_path)
    return loaded_model, env_config