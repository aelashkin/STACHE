import sys
import os

pythonpath = os.getenv("PYTHONPATH")
if (pythonpath and pythonpath not in sys.path):
    sys.path.append(pythonpath)

import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from src.wrappers import FactorizedSymbolicWrapper, PaddedObservationWrapper
from src.utils import save_model, save_logs, evaluate_agent, tqdm_monitor_training, load_config, get_device
from datetime import datetime
import yaml

import torch.nn as nn


def train_agent(env_config, model_config):
    """
    Train the agent and save the model and logs.
    """
    # Step 1: Determine the device
    device = get_device(model_config.get("device"))
    print(f"Using device: {device}")

    # Step 2: Initialize the environment
    env_name = env_config["env_name"]
    print(f"Initializing the environment: {env_name}")
    env = gym.make(env_name)
    env = FullyObsWrapper(env)
    env = FactorizedSymbolicWrapper(env)
    env = PaddedObservationWrapper(env, max_objects=env_config["max_objects"], max_walls=env_config["max_walls"])
    env = DummyVecEnv([lambda: env])

    # Step 3: Initialize the model based on the config
    model_type = model_config["model_type"]
    print(f"Setting up the {model_type} model...")

    if model_type == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            n_steps=config["n_steps"],
            batch_size=config["batch_size"],
            ent_coef=config["ent_coef"],
            device=device  # Pass the device here
        )
    elif model_type == "A2C":
        model = A2C(
            "MlpPolicy",
            env,
            verbose=1,
            n_steps=model_config.get("n_steps", 5),
            ent_coef=model_config.get("ent_coef", 0.01),
            device=device,
            learning_rate=model_config.get("learning_rate", 0.0007),
            gamma=model_config.get("gamma", 0.99),
            gae_lambda=model_config.get("gae_lambda", 0.95),
            max_grad_norm=model_config.get("max_grad_norm", 0.5),
            # policy_kwargs={
            #     "net_arch": model_config.get("net_arch", [{"pi": [64, 64], "vf": [64, 64]}]),
            #     "activation_fn": getattr(nn, model_config.get("activation_fn", "ReLU"))()
            # }
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model_type = None
    if isinstance(model, PPO):
        model_type = "PPO"
        print("The model is PPO.")
    elif isinstance(model, A2C):
        model_type = "A2C"
        print("The model is A2C.")
    else:
        raise ValueError("Unsupported model type or instance.")

    # Step 4: Train the model
    print(f"Starting training for {model_config['total_timesteps']} timesteps...")
    model.learn(total_timesteps=model_config["total_timesteps"])

    # Step 5: Save the trained model and logs with the same timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("Training complete. Saving the model...")
    model_path = save_model(model, env_name=env_name, model_type=model_type, timestamp=timestamp)

    # Step 6: Evaluate the agent
    print("Evaluating the agent...")
    mean_reward, std_reward = evaluate_agent(model, env, n_eval_episodes=100)

    # Step 7: Save evaluation results
    print("Saving evaluation results...")
    logs = {
        "env_name": env_name,
        "total_timesteps": model_config["total_timesteps"],
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "model_path": model_path,
        "training_config": model_config
    }
    save_logs(logs, env_name=env_name, model_type=model_type, timestamp=timestamp)
    print("Training and evaluation complete. Logs saved.")

if __name__ == "__main__":
    print("Starting training pipeline...")
    try:
        # Load the environment configuration
        print("Loading environment configuration...")
        env_config = load_config("config/training_config_env.yml")
        print("Environment configuration loaded successfully.")

        # Load the model configuration
        print("Loading model configuration...")
        model_config = load_config("config/training_config_A2C.yml")
        print("Model configuration loaded successfully.")
    except (FileNotFoundError, PermissionError, yaml.YAMLError, ValueError) as e:
        print(f"Error loading configuration: {e}")
        exit(1)

    # Train the agent
    try:
        print("Beginning training...")
        train_agent(env_config, model_config)
        print("Training pipeline completed successfully.")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        exit(1)
