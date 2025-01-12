import sys
import os
import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.wrappers import FactorizedSymbolicWrapper, PaddedObservationWrapper
from src.utils import save_model, save_logs, evaluate_agent, tqdm_monitor_training, load_config, get_device

def train_agent(config):
    """
    Train the agent and save the model and logs.
    """
    # Step 1: Determine the device
    device = get_device(config.get("device"))
    print(f"Using device: {device}")

    # Step 2: Initialize the environment
    env_name = config["env_name"]
    print(f"Initializing the environment: {env_name}")
    env = gym.make(env_name)
    env = FullyObsWrapper(env)
    env = FactorizedSymbolicWrapper(env)
    env = PaddedObservationWrapper(env, max_objects=config["max_objects"], max_walls=config["max_walls"])
    env = DummyVecEnv([lambda: env])

    # Step 3: Initialize the PPO model
    print("Setting up the PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        ent_coef=config["ent_coef"],
        device=device  # Pass the device here
    )

    # Step 4: Train the model
    print(f"Starting training for {config['total_timesteps']} timesteps...")
    tqdm_monitor_training(model, total_timesteps=config["total_timesteps"])

    # Step 5: Save the trained model
    print("Training complete. Saving the model...")
    model_path = save_model(model)

    # Step 6: Evaluate the agent
    print("Evaluating the agent...")
    mean_reward, std_reward = evaluate_agent(model, env, n_eval_episodes=100)

    # Step 7: Save evaluation results
    print("Saving evaluation results...")
    logs = {
        "env_name": env_name,
        "total_timesteps": config["total_timesteps"],
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "model_path": model_path,
    }
    save_logs(logs)
    print("Training and evaluation complete. Logs saved.")

if __name__ == "__main__":
    print("Starting training pipeline...")
    try:
        # Load the configuration
        print("Loading configuration...")
        config = load_config("config/training_config.yml")
        print("Configuration loaded successfully.")
    except (FileNotFoundError, PermissionError, yaml.YAMLError, ValueError) as e:
        print(f"Error loading configuration: {e}")
        exit(1)

    # Train the agent
    try:
        print("Beginning training...")
        train_agent(config)
        print("Training pipeline completed successfully.")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        exit(1)
