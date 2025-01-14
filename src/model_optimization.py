import os
import optuna
import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from src.wrappers import FactorizedSymbolicWrapper, PaddedObservationWrapper
from src.utils import save_model, save_logs, evaluate_agent, load_config, get_device
from datetime import datetime

def get_valid_batch_sizes(n_steps):
    """
    Get valid batch sizes that are factors of n_steps.
    """
    return [2**i for i in range(5, 9) if n_steps % (2**i) == 0]

def optimize_agent(trial, config):
    """
    Optimize the agent using Optuna.
    """
    # Step 1: Determine the device
    device = get_device(config.get("device"))

    # Step 2: Initialize the environment
    env_name = config["env_name"]
    env = gym.make(env_name)
    env = FullyObsWrapper(env)
    env = FactorizedSymbolicWrapper(env)
    env = PaddedObservationWrapper(env, max_objects=config["max_objects"], max_walls=config["max_walls"])
    env = DummyVecEnv([lambda: env])

    # Step 3: Suggest hyperparameters
    while True:
        n_steps = trial.suggest_int("n_steps", 128, 2048)
        valid_batch_sizes = get_valid_batch_sizes(n_steps)
        if valid_batch_sizes:
            break
    batch_size = trial.suggest_categorical("batch_size", valid_batch_sizes)
    ent_coef = trial.suggest_float("ent_coef", 1e-8, 1e-2, log=True)

    # Step 4: Initialize the model
    model_class = PPO if config.get("model", "PPO") == "PPO" else A2C
    model = model_class(
        "MlpPolicy",
        env,
        verbose=0,
        n_steps=n_steps,
        batch_size=batch_size,
        ent_coef=ent_coef,
        device=device
    )

    # Step 5: Train the model
    print("Starting model training...")
    model.learn(total_timesteps=config.get("trial_timesteps", 50000))
    print("Model training completed.")

    # Step 6: Evaluate the agent
    print("Starting agent evaluation...")
    mean_reward, _ = evaluate_agent(model, env, n_eval_episodes=50)
    print("Agent evaluation completed.")

    return mean_reward

def main():
    # Load the configuration
    config = load_config("config/training_config.yml")

    # Create an Optuna study
    study = optuna.create_study(direction="maximize")

    # Print study.optimize parameters
    n_trials = 100
    n_jobs = os.cpu_count()
    print(f"Number of trials: {n_trials}")
    print(f"Number of CPU cores in use: {n_jobs}")

    # Optimize the agent
    study.optimize(lambda trial: optimize_agent(trial, config), n_trials=n_trials, n_jobs=6, show_progress_bar=True)

    # Get the best trial
    best_trial = study.best_trial
    print(f"Best trial: {best_trial.params}")

    # Save the best model and logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_name = config["env_name"]
    device = get_device(config.get("device"))

    # Initialize the environment
    env = gym.make(env_name)
    env = FullyObsWrapper(env)
    env = FactorizedSymbolicWrapper(env)
    env = PaddedObservationWrapper(env, max_objects=config["max_objects"], max_walls=config["max_walls"])
    env = DummyVecEnv([lambda: env])

    # Initialize the best model
    model_class = PPO if config.get("model", "PPO") == "PPO" else A2C
    model = model_class(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=best_trial.params["n_steps"],
        batch_size=best_trial.params["batch_size"],
        ent_coef=best_trial.params["ent_coef"],
        device=device
    )

    # Train the best model
    model.learn(total_timesteps=config.get("trial_timesteps", 200000))

    # Save the trained model and logs
    model_path = save_model(model, env_name=env_name, timestamp=timestamp)
    mean_reward, std_reward = evaluate_agent(model, env, n_eval_episodes=100)
    logs = {
        "env_name": env_name,
        "total_timesteps": config["trial_timesteps"],
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "model_path": model_path,
        "training_config": config,
        "best_params": best_trial.params
    }
    save_logs(logs, env_name=env_name, timestamp=timestamp)

if __name__ == "__main__":
    main()