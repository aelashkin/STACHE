import os
import optuna
import gymnasium as gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from src.wrappers import FactorizedSymbolicWrapper, PaddedObservationWrapper
from src.utils import save_model, evaluate_agent, load_config
from datetime import datetime

def create_model(model_name, env, **kwargs):
    if model_name == "PPO":
        return PPO("MlpPolicy", env, **kwargs)
    elif model_name == "A2C":
        return A2C("MlpPolicy", env, **kwargs)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def optimize_model(trial, config):
    model_name = trial.suggest_categorical("model_name", ["PPO", "A2C"])
    n_steps = trial.suggest_int("n_steps", 64, 512, step=64)
    batch_size = trial.suggest_int("batch_size", 16, 128, step=16)
    ent_coef = trial.suggest_loguniform("ent_coef", 1e-5, 1e-1)

    # Step 1: Initialize the environment
    env_name = config["env_name"]
    env = gym.make(env_name)
    env = FactorizedSymbolicWrapper(env)
    env = PaddedObservationWrapper(env, max_objects=config["max_objects"], max_walls=config["max_walls"])
    env = DummyVecEnv([lambda: env])

    # Step 2: Create the model
    model = create_model(model_name, env, n_steps=n_steps, batch_size=batch_size, ent_coef=ent_coef)

    # Step 3: Train the model
    model.learn(total_timesteps=200000)

    # Step 4: Evaluate the model
    mean_reward, _ = evaluate_agent(model, env, n_eval_episodes=50)
    return mean_reward

def main():
    # Load the configuration
    config = load_config("config/training_config.yml")

    # Create a study object
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: optimize_model(trial, config), n_trials=50)

    # Save the best model
    best_params = study.best_params
    best_model_name = best_params["model_name"]
    best_n_steps = best_params["n_steps"]
    best_batch_size = best_params["batch_size"]
    best_ent_coef = best_params["ent_coef"]

    # Initialize the environment again for the best model
    env_name = config["env_name"]
    env = gym.make(env_name)
    env = FactorizedSymbolicWrapper(env)
    env = PaddedObservationWrapper(env, max_objects=config["max_objects"], max_walls=config["max_walls"])
    env = DummyVecEnv([lambda: env])

    # Create and train the best model
    best_model = create_model(best_model_name, env, n_steps=best_n_steps, batch_size=best_batch_size, ent_coef=best_ent_coef)
    best_model.learn(total_timesteps=200000)

    # Save the best model and logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = save_model(best_model, env_name=env_name, timestamp=timestamp)
    logs = {
        "best_params": best_params,
        "model_path": model_path,
        "mean_reward": study.best_value
    }
    save_logs(logs, env_name=env_name, timestamp=timestamp)

if __name__ == "__main__":
    main()