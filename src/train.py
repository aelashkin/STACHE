import sys
import os

pythonpath = os.getenv("PYTHONPATH")
if (pythonpath and pythonpath not in sys.path):
    sys.path.append(pythonpath)

from stable_baselines3 import PPO, A2C
from src.utils import save_model, save_logs, evaluate_agent, load_config, get_device
from src.environment_utils import create_minigrid_env, MinigridFeaturesExtractor
from src.utils import ModelType

from datetime import datetime
import yaml

MODEL_TYPE = ModelType.PPO


def train_agent(env_config, model_config):
    """
    Train the agent and save the model and logs.
    """
    # Step 1: Determine the device
    device = get_device(model_config.get("device"))
    print(f"Using device: {device}")

    # Step 2: Initialize the environment based on the config
    env = create_minigrid_env(env_config)
    env_name = env_config["env_name"]
    representation = env_config.get("representation")

    # Step 3: Choose policy based on representation
    if representation == "image":
        # Use CNN-based policy
        policy = "CnnPolicy"
        policy_kwargs = dict(
            features_extractor_class=MinigridFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=128)
        )
        print("Using CnnPolicy for image-based observation.")
    else:
        # Use MLP-based policy
        policy = "MlpPolicy"
        policy_kwargs = {}
        print("Using MlpPolicy for non-image (symbolic or standard) observations.")

    # Step 4: Initialize the model (PPO or A2C)
    model_type = model_config["model_type"]
    print(f"Setting up the {model_type} model...")

    if model_type == "PPO":
        model = PPO(
            policy,
            env,
            verbose=1,
            n_steps=model_config.get("n_steps", 2048),
            batch_size=model_config.get("batch_size", 64),
            ent_coef=model_config.get("ent_coef", 0.0),
            device=device,
            policy_kwargs=policy_kwargs,
        )
    elif model_type == "A2C":
        model = A2C(
            policy,
            env,
            verbose=1,
            n_steps=model_config.get("n_steps", 5),
            ent_coef=model_config.get("ent_coef", 0.01),
            device=device,
            learning_rate=model_config.get("learning_rate", 0.0007),
            gamma=model_config.get("gamma", 0.99),
            gae_lambda=model_config.get("gae_lambda", 0.95),
            max_grad_norm=model_config.get("max_grad_norm", 0.5),
            policy_kwargs=policy_kwargs,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Determine model type string for saving/evaluation
    if isinstance(model, PPO):
        model_type_str = "PPO"
        print("The model is PPO.")
    elif isinstance(model, A2C):
        model_type_str = "A2C"
        print("The model is A2C.")
    else:
        raise ValueError("Unsupported model type or instance.")

    # Step 5: Train the model (wraps in Monitor and DummyVecEnv internally)
    print(f"Starting training for {model_config['total_timesteps']} timesteps...")
    model.learn(total_timesteps=model_config["total_timesteps"])

    # Step 6: Save the trained model and logs with the same timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("Training complete. Saving the model...")
    model_path = save_model(model, env_name=env_name, model_type=model_type_str, timestamp=timestamp)

    # Step 7: Evaluate the agent
    print("Evaluating the agent...")
    mean_reward, std_reward = evaluate_agent(model, env, n_eval_episodes=100)

    # Step 8: Save evaluation results
    print("Saving evaluation results...")
    logs = {
        "env_name": env_name,
        "total_timesteps": model_config["total_timesteps"],
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "model_path": model_path,
        "training_config": model_config
    }
    save_logs(logs, env_name=env_name, model_type=model_type_str, timestamp=timestamp)
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
        if MODEL_TYPE == ModelType.PPO:
            model_config = load_config("config/training_config_PPO.yml")
        elif MODEL_TYPE == ModelType.A2C:
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
