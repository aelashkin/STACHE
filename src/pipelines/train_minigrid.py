import sys
import os
import traceback
from datetime import datetime
import yaml
import copy

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy

from utils.experiment_io import save_experiment, load_config, get_device
from minigrid_ext.environment_utils import create_minigrid_env, MinigridFeaturesExtractor
from utils.experiment_io import ModelType

MODEL_TYPE = ModelType.PPO

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np

class CustomEvalAndSaveCallback(BaseCallback):
    """
    Custom callback for evaluating the model during training and saving the best model.
    
    Evaluates the model every max(total_timesteps // 20, 5000) timesteps over n_eval_episodes.
    If the current evaluation shows a higher mean reward than all previous evaluations,
    the model is immediately saved (using save_model), replacing the old best model.
    All evaluation results are stored for creating a complete training log.
    
    Parameters:
        eval_env (gym.Env): Environment used for evaluation.
        experiment_dir (str): Directory where the best model will be saved.
        total_timesteps (int): Total number of training timesteps.
        n_eval_episodes (int): Number of episodes to evaluate on.
        verbose (int): Verbosity level (0: no output, 1: info).
    """
    def __init__(self, eval_env, experiment_dir, total_timesteps, n_eval_episodes=100, verbose=1):
        super(CustomEvalAndSaveCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.experiment_dir = experiment_dir
        self.total_timesteps = total_timesteps
        self.n_eval_episodes = n_eval_episodes
        self.eval_interval = max(5000, total_timesteps // 20) # Evaluate no less than 5000 timesteps
        self.next_eval = self.eval_interval
        self.best_mean_reward = -np.inf
        self.best_std_reward = None
        self.best_eval_timestamp = None
        self.evaluation_results = []  # List to store all evaluation results

    def _on_step(self) -> bool:
        # Check if it is time to evaluate
        if self.num_timesteps >= self.next_eval:
            # Run evaluation over n_eval_episodes (using deterministic actions)
            mean_reward, std_reward = evaluate_policy(self.model, self.eval_env, 
                                                        n_eval_episodes=self.n_eval_episodes, 
                                                        deterministic=True)
            # Record the evaluation details
            eval_info = {
                "timestamp": self.num_timesteps,
                "mean_reward": mean_reward,
                "std_reward": std_reward
            }
            self.evaluation_results.append(eval_info)
            if self.verbose:
                print(f"Evaluation at timestep {self.num_timesteps}: Mean Reward = {mean_reward:.2f}, Std Reward = {std_reward:.2f}")
            # If the evaluation is better than previous best, update best and save the model immediately
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.best_std_reward = std_reward
                self.best_eval_timestamp = self.num_timesteps
                # Save the best model using the provided save_model function
                from utils import save_model
                save_model(self.model, self.experiment_dir)
                if self.verbose:
                    print(f"New best model found at timestep {self.num_timesteps} with mean reward {mean_reward:.2f}")
            # Schedule the next evaluation
            self.next_eval += self.eval_interval
        return True

    def _on_training_end(self) -> None:
        if self.verbose:
            print(f"Training completed for {self.total_timesteps} timesteps.")
            print(f"Using device: {self.model.device}")
            print(f"Evaluation of the best agent (on timestamp {self.best_eval_timestamp}) over {self.n_eval_episodes} episodes:")
            print(f"Mean Reward: {self.best_mean_reward:.2f}, Std Reward: {self.best_std_reward:.2f}")




def train_agent(env_config, model_config):
    """
    Train a reinforcement learning agent based on provided configurations.
    
    Sets up the environment and model according to configurations, trains the model,
    evaluates its performance, and saves all experiment artifacts.
    
    Parameters:
        env_config (dict): Environment configuration containing parameters like:
            - env_name: Name of the Gymnasium/MiniGrid environment.
            - representation: Observation type ("symbolic", "image", or "standard").
            - reward_wrappers: Optional reward modification wrappers.
            - max_objects: Maximum number of objects for symbolic representation.
            - max_walls: Maximum number of walls for symbolic representation.
            - evaluate_with_modified_reward: Whether to use reward wrappers during evaluation.
            
        model_config (dict): Model configuration containing parameters like:
            - model_type: Type of RL algorithm ("PPO" or "A2C").
            - total_timesteps: Number of timesteps to train.
            - n_steps, batch_size, learning_rate, etc.: Algorithm-specific parameters.
            - device: Computing device to use ("cpu", "cuda", or "mps").
    
    Notes:
        The function saves the trained model, configuration files, and training logs 
        to the experiment directory using save_experiment(). It does not return any values.
    """
    # Generate a unified experiment directory under data/experiments/models
    experiment_base_dir = "data/experiments/models"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_name = env_config.get("env_name", "unknown_env")
    model_type = model_config.get("model_type", "unknown_model")
    experiment_folder_name = f"{env_name}_{model_type}_model_{timestamp}"
    experiment_dir = os.path.join(experiment_base_dir, experiment_folder_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # Step 1: Determine the device
    device = get_device(model_config.get("device"))
    print(f"Using device: {device}")

    # Disable rendering during training
    env_config["render_mode"] = None

    # Step 2: Initialize the environment based on the config
    train_env = create_minigrid_env(env_config)
    representation = env_config.get("representation")

    # Step 3: Choose policy based on representation
    if representation == "image":
        # Use CNN policy for image-based observations
        # Note: MinigridFeaturesExtractor is a custom feature extractor for image observations
        policy = "CnnPolicy"
        policy_kwargs = dict(
            features_extractor_class=MinigridFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=128)
        )
        print("Using CnnPolicy for image-based observation.")
    else:
        policy = "MlpPolicy"
        policy_kwargs = {}
        print("Using MlpPolicy for non-image (symbolic or standard) observations.")

    # Step 4: Initialize the model (PPO or A2C)
    model_type_str = model_config["model_type"]
    print(f"Setting up the {model_type_str} model...")

    if model_type_str == "PPO":
        model = PPO(
            policy,
            train_env,
            verbose=1,
            n_steps=model_config.get("n_steps"),
            batch_size=model_config.get("batch_size"),
            ent_coef=model_config.get("ent_coef"),
            learning_rate=model_config.get("learning_rate"),
            gamma=model_config.get("gamma"),
            gae_lambda=model_config.get("gae_lambda"),
            n_epochs=model_config.get("n_epochs"),
            normalize_advantage=model_config.get("normalize_advantage"),
            device=device,
            policy_kwargs=policy_kwargs,
        )
    elif model_type_str == "A2C":
        model = A2C(
            policy,
            train_env,
            verbose=1,
            n_steps=model_config.get("n_steps"),
            ent_coef=model_config.get("ent_coef"),
            device=device,
            learning_rate=model_config.get("learning_rate"),
            gamma=model_config.get("gamma"),
            gae_lambda=model_config.get("gae_lambda"),
            max_grad_norm=model_config.get("max_grad_norm"),
            policy_kwargs=policy_kwargs,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type_str}")

    if isinstance(model, PPO):
        print("The model is PPO.")
    elif isinstance(model, A2C):
        print("The model is A2C.")
    else:
        raise ValueError("Unsupported model type or instance.")


    # Initialize the evaluation environment
    eval_env_config = copy.deepcopy(env_config)
    if eval_env_config.get("evaluate_with_modified_reward") == False:
        # Disable reward wrappers for evaluation
        eval_env_config["reward_wrappers"] = None

    eval_env = create_minigrid_env(eval_env_config)

    total_timesteps = env_config["total_timesteps"]

    # Instantiate the custom evaluation callback
    eval_callback = CustomEvalAndSaveCallback(
        eval_env=eval_env, 
        experiment_dir=experiment_dir,
        total_timesteps=total_timesteps, 
        n_eval_episodes=100, 
        verbose=1
    )

    # Step 5: Train the model
    print(f"Starting training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback, log_interval=10)



    # Step 6: Evaluate the agent on the training environment
    n_eval_episodes = 100
    print("Evaluating the agent on training environment...")
    mean_reward, std_reward = evaluate_policy(model, train_env, n_eval_episodes=n_eval_episodes, deterministic=True)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

    # Step 7: Create a training log
    training_log_header = (
        f"Training completed for {total_timesteps} timesteps.\n"
        f"Using device: {device}\n"
        f"Evaluation of the best agent (on timestamp {eval_callback.best_eval_timestamp}) over 100 episodes:\n"
        f"Mean Reward: {eval_callback.best_mean_reward:.2f}, Std Reward: {eval_callback.best_std_reward:.2f}\n\n"
        f"*Evaluation of all agents over {n_eval_episodes} episodes, in chronological order*\n"
    )
    log_lines = [training_log_header]
    for eval_info in eval_callback.evaluation_results:
        log_lines.append(f"{eval_info['timestamp']} - Mean Reward: {eval_info['mean_reward']:.2f}, Std Reward: {eval_info['std_reward']:.2f}")
    training_log = "\n".join(log_lines)

    # Step 8: Save the experiment using the provided experiment directory.
    # Since the best model has already been saved during training via the callback,
    # we pass None as the model argument.
    experiment_info = save_experiment(None, env_config, model_config, training_log, experiment_dir=experiment_dir)
    print("Training and evaluation complete. Experiment saved:")
    print(experiment_info)






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
        traceback.print_exc()
        exit(1)
