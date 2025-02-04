import os
import gymnasium as gym
from stable_baselines3 import PPO
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3.common.evaluation import evaluate_policy
from src.environment_utils import create_symbolic_minigrid_env, create_standard_minigrid_env

def evaluate_model(model_path, env_name='MiniGrid-Fetch-5x5-N2-v0', n_eval_episodes=10):
    """
    Load the PPO model from the given path and evaluate it on the specified environment.
    
    :param model_path: Path to the model zip file
    :param env_name: Name of the MiniGrid environment
    :param n_eval_episodes: Number of episodes to evaluate the model on
    """
    # Load environment
    env = gym.make(env_name, render_mode='rgb_array')
    env = FlatObsWrapper(env)  # Use flat observation wrapper
    
    # Load model
    model = PPO.load(model_path)
    
    # Evaluate model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, deterministic=True)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")
    
    env.close()

def evaluate_symbolic_env(env_config):
    """
    Create a symbolic MiniGrid environment and interact with it step-by-step.
    
    :param env_config: Configuration dictionary for the environment
    """
    env = create_symbolic_minigrid_env(env_config)
    # env = create_standard_minigrid_env(env_config)
    obs, info = env.reset()
    done = False
    step_count = 0

    while not done and step_count < 4:  # Limit to 10 steps for demonstration
        env.render()  # Render the environment
        print(f"Step {step_count}: {obs}")
        action = env.action_space.sample()  # Random action, replace with your own logic
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(f"Observation: {obs}")
        print(f"Reward: {reward}")
        print(f"Terminated: {terminated}")
        print(f"Truncated: {truncated}")
        print(f"Info: {info}")
        step_count += 1

    env.close()

if __name__ == "__main__":
    # model_name = "huggingface_ppo-MiniGrid-Fetch-5x5-N2-v0.zip"  # Change to your actual model name
    # model_path = os.path.join("data", "models", model_name)
    
    # if os.path.exists(model_path):
    #     evaluate_model(model_path)
    # else:
    #     print(f"Model file {model_path} not found.")
    
    # Example configuration for symbolic environment
    env_config = {
        "env_name": "MiniGrid-Fetch-5x5-N2-v0",
        "max_objects": 10,
        "max_walls": 25,
        "representation": "symbolic",
        "render_mode": "human"
    }
    evaluate_symbolic_env(env_config)
