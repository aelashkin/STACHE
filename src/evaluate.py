import os
import gymnasium as gym
from stable_baselines3 import PPO
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3.common.evaluation import evaluate_policy
from src.environment_utils import create_symbolic_minigrid_env, create_standard_minigrid_env

def evaluate_model(model_path, env_name='MiniGrid-Fetch-5x5-N2-v0', n_eval_episodes=10):
    """
    Load a PPO model from the given path and evaluate it on the specified environment.
    
    Parameters:
model_path (str): Path to the saved model .zip file.
env_name (str): Name of the MiniGrid environment to use.
n_eval_episodes (int): Number of episodes for evaluation.

    Returns:
        None
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
    
    Parameters:
env_config (dict): Configuration dictionary with environment parameters.

    Returns:
        None
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



def evaluate_policy_performance(model, env_config, n_eval_episodes=50, histogram_bins=20):
    """
    Evaluate the model on a symbolic environment, collecting rewards over multiple episodes.

    Parameters:
        model (object): Trained RL model with a predict() method.
        env_config (dict): Environment configuration dict.
        n_eval_episodes (int): Number of episodes to evaluate.
        histogram_bins (int): Bin count for reward histogram.

    Returns:
        dict: Summary statistics including mean, median, std, min, max, and distribution.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Create environment using the symbolic MiniGrid environment
    env_config["render_mode"] = None  # Disable rendering for evaluation
    env = create_symbolic_minigrid_env(env_config)
    
    rewards = []
    for ep in range(n_eval_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
    env.close()
    
    rewards = np.array(rewards)
    mean_reward = rewards.mean()
    median_reward = np.median(rewards)
    std_reward = rewards.std()
    min_reward = rewards.min()
    max_reward = rewards.max()
    
    print("Evaluation over {} episodes:".format(n_eval_episodes))
    print("Mean Reward: {:.2f}".format(mean_reward))
    print("Median Reward: {:.2f}".format(median_reward))
    print("Std Reward: {:.2f}".format(std_reward))
    print("Min Reward: {:.2f}".format(min_reward))
    print("Max Reward: {:.2f}".format(max_reward))
    print("Reward distribution per episode: ", rewards.tolist())
    
    # Generate a histogram for the reward distribution
    plt.figure(figsize=(8, 6))
    plt.hist(rewards, bins=histogram_bins, edgecolor='black', alpha=0.75)
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.title('Reward Histogram over {} Episodes'.format(n_eval_episodes))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    
    return {
        "mean_reward": mean_reward,
        "median_reward": median_reward,
        "std_reward": std_reward,
        "min_reward": min_reward,
        "max_reward": max_reward,
        "reward_distribution": rewards.tolist(),
    }


def evaluate_single_policy_run(model, env_config, seed=42):
    """
    Run a single evaluation episode with a fixed seed, saving frames and logs.

    Parameters:
        model (object): Trained RL model.
        env_config (dict): Environment config dict.
        seed (int): The random seed for reproducibility.

    Returns:
        None
    """
    import os
    from datetime import datetime
    from PIL import Image

    model_name = "evaluation_model"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_dir = os.path.join("data", "experiments", "evaluation", model_name, f"{seed}_{timestamp}")
    os.makedirs(base_dir, exist_ok=True)
    images_dir = os.path.join(base_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    log_file_path = os.path.join(base_dir, "evaluation_log.txt")
    log_file = open(log_file_path, "w")
    
    # Create the environment using the symbolic MiniGrid environment and reset with a fixed seed
    env = create_symbolic_minigrid_env(env_config)
    obs, info = env.reset(seed=seed)
    log_file.write(f"Episode start (seed={seed}):\n")
    log_file.write(f"Initial observation: {obs}\n")
    log_file.write(f"Initial info: {info}\n\n")
    
    # Save the initial frame
    step_count = 0
    frame = env.render(mode="rgb_array")
    if frame is not None:
        img_path = os.path.join(images_dir, f"step_{step_count:03d}.png")
        Image.fromarray(frame).save(img_path)
        log_file.write(f"Step {step_count}: Saved initial frame at {img_path}\n\n")
    
    done = False
    while not done:
        # Get action from the policy
        action, _ = model.predict(obs, deterministic=True)
        log_file.write(f"Step {step_count}: Observation: {obs}\n")
        log_file.write(f"Step {step_count}: Action taken: {action}\n")
        
        # Take a step in the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        log_file.write(f"Step {step_count}: Reward: {reward}\n")
        log_file.write(f"Step {step_count}: Terminated: {terminated}, Truncated: {truncated}\n")
        log_file.write(f"Step {step_count}: Info: {info}\n")
        
        # Capture and save the current frame (using rgb_array mode for image capture)
        frame = env.render(mode="rgb_array")
        if frame is not None:
            img_path = os.path.join(images_dir, f"step_{step_count:03d}.png")
            Image.fromarray(frame).save(img_path)
            log_file.write(f"Step {step_count}: Saved frame at {img_path}\n")
        
        log_file.write("\n")
        step_count += 1
    
    log_file.write("Episode ended.\n")
    log_file.close()
    env.close()
    
    print(f"Detailed evaluation saved at: {base_dir}")


if __name__ == "__main__":
    from src.utils import load_experiment
    experiment_dir = "data/experiments/MiniGrid-Fetch-5x5-N2-v0_PPO_model_20250216_040438"
    model, config_data = load_experiment(experiment_dir)
    env_config = config_data["env_config"]
    
    run_statistics_evaluation = True
    run_detailed_evaluation = False

    if run_statistics_evaluation:
        print("Running policy statistics evaluation over 50 episodes...")
        stats = evaluate_policy_performance(model, env_config, n_eval_episodes=50)
        print("Statistics:", stats)
    
    if run_detailed_evaluation:
        print("Running detailed policy evaluation for a single episode with seed 42...")
        evaluate_single_policy_run(model, env_config, seed=42)
