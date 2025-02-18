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



def evaluate_policy_statistics(model_path, env_config, n_eval_episodes=50, histogram_bins=20):
    """
    Load the PPO model from the given path and evaluate it on the specified symbolic environment
    over multiple episodes. Computes statistics including mean, median, standard deviation,
    minimum, and maximum reward as well as the reward distribution. Additionally, a histogram
    of rewards is generated using the specified number of bins.

    :param model_path: Path to the model zip file.
    :param env_config: Configuration dictionary for the environment.
    :param n_eval_episodes: Number of episodes to evaluate (default is 50).
    :param histogram_bins: Number of bins for the histogram (default is 20).
    :return: Dictionary containing reward statistics.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Create environment using the symbolic MiniGrid environment
    env_config["render_mode"] = None  # Disable rendering for evaluation
    env = create_symbolic_minigrid_env(env_config)
    model = PPO.load(model_path)
    
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


def evaluate_detailed_policy_run(model_path, env_config, seed=42):
    """
    Run a detailed evaluation of the PPO model on a single episode with a fixed seed.
    At each step, logs the observation, the action chosen by the policy, reward, termination flags,
    and additional debug info. In addition, the function saves the rendered image (using rgb_array mode)
    for each step into a subfolder. The outputs are saved in:
    
        data/experiments/evaluation/<model_name>/<seed>_<timestamp>/
    
    This folder will contain a text log (evaluation_log.txt) and a subfolder "images" with one image per step.
    
    :param model_path: Path to the PPO model zip file.
    :param env_config: Configuration dictionary for the environment.
    :param seed: Seed to use for environment reset (default is 42).
    """
    import os
    from datetime import datetime
    from PIL import Image

    # Extract model name from the model file path (e.g., "huggingface_ppo-MiniGrid-Fetch-5x5-N2-v0")
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    
    # Create the output directory with a timestamp and the seed number
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_dir = os.path.join("data", "experiments", "evaluation", model_name, f"{seed}_{timestamp}")
    os.makedirs(base_dir, exist_ok=True)
    images_dir = os.path.join(base_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    log_file_path = os.path.join(base_dir, "evaluation_log.txt")
    log_file = open(log_file_path, "w")
    
    # Load the model
    model = PPO.load(model_path)
    
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


# if __name__ == "__main__":
#     # model_name = "huggingface_ppo-MiniGrid-Fetch-5x5-N2-v0.zip"  # Change to your actual model name
#     # model_path = os.path.join("data", "models", model_name)
    
#     # if os.path.exists(model_path):
#     #     evaluate_model(model_path)
#     # else:
#     #     print(f"Model file {model_path} not found.")
    
#     # Example configuration for symbolic environment
#     env_config = {
#         "env_name": "MiniGrid-Fetch-5x5-N2-v0",
#         "max_objects": 10,
#         "max_walls": 25,
#         "representation": "symbolic",
#         "render_mode": "human"
#     }
#     evaluate_symbolic_env(env_config)


if __name__ == "__main__":
    # Model path for evaluation
    model_path = "data/experiments/MiniGrid-Fetch-5x5-N2-v0_PPO_model_20250216_040438/model.zip"
    
    # Environment configuration for symbolic evaluation
    env_config = {
        "env_name": "MiniGrid-Fetch-5x5-N2-v0",
        "max_objects": 10,
        "max_walls": 32,
        "representation": "symbolic",
        "render_mode": "human"
    }
    
    # Manual toggle for evaluation modes
    run_statistics_evaluation = True
    run_detailed_evaluation = True

    if run_statistics_evaluation:
        print("Running policy statistics evaluation over 50 episodes...")
        stats = evaluate_policy_statistics(model_path, env_config, n_eval_episodes=50)
        print("Statistics:", stats)
    
    if run_detailed_evaluation:
        print("Running detailed policy evaluation for a single episode with seed 42...")
        evaluate_detailed_policy_run(model_path, env_config, seed=42)
