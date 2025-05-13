from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv # Recommended for CPU-bound tasks
from stable_baselines3.common.evaluation import evaluate_policy


import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper
# Removed: from minigrid.wrappers import FlatObsWrapper

def create_env_fn(env_name="MiniGrid-Fetch-5x5-N2-v0", render_mode=None):
    env = gym.make(env_name, render_mode=render_mode)
    env = FullyObsWrapper(env) # This provides a Dict observation
    return env


if __name__ == '__main__':

    env_id = "MiniGrid-Fetch-5x5-N2-v0"
    N_ENVS = 8 # Number of parallel environments

    # Create the vectorized environment
    # The lambda function is a common way to pass arguments to the env creator
    vec_env = make_vec_env(lambda: create_env_fn(env_id), n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)

    # PPO hyperparameters
    ppo_hyperparams = {
        "learning_rate": 0.0003,       # Adjusted: 3e-4
        "n_steps": 1024,               # Adjusted: Number of steps to run for each environment per update
        "batch_size": 64,              # Mini-batch size for PPO updates
        "n_epochs": 10,                # Adjusted: Number of epochs when optimizing the surrogate loss
        "gamma": 0.99,                 # Adjusted: Discount factor
        "gae_lambda": 0.95,            # Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        "ent_coef": 0.01,              # Adjusted: Entropy coefficient for exploration
        "vf_coef": 0.5,                # Value function coefficient in the loss calculation
        "max_grad_norm": 0.5,          # Maximum value for gradient clipping
        "normalize_advantage": True,   # Normalize advantage (often helpful)
    }
    
    # PPO policy_kwargs
    policy_kwargs_ppo = dict(
        net_arch=[dict(pi=[128, 128], vf=[128, 128])] # Separate 2-layer MLPs for policy and value functions
        # Or a shared architecture: net_arch=[128, 128]
    )

    model = PPO(
        "MultiInputPolicy",
        vec_env,
        policy_kwargs=policy_kwargs_ppo,
        verbose=1,
        tensorboard_log="./ppo_minigrid_fetch_tensorboard/",
        **ppo_hyperparams
    )

    print(f"Starting training PPO with MultiInputPolicy on {env_id}...")
    model.learn(total_timesteps=200_000) # Increase training duration

    # Step: Evaluate the agent on the training environment

    train_env = create_env_fn(env_id) # Create a new environment for evaluation
    n_eval_episodes = 100
    print("Evaluating the agent on training environment...")
    mean_reward, std_reward = evaluate_policy(model, train_env, n_eval_episodes=n_eval_episodes, deterministic=True)

    model.save("ppo_minigrid_fetch5x5_multi_input")
    print("Training complete. Model saved.")