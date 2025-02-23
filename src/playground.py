import gymnasium as gym
import minigrid
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch
from gymnasium.wrappers import FlattenObservation
from minigrid.core.mission import MissionSpace
from minigrid.wrappers import ImgObsWrapper
import numpy as np


class CountBasedExplorationWrapper(gym.Wrapper):
    def __init__(self, env, coefficient=0.005):
        super().__init__(env)
        self.counts = {}
        self.coefficient = coefficient

    def _get_state_key(self):
        agent_x, agent_y = self.env.agent_pos
        direction = self.env.agent_dir
        return (agent_x, agent_y, direction)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        state_key = self._get_state_key()
        self.counts[state_key] = self.counts.get(state_key, 0) + 1
        bonus = self.coefficient / np.sqrt(self.counts[state_key])
        modified_reward = reward + bonus
        return obs, modified_reward, done, info

# Step 1: Define a custom wrapper to handle the observation space
class CustomObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(CustomObsWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Dict({
            'image': env.observation_space['image'],
            'mission': gym.spaces.Box(low=0, high=1, shape=(8,)),
            'direction': gym.spaces.Box(low=0, high=1, shape=(4,))
        })

    def observation(self, obs):
        mission_str = obs['mission']
        words = mission_str.split()
        color = words[-2]
        obj_type = words[-1]
        colors = ['red', 'green', 'blue', 'purple', 'yellow', 'grey']
        types = ['key', 'ball']
        color_idx = colors.index(color)
        type_idx = types.index(obj_type)
        mission_encoding = np.zeros(8)
        mission_encoding[color_idx] = 1
        mission_encoding[6 + type_idx] = 1
        direction = obs['direction']
        direction_onehot = np.zeros(4)
        direction_onehot[direction] = 1
        return {
            'image': obs['image'],
            'mission': mission_encoding,
            'direction': direction_onehot
        }

# Step 2: Define a custom features extractor for the policy
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.mission_mlp = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
        )
        # Compute the output dimension
        with torch.no_grad():
            sample_image = torch.zeros(1, 3, 7, 7)
            cnn_output_dim = self.cnn(sample_image).shape[1]
        self._features_dim = cnn_output_dim + 16 + 4  # +4 for direction

    def forward(self, observations):        
        # Convert to float and normalize
        image = observations['image'].float() / 255.0
        # Transpose from (batch_size, 7, 7, 3) to (batch_size, 3, 7, 7)
        # image = image.permute(0, 3, 1, 2)
        image_features = self.cnn(image)
        mission = observations['mission']
        mission_features = self.mission_mlp(mission)
        direction = observations['direction']
        return torch.cat([image_features, mission_features, direction], dim=1)



# Step 3: Create the environment and wrap it
import os
import yaml
from datetime import datetime
import gymnasium as gym
import minigrid
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy

# Assume CustomObsWrapper and CustomCombinedExtractor are defined as above

from utils import save_model, save_config
# Add the new import for evaluation
from stable_baselines3.common.evaluation import evaluate_policy

# ... (other existing imports like gym, PPO, datetime, os, etc., remain unchanged)
# ... (CustomObsWrapper and CustomCombinedExtractor definitions remain unchanged)
# ... (save_model and save_config functions remain unchanged if defined elsewhere)

def main():
    """Main function to set up, train, and evaluate the reinforcement learning agent."""
    # Create experiment directory with a timestamp
    experiment_dir = f"experiments/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Set up logger
    logger = configure(experiment_dir, ["stdout", "log"])
    
    # Create and wrap the environment
    env = gym.make('MiniGrid-Fetch-5x5-N2-v0')
    env = CustomObsWrapper(env)

    # train_env = gym.make('MiniGrid-Fetch-5x5-N2-v0')
    # train_env = CountBasedExplorationWrapper(train_env, coefficient=0.005)
    # env = CustomObsWrapper(train_env)
    
    # Define policy kwargs for the custom feature extractor
    policy_kwargs = {
        "features_extractor_class": CustomCombinedExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
    }
    
    # Initialize the PPO model
    model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    model.set_logger(logger)
    
    # Define configuration dictionaries
    env_config = {
        "env_id": "MiniGrid-Fetch-5x5-N2-v0",
    }
    model_config = {
        "algorithm": "PPO",
        "policy": "MultiInputPolicy",
        "policy_kwargs": policy_kwargs,
        "learning_rate": model.learning_rate,
        "n_steps": model.n_steps,
        "batch_size": model.batch_size,
        "n_epochs": model.n_epochs,
    }
    
    # Train the model
    model.learn(total_timesteps=800000)
    
    # Save the trained model
    save_model(model, experiment_dir)
    
    # Save the configurations
    save_config(env_config, model_config, experiment_dir)
    
    # Evaluate the agent over 100 episodes
    mean_reward, std_reward = evaluate_policy(
        model, 
        env, 
        n_eval_episodes=100, 
        deterministic=True
    )
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Run the main function when the script is executed directly
if __name__ == "__main__":
    main()