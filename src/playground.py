import gymnasium as gym
import minigrid
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch
from gymnasium.wrappers import FlattenObservation
from minigrid.core.mission import MissionSpace
from minigrid.wrappers import ImgObsWrapper

# Step 1: Define a custom wrapper to handle the observation space
class CustomObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(CustomObsWrapper, self).__init__(env)
        # Assuming the mission space is a string, we'll need to process it
        self.observation_space = gym.spaces.Dict({
            'image': env.observation_space['image'],
            'mission': gym.spaces.Box(low=0, high=1, shape=(8,))  # Example: one-hot encoding for mission
        })

    def observation(self, obs):
        # Process the mission string into a numerical representation
        mission_str = obs['mission']
        # Simple example: one-hot encode the last two words (color and type)
        words = mission_str.split()
        color = words[-2]  # Assuming the color is the second last word
        obj_type = words[-1]  # Assuming the type is the last word
        # Define possible colors and types
        colors = ['red', 'green', 'blue', 'purple', 'yellow', 'grey']
        types = ['key', 'ball']
        color_idx = colors.index(color)
        type_idx = types.index(obj_type)
        mission_encoding = torch.zeros(8)  # 6 for colors, 2 for types
        mission_encoding[color_idx] = 1
        mission_encoding[6 + type_idx] = 1
        return {'image': obs['image'], 'mission': mission_encoding}

# Step 2: Define a custom features extractor for the policy
class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim)
        # Define extractors for image and mission
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
            sample_image = torch.zeros(1, 3, 7, 7)  # Assuming 7x7 image
            cnn_output_dim = self.cnn(sample_image).shape[1]
        self._features_dim = cnn_output_dim + 16  # From mission MLP

    def forward(self, observations):
        image = observations['image']
        # print("Image shape:", image.shape)  # Should be [1, 3, 7, 7]
        image_features = self.cnn(image)  # Pass directly to CNN
        mission = observations['mission']
        mission_features = self.mission_mlp(mission)
        return torch.cat([image_features, mission_features], dim=1)

# Step 3: Create the environment and wrap it
env = gym.make('MiniGrid-Fetch-5x5-N2-v0')
env = CustomObsWrapper(env)

# Step 4: Define the policy kwargs with the custom features extractor
policy_kwargs = {
    "features_extractor_class": CustomCombinedExtractor,
    "features_extractor_kwargs": {"features_dim": 256},
}

# Step 5: Create and train the PPO agent
model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
model.learn(total_timesteps=100000)

# Step 6: Save the model
model.save("ppo_minigrid_fetch")

# Step 7: Evaluate the agent (optional)
obs = env.reset()
for _ in range(100):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        obs = env.reset()