from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import PyTorchObs # <<< MOVED HERE

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from minigrid.wrappers import FullyObsWrapper
from minigrid.core.constants import COLOR_NAMES

import torch as th
import torch.nn as nn

import argparse
import os
import multiprocessing as mp


# Custom CNN for Minigrid's small image observations (from previous step)
class CustomMiniGridCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0), 
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            dummy_input = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(dummy_input).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

# Custom Combined Extractor
class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, cnn_features_dim: int = 64):
        super().__init__(observation_space, features_dim=1) 

        extractors = {}
        total_concat_size = 0

        for key, subspace in observation_space.spaces.items():
            if key == "image":
                extractors[key] = CustomMiniGridCNN(subspace, features_dim=cnn_features_dim)
                total_concat_size += cnn_features_dim
            elif isinstance(subspace, spaces.Discrete):
                extractors[key] = nn.Flatten()
                total_concat_size += subspace.n 
            elif isinstance(subspace, spaces.Box) and len(subspace.shape) == 1:
                extractors[key] = nn.Flatten()
                total_concat_size += int(np.prod(subspace.shape))
            else:
                raise ValueError(f"Unsupported subspace type for key {key}: {subspace}")

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

    def forward(self, observations: PyTorchObs) -> th.Tensor: # PyTorchObs is now defined
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return th.cat(encoded_tensor_list, dim=1)


# Custom Wrapper to Encode Mission Strings (from previous step)
class MissionStringEncoderWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.mission_syntaxes = [
            "get a", "go get a", "fetch a", "go fetch a", "you must fetch a"
        ]
        self.mission_colors = COLOR_NAMES 
        self.mission_types = ["key", "ball"] 

        self.possible_missions = []
        for syntax in self.mission_syntaxes:
            for color in self.mission_colors:
                for obj_type in self.mission_types:
                    self.possible_missions.append(f"{syntax} {color} {obj_type}")
        
        self.mission_to_idx = {mission: i for i, mission in enumerate(self.possible_missions)}
        self.num_possible_missions = len(self.possible_missions)
        
        original_obs_space_dict = env.observation_space.spaces
        new_mission_space = spaces.Discrete(self.num_possible_missions)
        
        self.observation_space = spaces.Dict({
            **original_obs_space_dict,
            'mission': new_mission_space
        })

    def observation(self, obs):
        mission_str = obs['mission']
        if mission_str not in self.mission_to_idx:
            obs['mission'] = 0 
        else:
            obs['mission'] = self.mission_to_idx[mission_str]
        return obs


def create_env_fn(env_name="MiniGrid-Fetch-5x5-N2-v0", render_mode=None):
    env = gym.make(env_name, render_mode=render_mode)
    env = FullyObsWrapper(env) 
    env = MissionStringEncoderWrapper(env)
    return env



def run(device: str, steps: int):
    env_id = "MiniGrid-Fetch-8x8-N3-v0"
    load_model_path = None
    # load_model_path = "ppo_minigrid_fetch5x5_multi_input_4M.zip"
    N_ENVS = 24
    vec_env = make_vec_env(lambda: create_env_fn(env_id), n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)

    ppo_hyperparams = {
        "learning_rate": 0.0001,
        "n_steps": 2048,        
        "batch_size": 128,       
        "n_epochs": 10,         
        "gamma": 0.99,          
        "gae_lambda": 0.95,     
        "ent_coef": 0.015,       
        "vf_coef": 0.5,         
        "max_grad_norm": 0.5,   
        "normalize_advantage": True,
    }
    
    policy_kwargs_ppo = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(
            cnn_features_dim=64 
        ),
        net_arch=dict(pi=[128, 128], vf=[128, 128]) 
    )

    if load_model_path:
        print(f"Loading model from {load_model_path} to continue training up to {steps} steps...")
        model = PPO.load(
            load_model_path,
            env=vec_env,
            device=device,
            learning_rate=ppo_hyperparams["learning_rate"]
        )
        # Optionally override lr schedule if needed
        # model.learning_rate = ppo_hyperparams["learning_rate"]
        model.batch_size = ppo_hyperparams["batch_size"]
        model.n_steps = ppo_hyperparams["n_steps"]
        reset_num_timesteps=False
    else:
        model = PPO(
            "MultiInputPolicy",
            vec_env,
            policy_kwargs=policy_kwargs_ppo,
            device=device,
            verbose=1,
            **ppo_hyperparams
        )
        reset_num_timesteps=True

    print(f"Starting training PPO with MultiInputPolicy on {env_id}...")
    model.learn(
        total_timesteps=steps,
        reset_num_timesteps=reset_num_timesteps,
        log_interval=1,
    )
    # model.learn(total_timesteps=steps, reset_num_timesteps=reset_num_timesteps) 
    print("Training complete. Evaluating the agent...")

    eval_env = create_env_fn(env_id) 
    n_eval_episodes = 100
    print("Evaluating the agent on training environment...")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=n_eval_episodes, deterministic=True)
    print(f"Evaluation results: Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    eval_env.close()

    model.save("ppo_minigrid_fetch8x8_multi_input_4M")
    print("Training complete. Model saved.")

    vec_env.close()            
    del model, vec_env



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--steps", type=int, default=4_000_000)
    parser.add_argument("--load-model", type=str, default=None, help="Path to saved model zip to resume training")
    args = parser.parse_args()
    run(args.device, args.steps)