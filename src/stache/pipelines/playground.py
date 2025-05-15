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


# This is the CustomMiniGridCNN from your original script, now with added print statements
class CustomMiniGridCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        # print(f"  [CustomMiniGridCNN.__init__] observation_space shape: {observation_space.shape}, features_dim: {features_dim}")
        
        # Assuming observation_space.shape is (C, H, W) due to VecTransposeImage
        n_input_channels = observation_space.shape[0]
        # print(f"    n_input_channels: {n_input_channels}")
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0), 
            nn.ReLU(),
            nn.Flatten(), # Default start_dim=1, flattens (B, C_out, H_out, W_out) to (B, C_out*H_out*W_out)
        )

        with th.no_grad():
            dummy_obs_sample = observation_space.sample() # Expected (C, H, W)
            dummy_input = th.as_tensor(dummy_obs_sample[None]).float() # Adds batch dim -> (1, C, H, W)
            # print(f"    CustomMiniGridCNN dummy_input for n_flatten calc: shape {dummy_input.shape}")
            
            n_flatten = self.cnn(dummy_input).shape[1] # Output of cnn is (1, n_flatten)
            # print(f"    Calculated n_flatten: {n_flatten}")

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
        # print(f"  [CustomMiniGridCNN.__init__] CNN module: {self.cnn}")
        # print(f"  [CustomMiniGridCNN.__init__] Linear module: {self.linear}")

    def forward(self, observations: th.Tensor) -> th.Tensor:
        print(f"    [CustomMiniGridCNN.forward] Input 'observations' (image): shape {observations.shape}, dtype {observations.dtype}")
        cnn_features = self.cnn(observations) # Expected output: (batch_size, n_flatten)
        print(f"    [CustomMiniGridCNN.forward] Output of self.cnn (after flatten): shape {cnn_features.shape}, dtype {cnn_features.dtype}")
        linear_output = self.linear(cnn_features) # Expected output: (batch_size, features_dim)
        print(f"    [CustomMiniGridCNN.forward] Output of self.linear: shape {linear_output.shape}, dtype {linear_output.dtype}")
        return linear_output

# This is the OneHotEncoder from your previous version, with added print statements
# Note: Not actively used by CustomCombinedExtractor for 'direction' or 'mission' processing
# as SB3 handles one-hot encoding for Discrete spaces before they reach .forward(),
# and 'mission' uses nn.Embedding.
class OneHotEncoder(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x: th.Tensor) -> th.Tensor:
        # print(f"      [OneHotEncoder.forward] Input 'x': shape {x.shape}, dtype {x.dtype}")
        original_shape = x.shape
        if x.ndim == 2 and x.shape[-1] == 1:
            x = x.squeeze(-1)
            # print(f"        OneHotEncoder after squeeze: shape {x.shape}")
        
        if not (x.dtype == th.int64 or x.dtype == th.int32):
             x = x.long()
            # print(f"        OneHotEncoder after .long(): dtype {x.dtype}")
        
        one_hot_output = th.nn.functional.one_hot(x, num_classes=self.num_classes).float()
        # print(f"      [OneHotEncoder.forward] Output: shape {one_hot_output.shape}, dtype {one_hot_output.dtype}")
        return one_hot_output

# MODIFIED CustomCombinedExtractor
class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, cnn_features_dim: int = 64, mission_embedding_dim: int = 16):
        super().__init__(observation_space, features_dim=1) # Placeholder, self._features_dim is set below
        # print(f"[CustomCombinedExtractor.__init__] Initializing...")
        # print(f"  Input observation_space: {observation_space}")
        # SB3 passes the original (pre-one-hotting) observation space to this __init__.

        extractors = {}
        total_concat_size = 0

        for key, subspace in observation_space.spaces.items():
            # print(f"  Processing key: '{key}', subspace: {subspace} (type: {type(subspace)})")
            if key == "image":
                # print(f"    Initializing CustomMiniGridCNN for '{key}' with features_dim={cnn_features_dim}")
                extractors[key] = CustomMiniGridCNN(subspace, features_dim=cnn_features_dim)
                total_concat_size += cnn_features_dim
            elif key == "mission": # This subspace is spaces.Discrete from MissionStringEncoderWrapper
                if not isinstance(subspace, spaces.Discrete):
                    raise ValueError(f"Mission subspace '{key}' must be of type spaces.Discrete for nn.Embedding, got {type(subspace)}")
                # print(f"    Initializing nn.Embedding for '{key}' with num_embeddings={subspace.n}, embedding_dim={mission_embedding_dim}")
                extractors[key] = nn.Embedding(num_embeddings=subspace.n, embedding_dim=mission_embedding_dim)
                total_concat_size += mission_embedding_dim
            elif isinstance(subspace, spaces.Discrete): # This handles 'direction' which is spaces.Discrete(4)
                # print(f"    Initializing nn.Identity for discrete key '{key}' (obs in forward() will be one-hot) with num_features={subspace.n}")
                extractors[key] = nn.Identity() 
                total_concat_size += subspace.n # The size of the one-hot encoded vector
            elif isinstance(subspace, spaces.Box) and len(subspace.shape) == 1: 
                # print(f"    Initializing nn.Flatten for 1D Box key '{key}' with shape {subspace.shape}")
                extractors[key] = nn.Flatten() # Or nn.Identity if it's already flat (N,)
                total_concat_size += int(np.prod(subspace.shape))
            else:
                raise ValueError(f"Unsupported subspace type for key '{key}': {subspace} with shape {subspace.shape}")
            # print(f"    Current total_concat_size: {total_concat_size}")

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size
        # print(f"[CustomCombinedExtractor.__init__] Final _features_dim: {self._features_dim}")
        # print(f"  Extractors dict: {self.extractors}")

    def forward(self, observations: PyTorchObs) -> th.Tensor:
        print("\n[CustomCombinedExtractor.forward] Called.")
        # print(f"  Input observations type: {type(observations)}")
        # for key, val_tensor in observations.items():
        #     print(f"  observations['{key}']: shape {val_tensor.shape}, dtype {val_tensor.dtype}, device {val_tensor.device}")

        encoded_tensor_list = []
        print("  Iterating through extractors to process observation components:")
        for key, extractor in self.extractors.items():
            print(f"  Processing observation component: '{key}' using {type(extractor).__name__}")
            observation_data = observations[key]
            print(f"    Input data for '{key}': shape {observation_data.shape}, dtype {observation_data.dtype}")

            processed_observation_data = observation_data 

            if key == "mission":
                # Input 'observation_data' for mission is one-hot encoded by SB3.
                # Shape can be (batch_size, num_mission_classes) or (batch_size, 1, num_mission_classes) during PPO.train().
                # nn.Embedding expects indices of dtype int64.
                # We take argmax over the last dimension (the one-hot classes).
                indices = th.argmax(observation_data, dim=-1) # Shape: (batch_size,) or (batch_size, 1)
                print(f"      '{key}' (mission) after argmax to get indices: shape {indices.shape}, dtype {indices.dtype}")
                # indices will be LongTensor, which is required by nn.Embedding.
                processed_observation_data = indices
            
            elif key == "image": 
                 # VecTransposeImage ensures CHW. Conv2D expects float.
                if processed_observation_data.dtype != th.float32:
                    # MiniGrid image values (obj indices, color indices, states) are small integers.
                    # Not scaling by /255.0 as they are not pixel values in 0-255 range.
                    # Just convert to float for the CNN.
                    processed_observation_data = processed_observation_data.float()
            
            # For 'direction': input 'observation_data' is already one-hot encoded by SB3,
            # e.g., (batch_size, 4) or (batch_size, 1, 4).
            # The nn.Identity extractor will pass it through with the same shape.
            
            extracted_features = extractor(processed_observation_data)
            print(f"    Output features from '{key}' extractor ({type(extractor).__name__}): shape {extracted_features.shape}, dtype {extracted_features.dtype}")

            # Ensure features are 2D (Batch, Features) for concatenation.
            # This handles cases where input obs had an extra time dim of 1, e.g. (B, 1, F),
            # or if an extractor (like nn.Embedding with (B,1) input indices) produces (B, 1, F_emb).
            if extracted_features.ndim == 3 and extracted_features.shape[1] == 1:
                print(f"      Squeezing feature tensor for '{key}' from {extracted_features.shape} to 2D.")
                extracted_features = extracted_features.squeeze(1)
            
            # Robustness: If batch_size was 1 and a squeeze made it 1D (e.g. (F,)), unsqueeze to (1,F)
            # This is less likely for typical batch_sizes > 1 but good for completeness.
            if extracted_features.ndim == 1:
                # Check original batch dimension from any observation component (e.g., 'image' usually has batch dim)
                # A simpler check might be if any of observations[key].shape[0] was 1.
                # For safety, if any observation had batch_size 1, assume current 1D tensor means (Features,) for that single sample.
                # This requires knowing the batch size. We can infer it from observations[key].shape[0] if ndim > 0.
                # Let's assume if ndim is 1, it's for a single sample that needs to be (1, Features).
                # This is more likely if the overall batch_size being processed by .forward() is 1.
                # For this specific problem, batch_size is 8 or 64, so ndim==1 for extracted_features is unlikely to be desired.
                # The only way it could be 1D is if Embedding output was (embedding_dim,) for a scalar index.
                # `th.argmax` on `(num_classes,)` gives scalar. `nn.Embedding` on scalar index gives `(embedding_dim,)`.
                # This case should be avoided by ensuring `observation_data` always has a batch dimension.
                # SB3 usually ensures this.
                # If `observations[key].shape[0]` (batch dim) was 1 and `extracted_features.ndim == 1`, then:
                # This check is somewhat heuristic without knowing if forward() is processing a single instance or a batch.
                # However, SB3's PPO calls forward with batches.
                pass # Current squeeze logic should make it 2D if it was (B,1,F) or (B,F)

            # Final check, all features must be 2D for concatenation
            if extracted_features.ndim != 2:
                raise ValueError(
                    f"Extractor for key '{key}' ({type(extractor).__name__}) produced features with ndim={extracted_features.ndim} "
                    f"(shape {extracted_features.shape}), after processing and squeezing. "
                    f"Expected 2D (Batch, Features) for concatenation."
                )

            encoded_tensor_list.append(extracted_features)

        print("  --- Tensor details before concatenation ---")
        if not encoded_tensor_list: # This case should ideally not be reached with valid observations
            print("  WARNING: encoded_tensor_list is empty!")
            batch_size_if_known = 0
            if observations and isinstance(observations, dict) and observations.keys():
                first_key = list(observations.keys())[0]
                if hasattr(observations[first_key], 'shape') and len(observations[first_key].shape) > 0:
                    batch_size_if_known = observations[first_key].shape[0]
            
            device_to_use = 'cpu' # Default device
            if self.extractors: # Try to infer device from the first available extractor's parameters
                first_extractor_key = list(self.extractors.keys())[0]
                try:
                    device_to_use = next(self.extractors[first_extractor_key].parameters()).device
                except StopIteration: # Extractor has no parameters (e.g., nn.Identity if not yet moved to device)
                     if hasattr(self.extractors[first_extractor_key], '_features_dim'): # A rough check if it's an SB3 BaseFeaturesExtractor
                        # SB3 extractors might not have parameters directly but are on a device.
                        # This is hard to get reliably without access to self.device from PPO.
                        pass # Stick to cpu default if unsure

            return th.empty((batch_size_if_known, 0), device=device_to_use)


        for i, t in enumerate(encoded_tensor_list):
            key_name = list(self.extractors.keys())[i]
            print(f"  Tensor for '{key_name}': shape {t.shape}, ndim {t.ndim}, dtype {t.dtype}")

        print(f"  Attempting to concatenate {len(encoded_tensor_list)} tensors along dim=1.")
        try:
            concatenated_output = th.cat(encoded_tensor_list, dim=1)
        except RuntimeError as e:
            print(f"  !!! RuntimeError during th.cat: {e} !!!")
            # Re-print details at point of error for clarity
            print(f"  Tensor details at point of th.cat failure:")
            for j, tensor_in_list in enumerate(encoded_tensor_list):
                k_name = list(self.extractors.keys())[j]
                print(f"    {j}. '{k_name}': shape {tensor_in_list.shape}, ndim {tensor_in_list.ndim}, dtype {tensor_in_list.dtype}")
            raise e 
            
        print(f"  Output of th.cat: shape {concatenated_output.shape}, dtype {concatenated_output.dtype}")
        print("[CustomCombinedExtractor.forward] Finished successfully.")
        return concatenated_output

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
        # Ensure 'mission' key exists in original_obs_space_dict for replacement
        if 'mission' not in original_obs_space_dict:
            # This case should ideally not happen if FullyObsWrapper is applied to a MiniGrid env
            # that has a mission string.
            pass

        new_mission_space = spaces.Discrete(self.num_possible_missions)
        
        self.observation_space = spaces.Dict({
            **original_obs_space_dict,
            'mission': new_mission_space
        })
        # print(f"MissionStringEncoderWrapper: new obs_space['mission']: {self.observation_space['mission']}")


    def observation(self, obs):
        mission_str = obs['mission']
        if mission_str not in self.mission_to_idx:
            # Default to index 0 for unknown missions.
            print(f"Warning: Unknown mission string '{mission_str}' encountered in MissionStringEncoderWrapper. Defaulting to index 0.")
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
    env_id = "MiniGrid-Fetch-5x5-N2-v0"
    N_ENVS = 8 # Or 1 for simpler debugging of shapes initially
    vec_env = make_vec_env(lambda: create_env_fn(env_id), n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)

    # print(f"Wrapped VecEnv observation space: {vec_env.observation_space}")
    # print(f"  VecEnv obs_space['image']: {vec_env.observation_space['image']}") # Should be (C,H,W) after VecTransposeImage
    # print(f"  VecEnv obs_space['direction']: {vec_env.observation_space['direction']}") # Should be Discrete(4)
    # print(f"  VecEnv obs_space['mission']: {vec_env.observation_space['mission']}") # Should be Discrete(60)


    ppo_hyperparams = {
        "learning_rate": 0.0003,
        "n_steps": 2048, # SB3 default for PPO. Rollout buffer size per env.       
        "batch_size": 64, # SB3 default. Minibatch size for PPO updates.     
        "n_epochs": 10, # SB3 default. Number of epochs when optimizing the surrogate loss.      
        "gamma": 0.99,          
        "gae_lambda": 0.95,     
        "ent_coef": 0.015, # Adjusted from 0.01
        "vf_coef": 0.5,         
        "max_grad_norm": 0.5,   
        "normalize_advantage": True,
    }
    
    # MODIFIED policy_kwargs_ppo
    policy_kwargs_ppo = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(
            cnn_features_dim=64,
            mission_embedding_dim=16 # Added parameter for mission embedding dimension
        ),
        net_arch=dict(pi=[128, 128], vf=[128, 128]) 
    )

    model = PPO(
        "MultiInputPolicy", 
        vec_env,
        policy_kwargs=policy_kwargs_ppo,
        device=device,
        verbose=1,
        **ppo_hyperparams
    )

    print(f"Starting training PPO with MultiInputPolicy on {env_id}...")
    model.learn(total_timesteps=steps) 

    eval_env = create_env_fn(env_id) 
    n_eval_episodes = 100
    print("Evaluating the agent on training environment...")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=n_eval_episodes, deterministic=True)
    print(f"Evaluation results: Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    eval_env.close()

    # model.save("ppo_minigrid_fetch5x5_multi_input_embedded_mission") 
    # print("Training complete. Model saved.")

    vec_env.close()            
    del model, vec_env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--steps", type=int, default=20_000) # Reduced for quicker testing if needed
    args = parser.parse_args()
    # It's good practice for SubprocVecEnv to be under if __name__ == "__main__":
    # because it uses multiprocessing.
    mp.set_start_method("spawn", force=True) # Recommended for multiprocessing with PyTorch on some systems
    run(args.device, args.steps)