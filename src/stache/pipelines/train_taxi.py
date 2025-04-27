import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
# Import the saving utility
from utils.experiment_io import save_experiment

# ───────────────────────────────────────────────────────────────────────────────
# SOLVES TAXI-v3
# ─────────────────────────────────────────────────────────────────────────────

# 1. One‑hot wrapper for Discrete(500) → Box(500)
class OneHotObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Discrete), \
            "OneHotObs only supports Discrete spaces"
        n = env.observation_space.n
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(n,), dtype=np.float32
        )

    def observation(self, obs):
        vec = np.zeros(self.observation_space.shape, dtype=np.float32)
        vec[obs] = 1.0
        return vec

# 2. Env factory for vectorization
def make_env():
    env = gym.make("Taxi-v3")
    return OneHotObs(env)

# 3. Build 8 parallel environments
n_envs = 8
train_env = DummyVecEnv([make_env for _ in range(n_envs)])

# 4. Instantiate vanilla DQN
model = DQN(
    policy="MlpPolicy",
    env=train_env,
    learning_rate=1e-4,            # lower LR for stability
    buffer_size=500_000,           # large enough to cover 2M steps
    learning_starts=10_000,        # let the buffer fill a bit first
    batch_size=128,
    gamma=0.99,
    train_freq=(1, "step"),        # update once every collected step
    gradient_steps=1,
    target_update_interval=10_000,
    exploration_initial_eps=1.0,
    exploration_fraction=0.2,      # decay ε from 1.0 to 0.02 over 20% of training
    exploration_final_eps=0.02,
    policy_kwargs=dict(net_arch=[256, 256]),
    verbose=1,
    seed=42,
)

# 5. Train for 2 million timesteps
total_timesteps_to_train = 1_200_000
model.learn(total_timesteps=total_timesteps_to_train)

# 6. Evaluate on a fresh, monitored env
eval_env = Monitor(OneHotObs(gym.make("Taxi-v3")))
n_eval_episodes = 100
print("Evaluating the agent on training environment...")
mean_reward, std_reward = evaluate_policy(
    model,
    eval_env,
    n_eval_episodes=n_eval_episodes,
    deterministic=True,
)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

# 7. Prepare configurations and log for saving
env_config = {
    "env_name": "Taxi-v3",
    "n_envs": n_envs,
    "wrapper": "OneHotObs"
}

model_config = {
    "model_type": "DQN", # Note: load_experiment might need adjustment for DQN
    "policy": "MlpPolicy",
    "learning_rate": 1e-4,
    "buffer_size": 500_000,
    "learning_starts": 10_000,
    "batch_size": 128,
    "gamma": 0.99,
    "train_freq": (1, "step"),
    "gradient_steps": 1,
    "target_update_interval": 10_000,
    "exploration_initial_eps": 1.0,
    "exploration_fraction": 0.2,
    "exploration_final_eps": 0.02,
    "policy_kwargs": dict(net_arch=[256, 256]),
    "seed": 42,
    "total_timesteps": total_timesteps_to_train
}

training_log = (
    f"Training completed for {total_timesteps_to_train} timesteps.\n"
    f"Evaluation over {n_eval_episodes} episodes:\n"
    f"Mean Reward: {mean_reward:.2f}, Std Reward: {std_reward:.2f}\n"
)

# 8. Save the experiment
print("\nSaving the experiment...")
experiment_info = save_experiment(
    model=model,
    env_config=env_config,
    model_config=model_config,
    training_log=training_log,
    # experiments_base_dir="data/experiments/models" # Default location
)
print("Experiment saved successfully:")
print(experiment_info)
