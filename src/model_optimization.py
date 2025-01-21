from typing import Any, Dict
import optuna
import optunahub
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from datetime import datetime
import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper
from src.wrappers import FactorizedSymbolicWrapper, PaddedObservationWrapper
from src.utils import save_logs, save_model, evaluate_agent, load_config, get_device
import torch
import torch.nn as nn
import yaml
from enum import Enum

N_TRIALS = 50  # Total number of Optuna trials for hyperparameter optimization
N_STARTUP_TRIALS = 5  # Initial random trials before optimization logic is applied
N_EVALUATIONS = 2  # Number of evaluations during each training trial
N_TIMESTEPS = 10000  # Total timesteps for training the agent in each trial
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)  # Timesteps between each evaluation
N_EVAL_EPISODES = 50  # Number of episodes to average performance during evaluation


class ModelType(Enum):
    A2C = "A2C"
    PPO = "PPO"

# Select which model type to optimize
MODEL_TYPE = ModelType.PPO


def sample_a2c_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for A2C hyperparameters."""
    gamma = 1.0 - trial.suggest_float("gamma", 0.0001, 0.1, log=True)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 5.0, log=True)
    gae_lambda = 1.0 - trial.suggest_float("gae_lambda", 0.001, 0.2, log=True)
    n_steps = 2 ** trial.suggest_int("exponent_n_steps", 3, 10)
    learning_rate = trial.suggest_float("lr", 1e-5, 1, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    ortho_init = trial.suggest_categorical("ortho_init", [False, True])
    net_arch = trial.suggest_categorical("net_arch", ["tiny", "small"])
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    trial.set_user_attr("gamma_", gamma)
    trial.set_user_attr("gae_lambda_", gae_lambda)
    trial.set_user_attr("n_steps", n_steps)

    net_arch = {"pi": [64], "vf": [64]} if net_arch == "tiny" else {"pi": [64, 64], "vf": [64, 64]}
    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn]

    return {
        "n_steps": n_steps,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "max_grad_norm": max_grad_norm,
        "policy_kwargs": {
            "net_arch": net_arch,
            "activation_fn": activation_fn,
            "ortho_init": ortho_init,
        },
    }

def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for PPO hyperparameters."""
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)

    n_steps = 2 ** trial.suggest_int("exponent_n_steps", 3, 10)
    # Filter batch_size options to ensure compatibility with n_steps
    valid_batch_sizes = [b for b in [32, 64, 128, 256] if n_steps % b == 0]
    if not valid_batch_sizes:
        raise ValueError(f"No valid batch_size for n_steps={n_steps}. Adjust ranges.")
    
    # Sample compatible batch_size
    batch_size = trial.suggest_categorical("batch_size", valid_batch_sizes)

    n_epochs = trial.suggest_int("n_epochs", 3, 12)

    gamma = 1.0 - trial.suggest_float("gamma", 0.0001, 0.2, log=True)
    gae_lambda = 1.0 - trial.suggest_float("gae_lambda", 0.05, 0.2, log=True) #екн keep 0.95 as default

    ent_coef = trial.suggest_float("ent_coef", 1e-8, 1e-2, log=True)
    vf_coef = trial.suggest_float("vf_coef", 0.1, 1.0)

    #check on these

    trial.set_user_attr("n_steps", n_steps)    
    trial.set_user_attr("gamma_", gamma)
    trial.set_user_attr("gae_lambda_", gae_lambda)


    return {
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
    }


def create_minigrid_env(env_config):
    env_name = env_config["env_name"]
    env = gym.make(env_name)
    env = FullyObsWrapper(env)
    env = FactorizedSymbolicWrapper(env)
    env = PaddedObservationWrapper(env, max_objects=env_config["max_objects"], max_walls=env_config["max_walls"])
    env = Monitor(env)
    return env


class TrialEvalCallback(EvalCallback):
    def __init__(
        self,
        eval_env: gym.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


def objective(trial: optuna.Trial) -> float:
    try:
        if MODEL_TYPE == ModelType.A2C:
            config_file = "config/training_config_A2C.yml"
            sample_params_func = sample_a2c_params
            ModelClass = A2C
        else:
            config_file = "config/training_config_PPO.yml"
            sample_params_func = sample_ppo_params
            ModelClass = PPO

        env_config = load_config("config/training_config_env.yml")
        model_config = load_config(config_file)
    except (FileNotFoundError, yaml.YAMLError) as e:
        raise RuntimeError(f"Configuration loading failed: {e}")

    device = get_device(model_config.get("device", "cpu"))
    env = create_minigrid_env(env_config)
    kwargs = sample_params_func(trial)
    kwargs["env"] = env
    kwargs["policy"] = "MlpPolicy"
    kwargs["device"] = device

    model = ModelClass(**kwargs)
    eval_env = Monitor(env)
    eval_callback = TrialEvalCallback(
        eval_env, trial, n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_FREQ, deterministic=True
    )

    nan_encountered = False
    try:
        model.learn(total_timesteps=model_config["total_timesteps"], callback=eval_callback)
    except Exception as e:
        print(e)
        nan_encountered = True
    finally:
        model.env.close()
        eval_env.close()

    if nan_encountered:
        return float("nan")
    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.last_mean_reward


if __name__ == "__main__":
    # Set pytorch num threads to 1 for faster training.
    torch.set_num_threads(1)

    sampler=optunahub.load_module("samplers/auto_sampler").AutoSampler()
    # sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    print(f"Sampler: {sampler} created successfully.")

    # Do not prune before 1/2 of the max budget is used.
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 2)
    print(f"Pruner: {pruner} created successfully.")

    study_name = f"{MODEL_TYPE.value}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    study = optuna.create_study(study_name=study_name, sampler=sampler, pruner=pruner, direction="maximize")
    print("Starting optimization...")
    try:
        study.optimize(objective, n_trials=N_TRIALS, timeout=600, n_jobs=1, show_progress_bar=True)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print(f"    {key}: {value}")
