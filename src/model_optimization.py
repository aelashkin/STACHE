from typing import Any, Dict
import optuna
import optunahub
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3 import A2C
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

N_TRIALS = 50  # Total number of Optuna trials for hyperparameter optimization
N_STARTUP_TRIALS = 5  # Initial random trials before optimization logic is applied
N_EVALUATIONS = 2  # Number of evaluations during each training trial
N_TIMESTEPS = 300000  # Total timesteps for training the agent in each trial
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)  # Timesteps between each evaluation
N_EVAL_EPISODES = 50  # Number of episodes to average performance during evaluation


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

    # net_arch = [
    #     {"pi": [64], "vf": [64]} if net_arch == "tiny" else {"pi": [64, 64], "vf": [64, 64]}
    # ]
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
        env_config = load_config("config/training_config_env.yml")
        model_config = load_config("config/training_config_A2C.yml")
    except (FileNotFoundError, yaml.YAMLError) as e:
        raise RuntimeError(f"Configuration loading failed: {e}")

    device = get_device(model_config.get("device", "cpu"))

    env_name = env_config["env_name"]
    env = gym.make(env_name)
    env = FullyObsWrapper(env)
    env = FactorizedSymbolicWrapper(env)
    env = PaddedObservationWrapper(env, max_objects=env_config["max_objects"], max_walls=env_config["max_walls"])
    env = Monitor(env)

    kwargs = sample_a2c_params(trial)
    kwargs["env"] = env
    kwargs["policy"] = "MlpPolicy"
    kwargs["device"] = device

    model = A2C(**kwargs)
    eval_env = Monitor(env)
    eval_callback = TrialEvalCallback(
        eval_env, trial, n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_FREQ, deterministic=True
    )

    nan_encountered = False
    try:
        model.learn(N_TIMESTEPS, callback=eval_callback)
    except AssertionError as e:
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

    study_name = f"A2C_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    study = optuna.create_study(study_name=study_name, sampler=sampler, pruner=pruner, direction="maximize")
    print("Starting optimization...")
    try:
        study.optimize(objective, n_trials=N_TRIALS, timeout=600, n_jobs=12)
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
