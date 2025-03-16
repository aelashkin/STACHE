import optuna
import optunahub
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.monitor import Monitor

import torch

import yaml
from datetime import datetime

from src.utils import load_config, get_device, ModelType
from src.environment_utils import create_symbolic_minigrid_env, create_minigrid_env
from src.hyperparameter_utils import sample_a2c_params, sample_ppo_params, TrialEvalCallback


N_TRIALS = 10  # Total number of Optuna trials for hyperparameter optimization
N_STARTUP_TRIALS = 5  # Initial random trials before optimization logic is applied
N_EVALUATIONS = 2  # Number of evaluations during each training trial
N_TIMESTEPS = 10000  # Total timesteps for training the agent in each trial
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)  # Timesteps between each evaluation
N_EVAL_EPISODES = 50  # Number of episodes to average performance during evaluation

# Select which model type to optimize
MODEL_TYPE = ModelType.A2C
MULTICORE = False


# def objective(trial: optuna.Trial) -> float:
#     try:
#         if MODEL_TYPE == ModelType.A2C:
#             config_file = "config/training_config_A2C.yml"
#             sample_params_func = sample_a2c_params
#             ModelClass = A2C
#         else:
#             config_file = "config/training_config_PPO.yml"
#             sample_params_func = sample_ppo_params
#             ModelClass = PPO

#         env_config = load_config("config/training_config_env.yml")
#         model_config = load_config(config_file)
#     except (FileNotFoundError, yaml.YAMLError) as e:
#         raise RuntimeError(f"Configuration loading failed: {e}")

#     device = get_device(model_config.get("device", "cpu"))
#     env = create_symbolic_minigrid_env(env_config)

#     kwargs = sample_params_func(trial)
#     kwargs["env"] = env
#     kwargs["policy"] = "MlpPolicy"
#     kwargs["device"] = device

#     model = ModelClass(**kwargs)
#     eval_env = Monitor(env)
#     eval_callback = TrialEvalCallback(
#         eval_env, trial, n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_FREQ, deterministic=True
#     )

#     nan_encountered = False
#     try:
#         model.learn(total_timesteps=model_config["total_timesteps"], callback=eval_callback)
#     except Exception as e:
#         print(e)
#         nan_encountered = True
#     finally:
#         model.env.close()
#         eval_env.close()

#     if nan_encountered:
#         return float("nan")
#     if eval_callback.is_pruned:
#         raise optuna.exceptions.TrialPruned()

#     return eval_callback.last_mean_reward

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

    # Return the best evaluation reward observed (the early stopping point)
    return eval_callback.best_mean_reward



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
