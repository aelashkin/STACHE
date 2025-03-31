import sys
import os

# Get the absolute path of the project's root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))

# Ensure the project root is in sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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
from src.minigrid.environment_utils import create_minigrid_env, create_symbolic_minigrid_env
from src.hyperparameter_utils import sample_a2c_params, sample_ppo_params, TrialEvalCallback
from utils import save_config


from optuna.study import MaxTrialsCallback

N_TRIALS = 70  # Total number of Optuna trials for hyperparameter optimization
N_STARTUP_TRIALS = 5  # Initial random trials before optimization logic is applied
N_EVALUATIONS = 2  # Number of evaluations during each training trial
N_TIMESTEPS = 50000  # Total timesteps for training the agent in each trial
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)  # Timesteps between each evaluation
N_EVAL_EPISODES = 50  # Number of episodes to average performance during evaluation

# Select which model type to optimize
MODEL_TYPE = ModelType.PPO
MULTICORE = True


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
        model.learn(total_timesteps=N_TIMESTEPS, callback=eval_callback)
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


    #Check on the best way to define study name with date here
    if MODEL_TYPE == ModelType.PPO:
        study_name = f"PPO_4"
    elif MODEL_TYPE == ModelType.A2C:
        study_name = f"A2C_2"

    # Dynamically determine the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the relative path to the database file
    db_path = os.path.join(script_dir, "../config/optuna_study.db")
    storage = f"sqlite:///{os.path.abspath(db_path)}"

    study = optuna.create_study(
            study_name=study_name, 
            sampler=sampler, 
            pruner=pruner, 
            direction="maximize", 
            storage=storage,
            load_if_exists=True
    )

    # study_name = f"A2C_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    # study = optuna.create_study(study_name=study_name, sampler=sampler, pruner=pruner, direction="maximize")
    
    
    print("Starting optimization...")
    # the study will stop when all processes reach N_TRIALS trials jointly
    try:
        study.optimize(objective, 
                       callbacks=[MaxTrialsCallback(N_TRIALS, states=None)],
                       n_jobs=1, 
                       show_progress_bar=True)
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
    
    # Save best parameters to file
    env_config = load_config("config/training_config_env.yml")
    env_name = env_config.get("env_name", "unknown_env")
    
    # Create directory structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = "PPO" if MODEL_TYPE == ModelType.PPO else "A2C"
    experiment_folder_name = f"{env_name}_{model_name}_model_{timestamp}"
    experiment_base_dir = "data/experiments/optuna"
    experiment_dir = os.path.join(project_root, experiment_base_dir, experiment_folder_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Create model config from best parameters
    model_config = {
        "model_type": model_name,
        **trial.params
    }
    # Add user attributes to model config
    for key, value in trial.user_attrs.items():
        model_config[key] = value
    
    # Save config
    config_path = save_config(env_config, model_config, experiment_dir)
    print(f"Best parameters saved at: {config_path}")
    
    # Create optimization log content
    optimization_log = [
        f"Optimization completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Study name: {study_name}",
        f"Number of finished trials: {len(study.trials)}",
        f"Best trial value: {trial.value}",
        "\nBest parameters:"
    ]
    for key, value in trial.params.items():
        optimization_log.append(f"  {key}: {value}")
    optimization_log.append("\nUser attributes:")
    for key, value in trial.user_attrs.items():
        optimization_log.append(f"  {key}: {value}")
    
    # Save optimization log
    log_path = os.path.join(experiment_dir, "optimization.log")
    with open(log_path, "w") as file:
        file.write("\n".join(optimization_log))
    print(f"Optimization log saved at: {log_path}")
