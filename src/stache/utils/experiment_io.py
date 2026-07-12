import os
from datetime import datetime
from enum import Enum
from pathlib import Path

import yaml
import torch
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stache.utils.safe_yaml import SafeInputError, read_bounded_regular_text


_LEGACY_TUPLE_TAG = "tag:yaml.org,2002:python/tuple"
EXPERIMENT_CONFIG_MAX_BYTES = 1024 * 1024


class _LegacyExperimentConfigLoader(yaml.SafeLoader):
    """SafeLoader extended only for STACHE's historical tuple encoding."""


def _construct_legacy_tuple(
    loader: _LegacyExperimentConfigLoader,
    node: yaml.nodes.SequenceNode,
) -> tuple:
    return tuple(loader.construct_sequence(node, deep=True))


_LegacyExperimentConfigLoader.add_constructor(
    _LEGACY_TUPLE_TAG,
    _construct_legacy_tuple,
)


class ModelType(Enum):
    A2C = "A2C"
    PPO = "PPO"
    DQN = "DQN"


class UntrustedModelError(ValueError):
    """Legacy experiment loading lacks an explicit model trust decision."""

def save_model(model, experiment_dir):
    """
    Save the model as model.zip inside the experiment directory.
    """
    model_path = os.path.join(experiment_dir, "model.zip")
    model.save(model_path)
    print(f"Model saved at: {model_path}")
    return model_path


def save_config(env_config, model_config, experiment_dir):
    """
    Save environment and model configurations in config.yaml inside the experiment directory.
    """
    config_path = os.path.join(experiment_dir, "config.yaml")
    config_data = {
        "env_config": env_config,
        "model_config": model_config,
    }
    with open(config_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(config_data, file)
    print(f"Configuration saved at: {config_path}")
    return config_path


def save_training_log(training_log, experiment_dir):
    """
    Save training logs into training.log inside the experiment directory.
    """
    log_path = os.path.join(experiment_dir, "training.log")
    with open(log_path, "w") as file:
        file.write(training_log)
    print(f"Training log saved at: {log_path}")
    return log_path


def save_experiment(
    model,
    env_config,
    model_config,
    training_log,
    experiment_dir=None,
    experiments_base_dir="data/experiments/models",
    *,
    model_connector=None,
    overwrite_model_manifest=False,
):
    """
    Save the experiment data (model, configuration, training log) into the specified experiment directory.
    If experiment_dir is None, a new experiment folder is created under experiments_base_dir with a timestamp.
    If model is None, the function assumes the model is already saved at {experiment_dir}/model.zip.
    
    Files created:
        - model.zip : the saved model (if model is provided, otherwise it should already exist).
        - config.yaml : merged environment and model configurations.
        - training.log : a summary log of the training and evaluation.
        - model.manifest.yaml : an exact model/connector semantic binding when
          ``model_connector`` is supplied explicitly.
        
    Raises:
        FileNotFoundError: If model is None and no model.zip exists in the experiment directory.
    """
    if experiment_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        env_name = env_config.get("env_name", "unknown_env")
        model_type = model_config.get("model_type", "unknown_model")
        experiment_folder_name = f"{env_name}_{model_type}_model_{timestamp}"
        experiment_dir = os.path.join(experiments_base_dir, experiment_folder_name)
        os.makedirs(experiment_dir, exist_ok=True)

    expected_model_path = os.path.join(experiment_dir, "model.zip")
    if model_connector is not None and not overwrite_model_manifest:
        from stache.explainability.model_manifest import manifest_path_for_model

        expected_manifest_path = manifest_path_for_model(Path(expected_model_path))
        if expected_manifest_path.exists() or expected_manifest_path.is_symlink():
            raise FileExistsError(
                f"model manifest already exists: {expected_manifest_path}; pass "
                "overwrite_model_manifest=True to replace it"
            )

    # Handle model saving or validation
    if model is not None:
        # Save the model if provided
        model_path = save_model(model, experiment_dir)
    else:
        # Check if a model already exists at the expected path
        model_path = expected_model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model provided and no existing model found at {model_path}")
        print(f"Using existing model at: {model_path}")

    manifest_path = None
    if model_connector is not None:
        from stache.explainability.model_manifest import (
            write_connector_model_manifest,
        )

        manifest_path = write_connector_model_manifest(
            Path(model_path),
            model_connector,
            overwrite=overwrite_model_manifest,
        )

    config_path = save_config(env_config, model_config, experiment_dir)
    log_path = save_training_log(training_log, experiment_dir)

    saved_paths = {
        "experiment_dir": experiment_dir,
        "model_path": model_path,
        "config_path": config_path,
        "log_path": log_path,
    }
    if manifest_path is not None:
        saved_paths["manifest_path"] = str(manifest_path)
    return saved_paths


def load_experiment(
    experiment_dir,
    *,
    acknowledge_trusted_model=False,
    model_connector=None,
):
    """
    Load an explicitly trusted experiment from a consistent model snapshot.

    Expects:
      - {experiment_dir}/config.yaml
      - {experiment_dir}/model.zip

    ``acknowledge_trusted_model=True`` is required before any path is read.
    When ``model_connector`` is supplied, the conventional semantic sidecar is
    required, validated against the snapshotted bytes and connector, and
    attached to the returned model for compatibility callers.

    Returns:
      A tuple (model, config_data) where:
          model      : the loaded model.
          config_data: the dictionary with 'env_config' and 'model_config'.
    """
    if acknowledge_trusted_model is not True:
        raise UntrustedModelError(
            "load_experiment requires explicit acknowledgement that model.zip "
            "came from a trusted source"
        )

    # Load configuration
    config_path = Path(experiment_dir) / "config.yaml"
    try:
        serialized_config = read_bounded_regular_text(
            config_path,
            max_bytes=EXPERIMENT_CONFIG_MAX_BYTES,
            label="experiment config",
        )
    except SafeInputError as error:
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}") from error
        raise ValueError(str(error)) from error
    config_data = yaml.load(
        serialized_config,
        Loader=_LegacyExperimentConfigLoader,
    )
    if not isinstance(config_data, dict):
        raise ValueError("Experiment config must contain a mapping.")

    # Load model
    model_path = Path(experiment_dir) / "model.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model_type = config_data.get("model_config", {}).get("model_type")
    model_map = {
        "PPO": PPO,
        "A2C": A2C,
        "DQN": DQN,
    }
    if model_type not in model_map:
        raise ValueError(f"Unsupported model type: {model_type}")

    from stache.explainability.core.policy import (
        validate_model_manifest_binding,
    )
    from stache.explainability.model_manifest import (
        load_model_manifest,
        manifest_path_for_model,
        snapshot_model_file,
    )

    model_snapshot, model_fingerprint = snapshot_model_file(model_path)
    model_manifest = None
    if model_connector is not None:
        model_manifest = load_model_manifest(
            manifest_path_for_model(model_path)
        )
        validate_model_manifest_binding(
            model_connector,
            model_fingerprint,
            model_manifest,
        )

    model = model_map[model_type].load(model_snapshot)
    if model_manifest is not None:
        model.stache_model_manifest = model_manifest
    print(f"Loaded model from: {model_path}")
    return model, config_data

def load_config(config_path):
    """
    Load a configuration file and return its contents as a dictionary.
    
    Parameters:
        config_path (str): Path to the configuration file.
        
    Returns:
        dict: Parsed configuration data.
        
    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the input is not a bounded regular UTF-8 file.
        yaml.YAMLError: If the config file contains invalid YAML.
    """
    path = Path(config_path)
    try:
        serialized = read_bounded_regular_text(
            path,
            max_bytes=EXPERIMENT_CONFIG_MAX_BYTES,
            label="configuration file",
        )
    except SafeInputError as error:
        if not path.exists():
            raise FileNotFoundError(
                f"Configuration file not found at: {path}"
            ) from error
        raise ValueError(str(error)) from error
    try:
        config = yaml.safe_load(serialized)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(
            f"Error parsing YAML configuration file at {config_path}: {e}"
        )
    if config is None:
        raise ValueError("Configuration file is empty or has invalid content.")
    return config


def get_device(config_device=None):
    """
    Determine the device to use for training.

    Parameters:
        config_device (str or None): Desired device ('cpu', 'cuda', or 'mps') from the configuration.
                                     If None, the device is chosen automatically based on availability.

    Returns:
        torch.device: The device to be used for training.
    
    Raises:
        ValueError: If the specified device is invalid or unavailable.
    """
    # Automatically determine the best available device if not specified
    if config_device is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    # Validate user-specified device
    config_device = config_device.lower()
    if config_device == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            raise ValueError("CUDA is not available on this system.")
    elif config_device == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            raise ValueError("MPS (Metal Performance Shaders) is not available on this system.")
    elif config_device == "cpu":
        return torch.device("cpu")
    else:
        raise ValueError(f"Unsupported device specified: {config_device}. Choose from 'cpu', 'cuda', or 'mps'.")
