# STACHE: Symbolic Training And CHaracterization Environment

A comprehensive framework for training and analyzing reinforcement learning agents in MiniGrid environments. STACHE supports multiple environment representations, agent algorithms, and provides explainability tools for understanding agent behavior.

This repository enables reinforcement learning with deep customization in MiniGrid environments, featuring symbolic state representations, hyperparameter tuning, and robust policy analysis tools to better understand agent decision-making.

## Features

- **Multiple Environment Support**: Train agents in various MiniGrid environments including Fetch, Empty, and DoorKey
- **Multiple Representation Types**: Choose between symbolic, image-based, and standard observation representations
- **Reinforcement Learning Algorithms**: Support for PPO and A2C algorithms via Stable-Baselines3
- **Hyperparameter Optimization**: Automated tuning using Optuna to find optimal model configurations
- **Policy Evaluation Tools**: Comprehensive evaluation metrics and visualization for trained models
- **Robustness Region Analysis**: Tools to analyze where agents maintain consistent behavior
- **Custom Environment Wrappers**: Specialized wrappers for different observation representations
- **Experiment Management**: Automatic saving and loading of models, configurations, and results

---

## Project Structure

```
STACHE/
├── README.md                      # Project overview and instructions
├── requirements.txt               # List of dependencies
├── pyproject.toml                 # Python project configuration
├── config/
│   ├── training_config_env.yml    # Environment configuration
│   ├── training_config_PPO.yml    # PPO algorithm configuration
│   └── training_config_A2C.yml    # A2C algorithm configuration
├── data/
│   └── experiments/
│       ├── models/                # Saved trained models with configurations
│       ├── evaluation/            # Evaluation results
│       ├── rr/                    # Robustness region analysis results
│       └── optuna_trials/         # Hyperparameter optimization results
├── examples/
│   └── playground.py              # Example usage script
├── src/
│   ├── explainability/            # Tools for analyzing and explaining agent behavior
│   │   ├── evaluate.py            # Evaluation metrics and visualization
│   │   └── rr_bfs.py              # Robustness Region analysis using BFS
│   ├── minigrid_ext/              # Extensions to the MiniGrid environment
│   │   ├── constants.py           # Shared constants for environments
│   │   ├── environment_utils.py   # Environment creation utilities
│   │   ├── set_state_extension.py # State manipulation utilities
│   │   └── wrappers.py            # Observation wrappers
│   ├── pipelines/                 # Training pipelines
│   │   └── train.py               # Main training script
│   ├── tuning/                    # Hyperparameter optimization
│   │   ├── hyperparameter_utils.py # Utility functions for hyperparameter tuning
│   │   └── model_optimization.py   # Optuna-based optimization
│   └── utils/                     # Shared utility functions
│       ├── experiment_io.py       # Experiment saving and loading
│       └── ...
└── tests/                         # Testing suite
    ├── test_set_state.py          # Tests for state manipulation
    └── test_utils.py              # Utility tests
```

---

## Environment Representations

STACHE supports three types of observation representations for MiniGrid environments:

1. **Symbolic Representation**: Factorizes the environment state into structured elements (agent direction, objects, walls, goals) with object-specific attributes. This representation enables more interpretable policy analysis.

2. **Image Representation**: Uses the RGB image view of the environment, similar to what would be visually rendered. This representation works well with CNN policies.

3. **Standard Representation**: Uses the default flattened representation provided by MiniGrid/Gymnasium.

You can select the representation in the configuration file:

```yaml
# Possible values: "symbolic", "image", "standard"
representation: "symbolic"
```

---

## Wrappers

This project includes custom Gymnasium wrappers to preprocess observations:

- **FactorizedSymbolicWrapper**: Converts fully observable grid environments into a structured dictionary of objects, outer walls, and the agent's goal.
- **PaddedObservationWrapper**: Converts the structured observations into a fixed-size 1D array for compatibility with neural networks.

Both wrappers are implemented in `src/wrappers.py`.

---

## Training Agents

To train a reinforcement learning agent:

1. Configure your environment and model parameters in the config files:
   - `config/training_config_env.yml`: Environment settings
   - `config/training_config_PPO.yml` or `config/training_config_A2C.yml`: Algorithm settings

2. Run the training script:
   ```bash
   python -m src.pipelines.train
   ```

Models are automatically saved to `data/experiments/models/` with timestamped directories.

---

## Explainability Features

### Policy Evaluation

The project includes comprehensive tools for evaluating trained policies:

```bash
python -m src.explainability.evaluate
```

This provides detailed performance statistics and visualizations of agent behavior.

### Robustness Region Analysis

The Robustness Region (RR) analysis identifies regions in the state space where an agent consistently takes the same action, helping to understand the boundaries of policy behavior:

```bash
python -m src.explainability.rr_bfs
```

Results are saved to `data/experiments/rr/` and include:
- YAML files describing the region
- Visualizations of states within the region
- Statistical summaries

---

## Hyperparameter Optimization

STACHE uses Optuna for automated hyperparameter tuning:

```bash
python -m src.tuning.model_optimization
```

This systematically explores the hyperparameter space to find optimal configurations for your specific environment.

---

## Configuration

Modify the `config/config.yml` file to adjust the following settings:

- Environment name
- PPO hyperparameters (`n_steps`, `batch_size`, `ent_coef`, etc.)
- Observation wrapper parameters (`max_objects`, `max_walls`, etc.)

Example:
```yaml
env_name: MiniGrid-Fetch-6x6-N2-v0
total_timesteps: 50000
n_steps: 256
batch_size: 64
ent_coef: 0.01
max_objects: 10
max_walls: 16
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [Gymnasium](https://gymnasium.farama.org/)
- [MiniGrid](https://github.com/Farama-Foundation/MiniGrid)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
