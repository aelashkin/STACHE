
# PPO Training for MiniGrid Fetch Environment

This repository implements PPO-based training for the `MiniGrid-Fetch` environment using custom wrappers for observation preprocessing and padded observations. It supports local execution and is optimized for use with Google Colab.

## Features

- Custom **FactorizedSymbolicWrapper** and **PaddedObservationWrapper** to preprocess and format observations.
- Modular design to support experimentation and extension.
- Training with **Stable-Baselines3 PPO**.
- Supports hyperparameter tuning using **Optuna** (future extension).
- Clear structure for local and Colab usage.

---

## Project Structure

```
my-ppo-fetch-project/
├── README.md                 # Project overview and instructions
├── requirements.txt          # List of dependencies
├── .gitignore                # Files and folders to ignore in Git
├── config/
│   └── config.yml            # Configuration for environment and training
├── data/
│   ├── logs/                 # Training logs and metrics
│   └── models/               # Saved trained models
├── notebooks/
│   ├── exploration.ipynb     # Notebook for exploration and debugging
│   └── training_colab.ipynb  # Colab notebook for training
├── src/
│   ├── __init__.py
│   ├── wrappers.py           # All custom wrappers
│   ├── train.py              # Training script
│   └── utils.py              # Utility functions
└── tests/
    ├── __init__.py
    └── test_wrappers.py      # Unit tests for wrappers
```

---


## Wrappers

This project includes custom Gymnasium wrappers to preprocess observations:

- **FactorizedSymbolicWrapper**: Converts fully observable grid environments into a structured dictionary of objects, outer walls, and the agent's goal.
- **PaddedObservationWrapper**: Converts the structured observations into a fixed-size 1D array for compatibility with neural networks.

Both wrappers are implemented in `src/wrappers.py`.

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
