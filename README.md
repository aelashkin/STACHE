# STACHE – State–Action Transparency through Counterfactual & Robustness Explanations


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Thesis](https://img.shields.io/badge/PDF-MSc_Thesis-red)](docs/MSc_Thesis_Andrew_Elashkin_2025.pdf)


STACHE is a **model-agnostic** toolkit for explaining discrete Reinforcement Learning agents. It maps the geometry of an agent's decision-making process by computing **Robustness Regions** (stability zones) and **Minimal Counterfactuals** (critical tipping points).

Unlike standard saliency maps that highlight pixels, STACHE operates on **symbolic, factored state spaces** (e.g., Taxi-v3, MiniGrid). This allows it to generate explanations that are semantically meaningful: *"The agent picked 'South' because the passenger was at (0,4); if the passenger were at (0,3), it would have picked 'North'."*

---

## Visual Evidence: The Evolution of Logic

STACHE allows researchers to visualize how an agent's logic stabilizes during training. Below is a comparison of **Robustness Regions (RR)** for a Taxi-v3 agent at **0%** training vs. **100%** training, starting from the same seed state $s = (0,0,0,2)$ (*Taxi at top-left, Passenger at R, Dest at Y*).

| **Untrained Policy ($`\pi_{0\%}`$)** | **Fully Trained Policy ($`\pi_{100\%}`$)** |
| :---: | :---: |
| <img src="assets/taxi/Taxi-v3_DQN_model_0/0_0_0_2/robustness_region_0_0_0_2.png" width="400" /> | <img src="assets/taxi/Taxi-v3_DQN_model_100/0_0_0_2/robustness_region_0_0_0_2.png" width="400" /> |
| **Erratic Behavior.** The colored regions (representing invariant actions) are scattered and small (Size=9). The agent's decision is unstable and sensitive to random noise. | **Stable Logic.** The region is compact and specific (Size=3). The agent correctly identifies "Pick-up" and holds that decision only while the taxi/passenger are co-located, demonstrating precise logic. |

---

## Core Concepts

This toolkit implements the "Composite Explanation" framework presented in the [accompanying thesis](docs/MSc_Thesis_Andrew_Elashkin_2025.pdf).

### 1. Robustness Regions (RR)
**Definition:** The **Robustness Region** $\mathcal{R}(s_0, \pi)$ is the connected set of states around a seed state $s_0$ where the agent's action $\pi(s)$ remains unchanged.

Instead of testing random perturbations, STACHE defines a graph stucture over the factored state space and performs a **Breadth-First Search (BFS)**-based exploration. A state belongs to the region if:
1.  The agent selects the same action as the seed state.
2.  It is reachable via a "continuous path" of atomic changes (e.g., moving one tile, changing one color) without the action ever flipping along the path.

**Why it matters:** Large regions imply **navigational stability** (the agent sticks to the plan). Small, specific regions imply **functional precision** (e.g., "Pick-up" is only valid in one specific spot).

### 2. Minimal Counterfactual States (CF)
**Definition:** A **Minimal Counterfactual** is the state $s'$ closest to the seed state $s_0$ (in terms of L1 distance) that forces the policy to change its action.

STACHE identifies these states by examining the **boundary** of the computed Robustness Region.

**Why it matters:** These are the "tipping points" of the policy. They reveal exactly which feature (e.g., *Passenger Location* vs. *Taxi X-Coordinate*) controls the decision.

---





## Table of Contents
1. [Quick install](#quick-install)  
2. [Key features](#key-features)  
3. [Getting started](#getting-started)  
4. [Explaining an agent](#explaining-an-agent)  
5. [Reproducing paper results](#reproducing-paper-results)  
6. [Project layout](#project-layout)  
7. [Citing](#citing)  
8. [Contributing](#contributing)  
9. [License](#license)

---

## Quick install

First, clone the repository and create a Python virtual environment (version 3.11 or newer is required).

```bash
git clone https://github.com/your-org/stache.git
cd stache
python3 -m venv .venv && source .venv/bin/activate
```

Next, install the project and its dependencies.

**For users** who want to run the core library without the extra tuning or testing tools, a standard installation is sufficient:

```bash
# Standard user installation
pip install .
```

**For developers** who will be modifying the code, install the package in "editable" mode with all optional dependencies (for tuning and testing):

```bash
# Developer setup (installs everything)
pip install -e .[dev]
```

> **Tip:** This package requires PyTorch to run Stable Baselines3 models. The default version will be for CPU. If you have a CUDA-enabled GPU, you can get better performance by pre-installing the correct PyTorch wheel for your system *before* running the commands above. Please see the official [PyTorch installation guide](https://pytorch.org/get-started/locally/) for instructions.

---

## Key features

| Module                  | What it does                                                                                 | Location                                |
| ----------------------- | -------------------------------------------------------------------------------------------- | --------------------------------------- |
| **Symbolic wrappers**   | Factorise MiniGrid observations into objects / walls / goals, pad to fixed-size vectors.     | `src/stache/envs/minigrid/wrappers.py`  |
| **Env factories**       | One-line creation of Taxi-v3 / MiniGrid with the right observation mode.                     | `src/stache/envs/factory.py`            |
| **Training pipelines**  | Opinionated scripts for PPO, A2C and DQN with auto-saving & Optuna tuning.                   | `src/stache/pipelines/`                 |
| **Explainability core** | Exact BFS over discrete state space to compute minimal counterfactuals & robustness regions. | `src/stache/explainability/`            |
| **Rich visualisers**    | YAML + colour-blind PNGs for policy maps, RR plots, counter-factual grids.                   | `scripts/run_*`, `…/taxi_policy_map.py` |
| **Experiment I/O**      | Unified save/load of models, configs, logs.                                                  | `src/stache/utils/experiment_io.py`     |

---

## Getting started

### 1 · Train an agent (MiniGrid–Fetch, PPO)

```bash
python -m src.stache.pipelines.train_minigrid \
    --env-name MiniGrid-Fetch-5x5-N2-v0 \
    --model-type PPO \
    --total-timesteps 1_000_000
# outputs → data/experiments/models/MiniGrid-Fetch-5x5-N2-v0_PPO_model_YYYYMMDD_HHMMSS/
```

### 2 · Evaluate

```bash
python -m src.stache.explainability.evaluate \
    --model-path data/experiments/models/.../model.zip
```

---

## Explaining an agent

Compute and visualise robustness regions **without touching model internals**:

```bash
python scripts/run_taxi_rr.sh \
    --model-path data/experiments/models/Taxi-v3_DQN_model_100 \
    --state "0,1,2,1"
# → data/experiments/rr/taxi_robustness_region/…
```

* **YAML** output lists all RR states, BFS depths, and minimal counter-factuals.
* **PNG** grids illustrate where the chosen action stays constant and where it flips.

See `src/stache/explainability/minigrid/minigrid_neighbor_generation.py` for environment-specific neighbour logic.

---

## Reproducing paper results

1. **Install** as above.
2. Models shown in the paper are already downloaded in `data/experiments/models/`.
3. Run the corresponding `scripts/run_*` helper—each script sets the exact seeds and configs used in the paper.
4. Generated artefacts (YAML, PNGs) reproduce Figures and Tables for Minigrid and TaxiV3 of the manuscript.

---

## Project layout

```
├── config/              # YAML presets for env & algo
├── scripts/             # Thin CLI wrappers for common tasks
├── src/stache/          # All library code (import as `stache`)
│   ├── envs/            # Factory + wrappers for MiniGrid / Taxi
│   ├── explainability/  # RR & CF algorithms, visualisation
│   ├── pipelines/       # Train / tune / evaluate workflows
│   └── utils/           # Experiment I/O, helpers
└── data/experiments/    # Created on-the-fly (models, RR, Optuna, …)
```

---

## Citing

If you use this repo, please cite the paper:

```bibtex
@misc{Elashkin2025,
  author    = {Elashkin, Andrey and Grumberg, Orna},
  title     = {Counterfactual and Robustness-Based Explanations for Reinforcement Learning Policies},
  year      = {2025},
  publisher = {Technion - Israel Institute of Technology},
  keywords  = {Reinforcement learning; Intelligent agents; Markov processes; Multiagent systems},
  note      = {MSc Thesis. Supervision: Orna Grumberg},
  url       = {https://github.com/aelashkin/STACHE/docs/MSc_Thesis_Andrew_Elashkin_2025.pdf}
}
```

---

## Contributing

Pull requests are welcome! Please:

1. Open an issue describing the bug / feature.
2. Create a branch (`feat/my-feature`), add unit tests where relevant.
3. Run `ruff`, `black`, and `pytest`.
4. Submit the PR; CI must pass before review.

---

## License

Distributed under the **MIT License**.
See [`LICENSE`](LICENSE) for full text.

---

### Acknowledgements

Built on top of
[Gymnasium](https://gymnasium.farama.org/) ·
[MiniGrid](https://github.com/Farama-Foundation/MiniGrid) ·
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/) ·
[Optuna](https://optuna.org/).
