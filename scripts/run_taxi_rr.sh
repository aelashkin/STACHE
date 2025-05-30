#!/usr/bin/env bash
# Script to run robustness region visualization for Taxi-v3 models and seed states
set -euo pipefail

# Define model directories
MODELS=(
  "Taxi-v3_DQN_model_0"
  "Taxi-v3_DQN_model_50"
  "Taxi-v3_DQN_model_100"
)

# Define seed states
STATES=(
  "0,0,0,2"
  "0,1,2,1"
  "1,1,1,2"
)

# Loop through each model and state
for model in "${MODELS[@]}"; do
  for state in "${STATES[@]}"; do
    echo "Running visualization for model ${model}, state ${state}..."
    python3 src/stache/explainability/taxi/taxi_robustness_region_visualization.py \
      --model-path "data/experiments/models/${model}" \
      --state "${state}"
  done
done
