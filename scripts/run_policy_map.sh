#!/usr/bin/env zsh
# Script to generate policy maps for Taxi-v3 DQN models 0, 50, and 100
# Run from the workspace root (/Users/eam/Documents/GIT/STACHE)

for model in data/experiments/models/Taxi-v3_DQN_model_{0,50,100}; do
  echo "Processing $model..."
  python src/stache/explainability/taxi/taxi_policy_map.py \
    --model-path "$model"
  echo
 done
