#!/usr/bin/env python3
"""
test_robustness_region.py

This module contains tests for the BFS-based Robustness Region (BFS-RR) algorithm
implemented in src/explainability/rr_bfs.py. The tests validate that the algorithm
correctly identifies states with the same action as the initial state and follows
the rules for atomic state modifications.
"""

import os
import time
import copy
import numpy as np
import gymnasium as gym
import pytest
from collections import deque

from utils.experiment_io import load_experiment
from minigrid_ext.environment_utils import create_minigrid_env
from minigrid_ext.state_utils import (
    symbolic_to_array,
    state_to_key,
    get_grid_dimensions
)
from minigrid_ext.set_state_extension import set_standard_state_minigrid, factorized_symbolic_to_fullobs
from explainability.rr_bfs import bfs_rr, get_symbolic_env, generate_rr_images
from explainability.minigrid_neighbor_generation import get_neighbors

# Test constants
TEST_ENV_NAME = "MiniGrid-Fetch-5x5-N2-v0"  
TEST_MODEL_PATH = "data/experiments/models/MiniGrid-Fetch-5x5-N2-v0_PPO_model_20250304_211758"
TEST_SEED = 42
MAX_OBJECTS = 10
MAX_WALLS = 25
MAX_GEN_OBJECTS = 2
MAX_NODES_EXPANDED = 10  # Small value for quick tests

class MockModel:
    """Mock model that always returns the same action for testing purposes."""
    
    def __init__(self, action=0):
        self.action = action
        self.called_states = []
    
    def predict(self, obs, deterministic=True):
        """
        Return a predetermined action. 
        If deterministic is True, records the observation for verification.
        """
        if deterministic:
            self.called_states.append(copy.deepcopy(obs))
        return self.action, None


def test_robustness_region_basic():
    """
    Test basic functionality of BFS-RR algorithm with a mock model that 
    always returns the same action.
    """
    # Create a simple environment config
    env_config = {
        "env_name": TEST_ENV_NAME,
        "render_mode": "rgb_array", 
        "max_objects": MAX_OBJECTS,
        "max_walls": MAX_WALLS,
        "representation": "symbolic",
    }
    
    # Create the environment and get initial state
    env = create_minigrid_env(env_config)
    symbolic_env = get_symbolic_env(env)
    initial_state, _ = symbolic_env.reset(seed=TEST_SEED)
    
    # Create mock model that always returns action 0
    mock_model = MockModel(action=0)
    
    # Run BFS-RR with small max_nodes_expanded
    robustness_region, stats = bfs_rr(
        initial_state,
        mock_model,
        env_name=env_config["env_name"],
        max_obs_objects=env_config["max_objects"],
        max_walls=env_config["max_walls"],
        max_gen_objects=MAX_GEN_OBJECTS,
        max_nodes_expanded=MAX_NODES_EXPANDED
    )
    
    # Validate results
    assert len(robustness_region) > 0, "Robustness region should not be empty"
    assert stats["initial_action"] == 0, "Initial action should be 0"
    assert stats["region_size"] == len(robustness_region), "Region size should match length of robustness_region"
    
    # Check that all states in robustness region have bfs_depth attribute
    for state in robustness_region:
        assert "bfs_depth" in state, "Each state should have a bfs_depth attribute"
    
    # Check that the initial state is in the robustness region with depth 0
    initial_state_in_region = False
    for state in robustness_region:
        if state_to_key(state) == state_to_key(initial_state) and state["bfs_depth"] == 0:
            initial_state_in_region = True
            break
    assert initial_state_in_region, "Initial state should be in robustness region with depth 0"
    
    # Check that mock_model was called at least once
    assert len(mock_model.called_states) > 0, "Model should have been called at least once"
    
    # Clean up
    env.close()


def test_robustness_region_action_consistency():
    """
    Test that BFS-RR only includes states with the same action as the initial state.
    Uses a mock model that returns different actions for different states.
    """
    # Create a simple environment config
    env_config = {
        "env_name": TEST_ENV_NAME,
        "render_mode": "rgb_array", 
        "max_objects": MAX_OBJECTS,
        "max_walls": MAX_WALLS,
        "representation": "symbolic",
    }
    
    # Create the environment and get initial state
    env = create_minigrid_env(env_config)
    symbolic_env = get_symbolic_env(env)
    initial_state, _ = symbolic_env.reset(seed=TEST_SEED)
    
    # Create a specialized mock model that returns action 1 for initial state,
    # then action 0 for all other states
    class VariableActionModel:
        def __init__(self, initial_state_action=1, other_action=0):
            self.initial_state_action = initial_state_action
            self.other_action = other_action
            self.initial_state_key = state_to_key(initial_state)
            self.called_states = []
            
        def predict(self, obs, deterministic=True):
            # For array observations, can't use state_to_key directly
            if isinstance(obs, np.ndarray):
                self.called_states.append(True)  # Just record that we were called
                return self.initial_state_action, None
                
            # For symbolic states
            state_key = state_to_key(obs)
            self.called_states.append(state_key)
            
            # Return different action based on whether this is the initial state
            if state_key == self.initial_state_key:
                return self.initial_state_action, None
            else:
                return self.other_action, None
    
    variable_model = VariableActionModel()
    
    # Run BFS-RR
    robustness_region, stats = bfs_rr(
        initial_state,
        variable_model,
        env_name=env_config["env_name"],
        max_obs_objects=env_config["max_objects"],
        max_walls=env_config["max_walls"],
        max_gen_objects=MAX_GEN_OBJECTS,
        max_nodes_expanded=MAX_NODES_EXPANDED
    )
    
    # The region should only contain the initial state because all other
    # states will have a different action
    assert len(robustness_region) == 1, "Robustness region should only contain the initial state"
    assert robustness_region[0]["bfs_depth"] == 0, "Initial state should have depth 0"
    assert stats["initial_action"] == 1, "Initial action should be 1"
    assert stats["region_size"] == 1, "Region size should be 1"
    
    # Check that variable_model was called at least once
    assert len(variable_model.called_states) > 0, "Model should have been called at least once"
    
    # Clean up
    env.close()


def test_max_nodes_expanded_limit():
    """
    Test that BFS-RR respects the max_nodes_expanded limit.
    """
    # Create a simple environment config
    env_config = {
        "env_name": TEST_ENV_NAME,
        "render_mode": "rgb_array", 
        "max_objects": MAX_OBJECTS,
        "max_walls": MAX_WALLS,
        "representation": "symbolic",
    }
    
    # Create the environment and get initial state
    env = create_minigrid_env(env_config)
    symbolic_env = get_symbolic_env(env)
    initial_state, _ = symbolic_env.reset(seed=TEST_SEED)
    
    # Create mock model that always returns action 0
    mock_model = MockModel(action=0)
    
    # Run BFS-RR with very small max_nodes_expanded limit
    tiny_limit = 3
    robustness_region, stats = bfs_rr(
        initial_state,
        mock_model,
        env_name=env_config["env_name"],
        max_obs_objects=env_config["max_objects"],
        max_walls=env_config["max_walls"],
        max_gen_objects=MAX_GEN_OBJECTS,
        max_nodes_expanded=tiny_limit
    )
    
    # Validate that it respects the limit
    assert stats["total_opened_nodes"] <= tiny_limit, f"Should not expand more than {tiny_limit} nodes"
    
    # Run again with larger limit to verify it can find more states
    larger_limit = 10
    larger_region, larger_stats = bfs_rr(
        initial_state,
        mock_model,
        env_name=env_config["env_name"],
        max_obs_objects=env_config["max_objects"],
        max_walls=env_config["max_walls"],
        max_gen_objects=MAX_GEN_OBJECTS,
        max_nodes_expanded=larger_limit
    )
    
    # Validate that with larger limit it finds more states or expands more nodes
    assert larger_stats["total_opened_nodes"] > stats["total_opened_nodes"], \
        "Should expand more nodes with larger limit"
    
    # Clean up
    env.close()


def test_bfs_depth_correctness():
    """
    Test that BFS-RR correctly assigns BFS depth to states
    based on L1 distance from initial state.
    """
    # Create a simple environment config
    env_config = {
        "env_name": TEST_ENV_NAME,
        "render_mode": "rgb_array", 
        "max_objects": MAX_OBJECTS,
        "max_walls": MAX_WALLS,
        "representation": "symbolic",
    }
    
    # Create the environment and get initial state
    env = create_minigrid_env(env_config)
    symbolic_env = get_symbolic_env(env)
    initial_state, _ = symbolic_env.reset(seed=TEST_SEED)
    
    # Create mock model that always returns action 0
    mock_model = MockModel(action=0)
    
    # Run BFS-RR
    robustness_region, stats = bfs_rr(
        initial_state,
        mock_model,
        env_name=env_config["env_name"],
        max_obs_objects=env_config["max_objects"],
        max_walls=env_config["max_walls"],
        max_gen_objects=MAX_GEN_OBJECTS,
        max_nodes_expanded=MAX_NODES_EXPANDED
    )
    
    # Group states by bfs_depth
    states_by_depth = {}
    for state in robustness_region:
        depth = state["bfs_depth"]
        if depth not in states_by_depth:
            states_by_depth[depth] = []
        states_by_depth[depth].append(state)
    
    # Validate that depths are assigned correctly
    # - Initial state should have depth 0
    # - Each depth level should contain states reachable from the previous level
    assert 0 in states_by_depth, "Initial state should have depth 0"
    assert len(states_by_depth[0]) == 1, "Only the initial state should have depth 0"
    
    # Check that each state at depth N+1 is a neighbor of some state at depth N
    for depth in sorted(states_by_depth.keys()):
        if depth == 0:
            continue  # Skip initial state
            
        # For each state at this depth
        found_valid_ancestor = False
        for state in states_by_depth[depth]:
            # Check that it's a neighbor of some state at previous depth
            for prev_state in states_by_depth[depth - 1]:
                # Generate neighbors of the previous state
                neighbors = get_neighbors(
                    prev_state, 
                    env_name=env_config["env_name"], 
                    max_gen_objects=MAX_GEN_OBJECTS
                )
                
                # Check if current state is one of these neighbors
                for neighbor in neighbors:
                    if state_to_key(neighbor) == state_to_key(state):
                        found_valid_ancestor = True
                        break
                
                if found_valid_ancestor:
                    break
            
            if found_valid_ancestor:
                break
        
        assert found_valid_ancestor, f"States at depth {depth} should be neighbors of states at depth {depth-1}"
    
    # Clean up
    env.close()


def test_with_real_model():
    """
    Test the BFS-RR algorithm with a real pre-trained model to ensure
    integration with the actual codebase.
    This test is skipped if the model file doesn't exist.
    """
    # Skip this test if the model file doesn't exist
    if not os.path.exists(TEST_MODEL_PATH):
        pytest.skip(f"Test model not found at {TEST_MODEL_PATH}")
    
    # Load the real model
    try:
        model, config_data = load_experiment(TEST_MODEL_PATH)
    except Exception as e:
        pytest.skip(f"Could not load model: {e}")
    
    # Extract environment configuration
    env_config = config_data["env_config"]
    
    # Create environment
    env = create_minigrid_env(env_config)
    symbolic_env = get_symbolic_env(env)
    initial_state, _ = symbolic_env.reset(seed=TEST_SEED)
    
    # Run BFS-RR with real model
    robustness_region, stats = bfs_rr(
        initial_state,
        model,
        env_name=env_config["env_name"],
        max_obs_objects=env_config["max_objects"],
        max_walls=env_config["max_walls"],
        max_gen_objects=MAX_GEN_OBJECTS,
        max_nodes_expanded=MAX_NODES_EXPANDED
    )
    
    # Basic validation
    assert len(robustness_region) > 0, "Robustness region should not be empty"
    assert stats["region_size"] == len(robustness_region), "Region size should match length of robustness_region"
    assert stats["initial_action"] is not None, "Initial action should be set"
    
    # Check that initial state is in the region
    initial_key = state_to_key(initial_state)
    assert any(state_to_key(state) == initial_key for state in robustness_region), \
        "Initial state should be in the robustness region"
    
    # Clean up
    env.close()


def test_image_generation():
    """
    Test the generate_rr_images function to ensure it correctly renders
    and saves images of states in the robustness region.
    """
    # Create a simple environment config
    env_config = {
        "env_name": TEST_ENV_NAME,
        "render_mode": "rgb_array", 
        "max_objects": MAX_OBJECTS,
        "max_walls": MAX_WALLS,
        "representation": "symbolic",
    }
    
    # Create the environment and get initial state
    env = create_minigrid_env(env_config)
    symbolic_env = get_symbolic_env(env)
    initial_state, _ = symbolic_env.reset(seed=TEST_SEED)
    
    # Add BFS depth to the initial state to mock a robustness region entry
    initial_state["bfs_depth"] = 0
    robustness_region = [initial_state]
    
    # Set up test output directory
    test_output_dir = os.path.join("tests", "test_data", "rr_images_test")
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Call generate_rr_images
    generate_rr_images(
        robustness_region=robustness_region,
        env=env,
        output_dir=test_output_dir,
        subset="all"
    )
    
    # Verify output
    expected_filename = os.path.join(test_output_dir, "depth_0_idx_0.png")
    assert os.path.exists(expected_filename), f"Expected image file not found: {expected_filename}"
    
    # Clean up test directory
    if os.path.exists(expected_filename):
        os.remove(expected_filename)
    
    # Clean up
    env.close()


def test_neighbor_generation_rules():
    """
    Test that neighbors generated for BFS-RR follow the atomic change rules:
    1. Direction change by ±1
    2. Goal color/type change
    3. Modifying one attribute of an existing object
    4. Adding a new object (if max_objects not exceeded)
    """
    # Create a simple environment config
    env_config = {
        "env_name": TEST_ENV_NAME,
        "render_mode": "rgb_array", 
        "max_objects": MAX_OBJECTS,
        "max_walls": MAX_WALLS,
        "representation": "symbolic",
    }
    
    # Create the environment and get initial state
    env = create_minigrid_env(env_config)
    symbolic_env = get_symbolic_env(env)
    initial_state, _ = symbolic_env.reset(seed=TEST_SEED)
    
    # Generate neighbors
    neighbors = get_neighbors(
        initial_state, 
        env_name=env_config["env_name"], 
        max_gen_objects=MAX_GEN_OBJECTS
    )
    
    # Validate that each neighbor differs by exactly one atomic change
    for neighbor in neighbors:
        # Count differences between initial_state and neighbor
        differences = []
        
        # Check direction change
        if neighbor["direction"] != initial_state["direction"]:
            differences.append("direction")
            # Validate it's a ±1 change, accounting for wrap-around (modulo arithmetic)
            # Directions are usually 0,1,2,3 where 3->0 is one step (not 3 steps)
            dir1 = int(neighbor["direction"])
            dir2 = int(initial_state["direction"])
            # Calculate minimum distance in a circular array of length 4
            dir_diff = min((dir1 - dir2) % 4, (dir2 - dir1) % 4)
            assert dir_diff == 1, f"Direction change should be a single step (considering circular nature), got: {dir_diff}"
            
        # Check goal change
        if neighbor["goal"] != initial_state["goal"]:
            differences.append("goal")
            # Validate it's a coordinate change by 1
            if len(differences) == 1:  # Only care if this is the only difference
                goal_diff = 0
                if neighbor["goal"][0] != initial_state["goal"][0]:
                    goal_diff += abs(neighbor["goal"][0] - initial_state["goal"][0])
                if neighbor["goal"][1] != initial_state["goal"][1]:
                    goal_diff += abs(neighbor["goal"][1] - initial_state["goal"][1])
                assert goal_diff == 1, f"Goal coordinate change should be by 1, got: {goal_diff}"
        
        # Check object changes
        if len(neighbor["objects"]) != len(initial_state["objects"]):
            differences.append("object_count")
            # Validate we're only adding one object
            if len(differences) == 1:  # Only care if this is the only difference
                assert len(neighbor["objects"]) == len(initial_state["objects"]) + 1, \
                    "Should only add one object at a time"
                assert len(neighbor["objects"]) <= MAX_GEN_OBJECTS, \
                    f"Should not exceed max_gen_objects ({MAX_GEN_OBJECTS})"
        else:
            # Same number of objects, check if any attributes changed
            for i, (init_obj, neigh_obj) in enumerate(zip(initial_state["objects"], neighbor["objects"])):
                if init_obj != neigh_obj:
                    differences.append(f"object_{i}")
                    # Validate it's only one attribute change
                    if len(differences) == 1:  # Only care if this is the only difference
                        attr_diffs = 0
                        for j in range(len(init_obj)):
                            if init_obj[j] != neigh_obj[j]:
                                attr_diffs += 1
                        assert attr_diffs == 1, f"Only one attribute should change per object, got: {attr_diffs}"
        
        # Ensure exactly one type of change occurred
        assert len(differences) == 1, \
            f"Each neighbor should differ by exactly one atomic change, got {len(differences)}: {differences}"
    
    # Clean up
    env.close()


if __name__ == "__main__":
    print(f"\n=== Running robustness region tests ===")
    
    print("\n=== Running test_robustness_region_basic ===")
    test_robustness_region_basic()
    
    print("\n=== Running test_robustness_region_action_consistency ===")
    test_robustness_region_action_consistency()
    
    print("\n=== Running test_max_nodes_expanded_limit ===")
    test_max_nodes_expanded_limit()
    
    print("\n=== Running test_bfs_depth_correctness ===")
    test_bfs_depth_correctness()
    
    if os.path.exists(TEST_MODEL_PATH):
        print("\n=== Running test_with_real_model ===")
        test_with_real_model()
    else:
        print(f"\n=== Skipping test_with_real_model (model not found at {TEST_MODEL_PATH}) ===")
    
    print("\n=== Running test_image_generation ===")
    test_image_generation()
    
    print("\n=== Running test_neighbor_generation_rules ===")
    test_neighbor_generation_rules()
    
    print("\nAll tests passed successfully!")