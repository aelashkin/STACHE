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
            # Store both the key and the array representation of the initial state
            self.initial_state_key = state_to_key(initial_state)
            self.initial_state_array = symbolic_to_array(initial_state, MAX_OBJECTS, MAX_WALLS)
            self.called_states = []
            
        def predict(self, obs, deterministic=True):
            # For array observations
            if isinstance(obs, np.ndarray):
                self.called_states.append(copy.deepcopy(obs))
                # Compare with the stored initial state array
                if np.array_equal(obs, self.initial_state_array):
                    return self.initial_state_action, None
                else:
                    return self.other_action, None
                
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
    
    This test strictly validates that each neighbor differs from the initial state
    by exactly one L1 difference, as specified in the BFS-RR algorithm description.
    It provides detailed debug information about all violations to help diagnose
    issues in the neighbor generation implementation.
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
    
    print(f"\n=== TESTING NEIGHBOR GENERATION RULES ===")
    print(f"Initial state: {initial_state}")
    print(f"Generated {len(neighbors)} neighbors")
    
    # Track violations for analysis
    violations = []
    violations_by_type = {
        "multiple_changes": [],
        "multiple_attributes": [],
        "large_direction_change": [],
        "multiple_goal_attrs": [],
        "invalid_object_count": [],
    }
    
    # Analyze each neighbor state
    for i, neighbor in enumerate(neighbors):
        print(f"\n--- Analyzing neighbor {i+1}/{len(neighbors)} ---")
        print(f"Neighbor state: {neighbor}")
        
        # Track all differences
        differences = []
        
        # Check direction change
        if neighbor["direction"] != initial_state["direction"]:
            dir1 = int(neighbor["direction"])
            dir2 = int(initial_state["direction"])
            # Calculate minimum distance in a circular array of length 4
            dir_diff = min((dir1 - dir2) % 4, (dir2 - dir1) % 4)
            
            if dir_diff == 1:
                differences.append(f"direction_change: {dir2} -> {dir1}")
                print(f"  ✓ Direction changed: {dir2} -> {dir1} (valid: diff = {dir_diff})")
            else:
                violations_by_type["large_direction_change"].append(i)
                diff_desc = f"direction_change: {dir2} -> {dir1} (invalid: diff = {dir_diff})"
                differences.append(diff_desc)
                print(f"  ✗ {diff_desc}")
        
        # Check goal change
        if neighbor["goal"] != initial_state["goal"]:
            goal_changes = []
            if neighbor["goal"][0] != initial_state["goal"][0]:
                goal_changes.append(f"type: {initial_state['goal'][0]} -> {neighbor['goal'][0]}")
            if neighbor["goal"][1] != initial_state["goal"][1]:
                goal_changes.append(f"color: {initial_state['goal'][1]} -> {neighbor['goal'][1]}")
            
            if len(goal_changes) == 1:
                differences.append(f"goal_change: {goal_changes[0]}")
                print(f"  ✓ Goal changed: {goal_changes[0]}")
            else:
                violations_by_type["multiple_goal_attrs"].append(i)
                diff_desc = f"goal_change: {', '.join(goal_changes)} (invalid: multiple attributes)"
                differences.append(diff_desc)
                print(f"  ✗ {diff_desc}")
        
        # Check object count changes
        if len(neighbor["objects"]) != len(initial_state["objects"]):
            diff = len(neighbor["objects"]) - len(initial_state["objects"])
            if diff == 1:
                # Valid: One object added
                new_obj_idx = len(neighbor["objects"]) - 1  # Assume it's the last one
                new_obj = neighbor["objects"][new_obj_idx]
                differences.append(f"object_added: {new_obj}")
                print(f"  ✓ Object added: {new_obj}")
            else:
                # Invalid: Multiple objects added or objects removed
                violations_by_type["invalid_object_count"].append(i)
                diff_desc = f"object_count_change: {len(initial_state['objects'])} -> {len(neighbor['objects'])} (invalid: diff = {diff})"
                differences.append(diff_desc)
                print(f"  ✗ {diff_desc}")
        else:
            # Same number of objects, check for attribute changes
            changed_objects = []
            
            for obj_idx, (init_obj, neigh_obj) in enumerate(zip(initial_state["objects"], neighbor["objects"])):
                if init_obj != neigh_obj:
                    # Detect which attributes changed
                    changed_attrs = []
                    attr_names = ["type", "color", "state", "x_pos", "y_pos"]
                    
                    for attr_idx, (initial_val, neighbor_val) in enumerate(zip(init_obj, neigh_obj)):
                        if initial_val != neighbor_val:
                            changed_attrs.append(f"{attr_names[attr_idx]}: {initial_val} -> {neighbor_val}")
                    
                    if len(changed_attrs) == 1:
                        # Valid: One attribute changed
                        diff_desc = f"object_{obj_idx}_changed: {changed_attrs[0]}"
                        differences.append(diff_desc)
                        print(f"  ✓ Object {obj_idx} changed: {changed_attrs[0]}")
                    else:
                        # Invalid: Multiple attributes changed
                        violations_by_type["multiple_attributes"].append(i)
                        diff_desc = f"object_{obj_idx}_changed: {', '.join(changed_attrs)} (invalid: multiple attributes)"
                        differences.append(diff_desc)
                        print(f"  ✗ {diff_desc}")
                    
                    changed_objects.append(obj_idx)
        
        # Check overall atomic change rule (exactly one L1 difference)
        if len(differences) == 1:
            print(f"  ✓ VALID NEIGHBOR: Exactly one L1 difference")
        else:
            violations_by_type["multiple_changes"].append(i)
            print(f"  ✗ INVALID NEIGHBOR: {len(differences)} differences found: {differences}")
            violations.append({
                "neighbor_idx": i,
                "differences": differences,
                "diff_count": len(differences)
            })
    
    # Print summary of violations
    print("\n=== NEIGHBOR GENERATION VALIDATION SUMMARY ===")
    print(f"Total neighbors generated: {len(neighbors)}")
    print(f"Valid neighbors (one L1 difference): {len(neighbors) - len(violations_by_type['multiple_changes'])}")
    print(f"Invalid neighbors: {len(violations_by_type['multiple_changes'])}")
    print("\nViolations by type:")
    print(f"- Multiple changes in one neighbor: {len(violations_by_type['multiple_changes'])}")
    print(f"- Multiple attributes changed for one object: {len(violations_by_type['multiple_attributes'])}")
    print(f"- Invalid direction changes: {len(violations_by_type['large_direction_change'])}")
    print(f"- Multiple goal attributes changed: {len(violations_by_type['multiple_goal_attrs'])}")
    print(f"- Invalid object count changes: {len(violations_by_type['invalid_object_count'])}")
    
    # Provide detailed examples of violations for debugging
    if violations:
        print("\n=== DETAILED VIOLATION EXAMPLES ===")
        for i, v in enumerate(violations[:5]):  # Show first 5 violations for brevity
            print(f"Violation {i+1} (Neighbor {v['neighbor_idx']+1}):")
            print(f"- Differences: {v['diff_count']}")
            for diff in v['differences']:
                print(f"  * {diff}")
        
        if len(violations) > 5:
            print(f"... and {len(violations) - 5} more violations")
        
        # Fail the test with a clear message about the number of violations
        assert False, (f"{len(violations_by_type['multiple_changes'])} neighbors violate the L1 difference rule. "
                      f"Fix the neighbor generation in minigrid_neighbor_generation.py to ensure each neighbor "
                      f"differs from the initial state by exactly one atomic change.")
    
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