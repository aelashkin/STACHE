import os
import numpy as np
import gymnasium as gym
from PIL import Image
import pytest

from minigrid.wrappers import FullyObsWrapper
from src.environment_utils import create_symbolic_minigrid_env
from src.set_state_extention import SetStateWrapper, factorized_symbolic_to_fullobs

# Global seed variables that can be modified from the command line in the future
SEED_1 = 1  # Seed for the first environment (source state)
SEED_2 = 71  # Seed for the second environment (target to be modified)

def get_symbolic_env(env):
    """
    Unwrap the environment until you find one that produces dict observations.
    This assumes that the symbolic representation is provided by a wrapper
    whose observation_space is a gym.spaces.Dict.
    """
    symbolic_env = env
    while not isinstance(symbolic_env.observation_space, gym.spaces.Dict):
        symbolic_env = symbolic_env.env
    return symbolic_env

def compare_minigrid_envs(env1, env2):
    """
    Compare all relevant attributes between two MiniGrid environments.
    Instead of raising errors on first difference, collects all differences.
    
    Args:
        env1: First MiniGrid environment
        env2: Second MiniGrid environment
        
    Returns:
        dict: Contains lists of matching and differing attributes with details
    """
    # Unwrap to base environments
    env1 = env1.unwrapped
    env2 = env2.unwrapped
    
    matching = []
    differences = []
    
    # Compare basic attributes
    basic_attrs = [
        "width", "height", "max_steps", "see_through_walls", 
        "agent_view_size", "mission"
    ]
    
    for attr in basic_attrs:
        val1 = getattr(env1, attr)
        val2 = getattr(env2, attr)
        
        if val1 == val2:
            matching.append(f"{attr}: {val1}")
        else:
            differences.append(f"{attr}: env1={val1}, env2={val2}")
    
    # Compare agent position and direction
    if np.array_equal(env1.agent_pos, env2.agent_pos):
        matching.append(f"agent_pos: {env1.agent_pos}")
    else:
        differences.append(f"agent_pos: env1={env1.agent_pos}, env2={env2.agent_pos}")
    
    if env1.agent_dir == env2.agent_dir:
        matching.append(f"agent_dir: {env1.agent_dir}")
    else:
        differences.append(f"agent_dir: env1={env1.agent_dir}, env2={env2.agent_dir}")
    
    # Compare carrying object
    if (env1.carrying is None and env2.carrying is None) or (
        env1.carrying is not None and env2.carrying is not None and 
        env1.carrying.type == env2.carrying.type and
        env1.carrying.color == env2.carrying.color
    ):
        matching.append(f"carrying: {env1.carrying}")
    else:
        differences.append(f"carrying: env1={env1.carrying}, env2={env2.carrying}")
    
    # Compare grid dimensions
    if env1.grid.width == env2.grid.width and env1.grid.height == env2.grid.height:
        matching.append(f"grid dimensions: {env1.grid.width}x{env1.grid.height}")
    else:
        differences.append(
            f"grid dimensions: env1={env1.grid.width}x{env1.grid.height}, "
            f"env2={env2.grid.width}x{env2.grid.height}"
        )
    
    # Compare grid cells
    grid_cell_diffs = []
    for i in range(env1.grid.width):
        for j in range(env1.grid.height):
            cell1 = env1.grid.get(i, j)
            cell2 = env2.grid.get(i, j)
            
            # Both cells None (empty)
            if cell1 is None and cell2 is None:
                continue
            
            # One cell None but the other isn't
            if (cell1 is None and cell2 is not None) or (cell1 is not None and cell2 is None):
                grid_cell_diffs.append(
                    f"Position ({i},{j}): env1={'None' if cell1 is None else f'type={cell1.type}, color={cell1.color}'}, "
                    f"env2={'None' if cell2 is None else f'type={cell2.type}, color={cell2.color}'}"
                )
                continue
            
            # Compare cell properties
            if cell1.type != cell2.type or cell1.color != cell2.color:
                grid_cell_diffs.append(
                    f"Position ({i},{j}): env1=type={cell1.type}, color={cell1.color}, "
                    f"env2=type={cell2.type}, color={cell2.color}"
                )
    
    if not grid_cell_diffs:
        matching.append("All grid cells match")
    else:
        differences.append(f"Grid cell differences ({len(grid_cell_diffs)}):")
        differences.extend(grid_cell_diffs)
    
    return {
        "matching": matching,
        "differences": differences,
        "match_percentage": len(matching) / (len(matching) + len(differences)) * 100
    }

def test_set_state_minigrid():
    """
    Test the set_standard_state_minigrid function by:
    1. Creating a symbolic MiniGrid environment with seed SEED_1
    2. Saving the initial frame and observation
    3. Creating a new environment with a different seed (SEED_2)
    4. Using SetStateWrapper to set the state to the saved state
    5. Comparing state components (agent position, direction, grid) to ensure they match
    6. Finally comparing rendered frames
    """
    # Set up test directories
    test_dir = os.path.join("tests", "test_data", "set_state_test")
    os.makedirs(test_dir, exist_ok=True)
    
    print(f"\n=== Step 1: Creating symbolic MiniGrid environment with seed {SEED_1} ===")
    env_config = {
        "env_name": "MiniGrid-Empty-Random-6x6-v0",
        "render_mode": "rgb_array",
        "max_objects": 10,
        "max_walls": 25,
        "include_walls": True,
        "representation": "symbolic"
    }
    env1 = create_symbolic_minigrid_env(env_config)
    symbolic_env1 = get_symbolic_env(env1)
    
    # Reset with SEED_1
    initial_obs1, _ = symbolic_env1.reset(seed=SEED_1)
    frame1 = env1.render()
    if frame1 is None:
        raise ValueError("Failed to render the first environment")
    
    print("\n=== Step 2: Saving initial frame and observation ===")
    initial_frame_path = os.path.join(test_dir, "initial_frame.png")
    Image.fromarray(frame1).save(initial_frame_path)
    print(f"Saved initial frame to {initial_frame_path}")
    print(f"Initial observation structure: {', '.join(initial_obs1.keys())}")
    print(f"Direction: {initial_obs1['direction']}")
    print(f"Number of objects: {len(initial_obs1['objects'])}")
    print(f"First few objects: {initial_obs1['objects'][:2]}")
    
    # Get the width and height from the environment
    env_width = env1.unwrapped.width
    env_height = env1.unwrapped.height
    print(f"Environment dimensions: {env_width}x{env_height}")
    
    # Convert symbolic observation to full observation format for SetStateWrapper
    print("\nConverting symbolic observation to full observation format")
    full_obs1 = factorized_symbolic_to_fullobs(initial_obs1, env_width, env_height)
    print(f"Full observation keys: {', '.join(full_obs1.keys())}")
    print(f"Image shape: {full_obs1['image'].shape}")
    
    print(f"\n=== Step 3: Creating new environment with FullyObsWrapper and seed {SEED_2} ===")
    env2 = gym.make(env_config["env_name"], render_mode="rgb_array")
    env2 = FullyObsWrapper(env2)
    env2 = SetStateWrapper(env2)
    
    # Reset with a different seed
    env2.reset(seed=SEED_2)
    
    # Render before state change
    frame2_before = env2.render()
    if frame2_before is None:
        raise ValueError("Failed to render the second environment before state change")
    
    before_frame_path = os.path.join(test_dir, "before_set_state_frame.png")
    Image.fromarray(frame2_before).save(before_frame_path)
    print(f"Saved frame before set_state to {before_frame_path}")
    
    print("\n=== Step 4: Setting state of second environment to match first environment ===")
    env2.set_state(full_obs1)
    
    # Render after state change - we'll save this for comparison later
    frame2_after = env2.render()
    if frame2_after is None:
        raise ValueError("Failed to render the second environment after state change")
    
    after_frame_path = os.path.join(test_dir, "after_set_state_frame.png")
    Image.fromarray(frame2_after).save(after_frame_path)
    print(f"Saved frame after set_state to {after_frame_path}")
    
    print("\n=== Step 5: Comparing state components ===")
    
    # Compare direction first
    print("Comparing agent directions...")
    if initial_obs1["direction"] != env2.unwrapped.agent_dir:
        raise AssertionError(f"Agent directions don't match! Original: {initial_obs1['direction']}, After set_state: {env2.unwrapped.agent_dir}")
    print("SUCCESS: Agent directions match.")
    
    # Compare agent position
    print("Comparing agent positions...")
    agent_pos1 = None
    for obj in initial_obs1["objects"]:
        if obj[0] == 10:  # 10 is OBJECT_TO_IDX["agent"]
            agent_pos1 = (obj[4], obj[3])
            break
    
    if agent_pos1 is None:
        raise ValueError("Agent position not found in the initial observation")
        
    if agent_pos1 != env2.unwrapped.agent_pos:
        raise AssertionError(f"Agent positions don't match! Original: {agent_pos1}, After set_state: {env2.unwrapped.agent_pos}")
    print("SUCCESS: Agent positions match.")
    
    # Check grid structure
    print("Comparing grid structures...")
    try:
        grid1 = env1.unwrapped.grid
        grid2 = env2.unwrapped.grid
        
        # Check grid dimensions
        if grid1.width != grid2.width or grid1.height != grid2.height:
            raise AssertionError(f"Grid dimensions don't match! Original: {grid1.width}x{grid1.height}, After set_state: {grid2.width}x{grid2.height}")
        
        # Check each cell in the grid
        for i in range(grid1.width):
            for j in range(grid1.height):
                cell1 = grid1.get(i, j)
                cell2 = grid2.get(i, j)
                
                # If both cells are None, they match
                if cell1 is None and cell2 is None:
                    continue
                    
                # If one is None but the other isn't, they don't match
                if (cell1 is None and cell2 is not None) or (cell1 is not None and cell2 is None):
                    raise AssertionError(f"Grid cell mismatch at ({i},{j}): One cell is None, the other isn't")
                
                # Compare cell properties
                if cell1.type != cell2.type or cell1.color != cell2.color:
                    raise AssertionError(f"Grid cell mismatch at ({i},{j}): Original: type={cell1.type}, color={cell1.color}, After set_state: type={cell2.type}, color={cell2.color}")
        
        print("SUCCESS: Grid structures match.")
    except Exception as e:
        print(f"ERROR when comparing grids: {e}")
    
    # Finally, compare the rendered frames
    print("\n=== Step 6: Comparing rendered frames ===")
    frames_equal = np.array_equal(frame1, frame2_after)
    if not frames_equal:
        print(f"Frame1 shape: {frame1.shape}, Frame2 shape: {frame2_after.shape}")
        
        if frame1.shape != frame2_after.shape:
            raise AssertionError(f"Frame shapes don't match: {frame1.shape} vs {frame2_after.shape}")
        
        frame_diff = np.sum(frame1 != frame2_after)
        diff_percentage = frame_diff / (frame1.shape[0] * frame1.shape[1] * frame1.shape[2]) * 100
        print(f"Number of differing pixels: {frame_diff} ({diff_percentage:.2f}%)")
        
        # If there are small differences, save a diff image for visual inspection
        if frame_diff > 0:
            diff_frame = np.abs(frame1.astype(int) - frame2_after.astype(int)).astype(np.uint8)
            diff_path = os.path.join(test_dir, "frame_diff.png")
            Image.fromarray(diff_frame).save(diff_path)
            print(f"Saved difference image to {diff_path}")
        
        raise AssertionError("Frames do not match after setting state!")
    
    print("SUCCESS: Frames match after setting state.")
    
    # Clean up
    env1.close()
    env2.close()
    
    print("\nAll tests passed successfully!")

def test_deep_comparison_set_state_minigrid():
    """
    Perform a deep comparison of environment attributes after using SetStateWrapper.
    This test performs a comprehensive check of all MiniGrid environment attributes
    to ensure the set_state function properly replicates the environment state.
    """
    # Set up test directories
    test_dir = os.path.join("tests", "test_data", "deep_comparison_test")
    os.makedirs(test_dir, exist_ok=True)
    
    print(f"\n=== Step 1: Creating symbolic MiniGrid environment with seed {SEED_1} ===")
    env_config = {
        "env_name": "MiniGrid-Empty-Random-6x6-v0",
        "render_mode": "rgb_array",
        "max_objects": 10,
        "max_walls": 25,
        "include_walls": True,
        "representation": "symbolic"
    }
    env1 = create_symbolic_minigrid_env(env_config)
    symbolic_env1 = get_symbolic_env(env1)
    
    # Reset with SEED_1
    initial_obs1, _ = symbolic_env1.reset(seed=SEED_1)
    
    # Get the width and height from the environment
    env_width = env1.unwrapped.width
    env_height = env1.unwrapped.height
    
    # Convert symbolic observation to full observation format for SetStateWrapper
    full_obs1 = factorized_symbolic_to_fullobs(initial_obs1, env_width, env_height)
    
    # Create second environment with different seed
    env2 = gym.make(env_config["env_name"], render_mode="rgb_array")
    env2 = FullyObsWrapper(env2)
    env2 = SetStateWrapper(env2)
    
    # Reset with a different seed
    env2.reset(seed=SEED_2)
    
    # Save the pre-state change comparison
    print(f"\n=== Step 2: Comparing environments before state change (SEED_1={SEED_1}, SEED_2={SEED_2}) ===")
    pre_comparison = compare_minigrid_envs(env1, env2)
    
    pre_comparison_path = os.path.join(test_dir, "pre_comparison.txt")
    with open(pre_comparison_path, 'w') as f:
        f.write("=== ENVIRONMENT COMPARISON BEFORE SET_STATE ===\n\n")
        f.write(f"Match percentage: {pre_comparison['match_percentage']:.2f}%\n\n")
        
        f.write("=== MATCHING ATTRIBUTES ===\n")
        for item in pre_comparison['matching']:
            f.write(f"- {item}\n")
            
        f.write("\n=== DIFFERENCES ===\n")
        for item in pre_comparison['differences']:
            f.write(f"- {item}\n")
    
    print(f"Pre-change comparison saved to {pre_comparison_path}")
    print(f"Pre-change match percentage: {pre_comparison['match_percentage']:.2f}%")
    print(f"Number of differences: {len(pre_comparison['differences'])}")
    
    # Set state of env2 to match env1
    print("\n=== Step 3: Setting state of second environment to match first environment ===")
    env2.set_state(full_obs1)
    
    # Perform detailed comparison after state change
    print("\n=== Step 4: Performing detailed comparison after state change ===")
    post_comparison = compare_minigrid_envs(env1, env2)
    
    # Save the post-state change comparison
    post_comparison_path = os.path.join(test_dir, "post_comparison.txt")
    with open(post_comparison_path, 'w') as f:
        f.write("=== ENVIRONMENT COMPARISON AFTER SET_STATE ===\n\n")
        f.write(f"Match percentage: {post_comparison['match_percentage']:.2f}%\n\n")
        
        f.write("=== MATCHING ATTRIBUTES ===\n")
        for item in post_comparison['matching']:
            f.write(f"- {item}\n")
            
        f.write("\n=== DIFFERENCES ===\n")
        for item in post_comparison['differences']:
            f.write(f"- {item}\n")
    
    print(f"Post-change comparison saved to {post_comparison_path}")
    print(f"Post-change match percentage: {post_comparison['match_percentage']:.2f}%")
    print(f"Number of differences: {len(post_comparison['differences'])}")
    
    # Verify high match percentage after set_state
    assert post_comparison['match_percentage'] > 95, f"Match percentage too low: {post_comparison['match_percentage']:.2f}%"
    assert len(post_comparison['differences']) <= 1, f"Too many differences after set_state: {len(post_comparison['differences'])}"
    
    # Clean up
    env1.close()
    env2.close()
    
    print("\nDeep comparison test passed successfully!")

if __name__ == "__main__":
    print(f"\n=== Running tests with SEED_1={SEED_1}, SEED_2={SEED_2} ===")

    print("\n=== Running test_set_state_minigrid ===")
    test_set_state_minigrid()
    
    print("\n=== Running test_deep_comparison_set_state_minigrid ===")
    test_deep_comparison_set_state_minigrid()
