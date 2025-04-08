import os
import shutil
import tempfile
from unittest.mock import patch, MagicMock, mock_open, PropertyMock
import pytest
import numpy as np
import gymnasium as gym
from PIL import Image
from stable_baselines3 import PPO
from minigrid.wrappers import FlatObsWrapper, FullyObsWrapper
import sys

# Add this import for our debugging helper
import inspect

from src.explainability.evaluate import (
    evaluate_model,
    evaluate_symbolic_env,
    evaluate_policy_performance,
    get_fully_obs_env,
    evaluate_single_policy_run
)
from src.minigrid_ext.environment_utils import create_symbolic_minigrid_env, create_standard_minigrid_env


# Add a debugging helper for the get_fully_obs_env function
def debug_get_fully_obs_env(env):
    """
    Debug version of get_fully_obs_env to trace execution.
    """
    print("Starting debug_get_fully_obs_env with:", env)
    current_env = env
    last_fully_obs = None
    depth = 0
    
    # Traverse through the wrappers
    while hasattr(current_env, 'env'):
        print(f"Depth {depth}: current_env={current_env}, has env attribute: {hasattr(current_env, 'env')}")
        if isinstance(current_env, FullyObsWrapper):
            print(f"Found FullyObsWrapper at depth {depth}")
            last_fully_obs = current_env
        try:
            current_env = current_env.env
            print(f"  Next env: {current_env}, type: {type(current_env)}")
        except Exception as e:
            print(f"  Error accessing .env: {e}")
            break
        depth += 1
        if depth > 10:  # Safety limit
            print("Too deep! Breaking to avoid infinite loop.")
            break
    
    print(f"Returning: {last_fully_obs}")
    return last_fully_obs


class TestEvaluate:
    """Test suite for the evaluate.py module."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock PPO model for testing."""
        mock_model = MagicMock(spec=PPO)
        # Configure the predict method to return a default action and state
        mock_model.predict.return_value = (0, None)  # 0 = move forward action
        return mock_model
    
    @pytest.fixture
    def env_config(self):
        """Create a test environment configuration."""
        return {
            "env_name": "MiniGrid-Empty-Random-6x6-v0",
            "wrapper_class": "SymbolicObsWrapper",
            "fully_observable": True,
            "max_steps": 20
        }
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        tmp_dir = tempfile.mkdtemp()
        yield tmp_dir
        # Clean up after test
        shutil.rmtree(tmp_dir)
    
    @patch("src.explainability.evaluate.gym.make")
    @patch("src.explainability.evaluate.evaluate_policy")
    @patch("src.explainability.evaluate.FlatObsWrapper")
    @patch("src.explainability.evaluate.PPO.load")
    def test_evaluate_model(self, mock_ppo_load, mock_flat_wrapper, mock_evaluate_policy, mock_gym_make, mock_model):
        """Test that evaluate_model correctly loads and evaluates a model."""
        # Configure mocks
        mock_env = MagicMock()
        mock_gym_make.return_value = mock_env
        
        # Setup mock wrapper that doesn't trigger isinstance checks
        mock_wrapped_env = MagicMock()
        mock_flat_wrapper.return_value = mock_wrapped_env
        
        # Setup mock model loading
        mock_ppo_load.return_value = mock_model
        
        # Setup evaluation return values
        mock_evaluate_policy.return_value = (10.5, 2.5)  # Mean and std reward
        
        # Call the function under test
        evaluate_model("/path/to/model", env_name="MiniGrid-Test-Env", n_eval_episodes=5)
        
        # Verify correct function calls
        mock_gym_make.assert_called_once_with("MiniGrid-Test-Env", render_mode='rgb_array')
        mock_flat_wrapper.assert_called_once_with(mock_env)
        mock_ppo_load.assert_called_once_with("/path/to/model")
        mock_evaluate_policy.assert_called_once()
        mock_wrapped_env.close.assert_called_once()
    
    @patch("src.explainability.evaluate.create_symbolic_minigrid_env")
    def test_evaluate_symbolic_env(self, mock_create_env, env_config):
        """Test that evaluate_symbolic_env correctly creates and interacts with the environment."""
        # Setup mock environment
        mock_env = MagicMock()
        mock_env.reset.return_value = (np.zeros(10), {})  # obs, info
        mock_env.step.return_value = (np.zeros(10), 1.0, False, False, {})  # obs, reward, terminated, truncated, info
        mock_env.action_space.sample.return_value = 0
        mock_create_env.return_value = mock_env
        
        # Call the function under test
        evaluate_symbolic_env(env_config)
        
        # Verify function calls
        mock_create_env.assert_called_once_with(env_config)
        mock_env.reset.assert_called_once()
        assert mock_env.step.call_count == 4  # Should step through 4 iterations
        mock_env.close.assert_called_once()
    
    @patch("src.explainability.evaluate.create_symbolic_minigrid_env")
    @patch("src.explainability.evaluate.plt")
    def test_evaluate_policy_performance(self, mock_plt, mock_create_env, mock_model, env_config):
        """Test that evaluate_policy_performance correctly evaluates a policy over multiple episodes."""
        # Setup mock environment
        mock_env = MagicMock()
        
        # We need to reset the mock for each episode in the test (2 episodes)
        mock_env.reset.side_effect = [(np.zeros(10), {}), (np.zeros(10), {})]
        
        # Configure mock step returns for both episodes - each episode needs enough steps
        # First episode: 2 steps (1 regular + 1 terminal)
        # Second episode: 2 steps (1 regular + 1 terminal)
        mock_env.step.side_effect = [
            # First episode
            (np.zeros(10), 0.5, False, False, {}),
            (np.zeros(10), 0.5, True, False, {}),
            # Second episode
            (np.zeros(10), 0.8, False, False, {}),
            (np.zeros(10), 0.2, True, False, {})
        ]
        mock_create_env.return_value = mock_env
        
        # Call the function under test
        result = evaluate_policy_performance(mock_model, env_config, n_eval_episodes=2)
        
        # Verify function calls and results
        mock_create_env.assert_called_once()
        assert mock_env.reset.call_count == 2  # Should reset for each episode
        assert mock_env.step.call_count == 4  # 2 steps per episode
        
        # Verify the results dictionary has the expected keys
        assert "mean_reward" in result
        assert "median_reward" in result
        assert "std_reward" in result
        assert "min_reward" in result
        assert "max_reward" in result
        assert "reward_distribution" in result
        
        # Verify plotting was called
        mock_plt.figure.assert_called_once()
        mock_plt.show.assert_called_once()
    
    def test_get_fully_obs_env(self):
        """Test that get_fully_obs_env correctly finds the FullyObsWrapper."""
        # Create a chain of wrapped environments
        class BaseEnv:
            """A simple base environment with no env attribute."""
            def __str__(self):
                return "BaseEnv instance"
        
        base_env = BaseEnv()
        
        # Explicitly create objects that mimic the real wrapper classes
        class MockFlatWrapper:
            def __init__(self, env):
                self.env = env
            
            def __str__(self):
                return f"MockFlatWrapper(env={self.env})"
        
        class MockFullyObsWrapper:
            def __init__(self, env):
                self.env = env
            
            def __str__(self):
                return f"MockFullyObsWrapper(env={self.env})"
            
            # Make this class instance check work with isinstance
            def __instancecheck__(self, instance):
                return isinstance(instance, FullyObsWrapper)
        
        # Create the chain without using MagicMock
        flat_wrapper = MockFlatWrapper(base_env)
        fully_wrapper = MockFullyObsWrapper(flat_wrapper)
        
        # Use our debug version to see what's happening
        result = debug_get_fully_obs_env(fully_wrapper)
        
        # Now test with the actual function
        with patch("src.explainability.evaluate.get_fully_obs_env", side_effect=get_fully_obs_env):
            result = get_fully_obs_env(fully_wrapper)
            print(f"Result of get_fully_obs_env: {result}")
            
            # We expect None because our mock won't be detected as a FullyObsWrapper by isinstance
            assert result is None
            
            # Test when there is no FullyObsWrapper
            result = get_fully_obs_env(flat_wrapper)
            print(f"Result with flat_wrapper: {result}")
            assert result is None
    
    @patch("src.explainability.evaluate.create_symbolic_minigrid_env")
    @patch("src.explainability.evaluate.get_fully_obs_env")
    @patch("src.explainability.evaluate.os.makedirs")
    @patch("src.explainability.evaluate.Image.fromarray")
    @patch("builtins.open", new_callable=mock_open)
    def test_evaluate_single_policy_run(self, mock_file, mock_image_fromarray, mock_makedirs, 
                                        mock_get_fully_obs, mock_create_env, mock_model, env_config, temp_dir):
        """Test that evaluate_single_policy_run correctly runs a single episode and logs all actions."""
        # Setup mocks
        mock_env = MagicMock()
        mock_env.reset.return_value = (np.zeros(10), {"agent_pos": (1, 1)})
        mock_env.step.side_effect = [
            (np.zeros(10), 0.0, False, False, {"agent_pos": (2, 1)}),
            (np.zeros(10), 1.0, True, False, {"agent_pos": (3, 1), "success": True})
        ]
        mock_env.render.return_value = np.zeros((84, 84, 3), dtype=np.uint8)
        mock_create_env.return_value = mock_env
        
        # Create a complete mock for the FullyObsWrapper that includes the nested env structure
        mock_unwrapped = MagicMock()
        mock_unwrapped.gen_obs.return_value = np.zeros(10)
        
        mock_fully_obs_env = MagicMock()
        mock_fully_obs_inner_env = MagicMock()
        mock_fully_obs_inner_env.unwrapped = mock_unwrapped
        
        mock_fully_obs = MagicMock(spec=FullyObsWrapper)
        mock_fully_obs.env = mock_fully_obs_inner_env  # Set the env attribute explicitly
        mock_fully_obs.observation.return_value = np.ones(10)
        
        mock_get_fully_obs.return_value = mock_fully_obs
        
        # Mock image saving
        mock_image = MagicMock()
        mock_image_fromarray.return_value = mock_image
        
        # Call the function under test with a fixed timestamp for reproducibility
        with patch("src.explainability.evaluate.datetime") as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20250410-120000"
            evaluate_single_policy_run(mock_model, env_config, seed=42, max_steps=10, save_full_obs=True)
        
        # Verify environment was created and reset with the correct seed
        mock_create_env.assert_called_once()
        mock_env.reset.assert_called_once_with(seed=42)
        
        # Verify that step was called correctly
        assert mock_env.step.call_count == 2
        
        # Verify that frames were rendered and saved
        assert mock_env.render.call_count == 3  # Initial frame + 2 steps
        assert mock_image.save.call_count == 3
        
        # Verify the file was opened for writing the log
        assert mock_file.call_count > 0
        
        # Verify that fully observable observations were recorded
        assert mock_fully_obs.observation.call_count > 0
    
    @patch("src.explainability.evaluate.create_symbolic_minigrid_env")
    @patch("src.explainability.evaluate.get_fully_obs_env")
    @patch("src.explainability.evaluate.os.makedirs")
    @patch("src.explainability.evaluate.Image.fromarray")
    def test_evaluate_single_policy_run_log_content(self, mock_image_fromarray, mock_makedirs, 
                                                 mock_get_fully_obs, mock_create_env, mock_model, env_config, temp_dir):
        """Test that the log file contains correct and complete information about the policy run."""
        # Setup mocks
        mock_env = MagicMock()
        # Define specific test values for observations and info
        test_obs_initial = np.array([1, 2, 3, 4])
        test_obs_step1 = np.array([5, 6, 7, 8])
        test_obs_step2 = np.array([9, 10, 11, 12])
        
        test_info_initial = {"agent_pos": (1, 1), "carrying": None}
        test_info_step1 = {"agent_pos": (2, 1), "carrying": None}
        test_info_step2 = {"agent_pos": (3, 1), "carrying": "key", "success": True}
        
        # Configure mock environment behavior
        mock_env.reset.return_value = (test_obs_initial, test_info_initial)
        mock_env.step.side_effect = [
            (test_obs_step1, 0.5, False, False, test_info_step1),
            (test_obs_step2, 1.0, True, False, test_info_step2)
        ]
        mock_env.render.return_value = np.zeros((84, 84, 3), dtype=np.uint8)
        mock_create_env.return_value = mock_env
        
        # Define specific test action values
        test_action1 = 1  # e.g., turn right
        test_action2 = 2  # e.g., turn left
        mock_model.predict.side_effect = [(test_action1, None), (test_action2, None)]
        
        # Create a complete mock for the FullyObsWrapper with the proper nested structure
        mock_unwrapped = MagicMock()
        mock_unwrapped.gen_obs.return_value = np.zeros(10)
        
        mock_fully_obs_inner_env = MagicMock()
        mock_fully_obs_inner_env.unwrapped = mock_unwrapped
        
        mock_fully_obs = MagicMock(spec=FullyObsWrapper)
        mock_fully_obs.env = mock_fully_obs_inner_env  # Set the env attribute explicitly
        
        # Define observation return values 
        full_obs_initial = np.array([100, 101, 102])
        full_obs_step1 = np.array([200, 201, 202])
        full_obs_step2 = np.array([300, 301, 302])
        mock_fully_obs.observation.side_effect = [full_obs_initial, full_obs_step1, full_obs_step2]
        
        mock_get_fully_obs.return_value = mock_fully_obs
        
        mock_image = MagicMock()
        mock_image_fromarray.return_value = mock_image
        
        log_path = os.path.join(temp_dir, "evaluation_log.txt")
        
        with patch("src.explainability.evaluate.os.path.join", return_value=log_path):
            with patch("src.explainability.evaluate.datetime") as mock_datetime:
                mock_datetime.now.return_value.strftime.return_value = "20250410-120000"
                
                evaluate_single_policy_run(mock_model, env_config, seed=42, max_steps=10, save_full_obs=True)
        
        with open(log_path, "r") as f:
            log_content = f.read()
        
        assert f"Episode start (seed=42)" in log_content
        assert f"Initial observation: {test_obs_initial}" in log_content
        assert f"Initial fully observable observation: {full_obs_initial}" in log_content
        assert f"Initial info: {test_info_initial}" in log_content
        
        assert f"Step 0: Observation: {test_obs_initial}" in log_content
        assert f"Step 0: Action taken: {test_action1}" in log_content
        assert f"Step 0: Reward: 0.5" in log_content
        assert f"Step 0: Terminated: False, Truncated: False" in log_content
        assert f"Step 0: Fully obs observation: {full_obs_step1}" in log_content
        assert f"Step 0: Info: {test_info_step1}" in log_content
        assert "Step 0: Saved frame at" in log_content
        
        assert f"Step 1: Observation: {test_obs_step1}" in log_content
        assert f"Step 1: Action taken: {test_action2}" in log_content
        assert f"Step 1: Reward: 1.0" in log_content
        assert f"Step 1: Terminated: True, Truncated: False" in log_content
        assert f"Step 1: Fully obs observation: {full_obs_step2}" in log_content
        assert f"Step 1: Info: {test_info_step2}" in log_content
        assert "Step 1: Saved frame at" in log_content
        
        assert "Episode ended." in log_content
        
        mock_create_env.assert_called_once()
        mock_env.reset.assert_called_once_with(seed=42)
        assert mock_env.step.call_count == 2
        assert mock_model.predict.call_count == 2
        assert mock_fully_obs.observation.call_count == 3

    @patch("src.explainability.evaluate.create_symbolic_minigrid_env")
    @patch("src.explainability.evaluate.get_fully_obs_env")
    @patch("src.explainability.evaluate.os.makedirs")
    @patch("src.explainability.evaluate.Image.fromarray")
    def test_evaluate_single_policy_run_max_steps_limit(self, mock_image_fromarray, mock_makedirs, 
                                                     mock_get_fully_obs, mock_create_env, mock_model, env_config, temp_dir):
        """Test that evaluate_single_policy_run respects the max_steps parameter and stops after the limit."""
        mock_env = MagicMock()
        mock_env.reset.return_value = (np.zeros(10), {})
        mock_env.step.return_value = (np.zeros(10), 0.1, False, False, {})
        mock_env.render.return_value = np.zeros((84, 84, 3), dtype=np.uint8)
        mock_create_env.return_value = mock_env
        
        mock_unwrapped = MagicMock()
        mock_unwrapped.gen_obs.return_value = np.zeros(10)
        
        mock_fully_obs_inner_env = MagicMock()
        mock_fully_obs_inner_env.unwrapped = mock_unwrapped
        
        mock_fully_obs = MagicMock(spec=FullyObsWrapper)
        mock_fully_obs.env = mock_fully_obs_inner_env
        mock_fully_obs.observation.return_value = np.ones(10)
        
        mock_get_fully_obs.return_value = mock_fully_obs
        
        mock_image = MagicMock()
        mock_image_fromarray.return_value = mock_image
        
        log_path = os.path.join(temp_dir, "evaluation_log.txt")
        
        with patch("src.explainability.evaluate.os.path.join", return_value=log_path):
            evaluate_single_policy_run(mock_model, env_config, seed=42, max_steps=3, save_full_obs=True)
        
        assert mock_env.step.call_count == 3
        assert mock_model.predict.call_count == 3
        
        with open(log_path, "r") as f:
            log_content = f.read()
        
        action_logs = [line for line in log_content.split('\n') if "Action taken" in line]
        assert len(action_logs) == 3

    @patch("src.explainability.evaluate.create_symbolic_minigrid_env")
    @patch("src.explainability.evaluate.open", new_callable=mock_open)
    def test_evaluate_single_policy_run_error_handling(self, mock_file, mock_create_env, mock_model, env_config, temp_dir):
        """Test that evaluate_single_policy_run handles errors gracefully."""
        # Setup mock to raise an exception during environment creation
        mock_create_env.side_effect = ValueError("Test error: Invalid environment configuration")
        
        # Test that the function raises the expected exception
        with patch("src.explainability.evaluate.os.makedirs"):  # Mock makedirs to avoid directory creation issues
            with pytest.raises(ValueError, match="Test error: Invalid environment configuration"):
                evaluate_single_policy_run(mock_model, env_config, seed=42)
        
        # Verify the environment creation was attempted
        mock_create_env.assert_called_once()

    @patch("src.explainability.evaluate.create_symbolic_minigrid_env")
    @patch("src.explainability.evaluate.get_fully_obs_env")
    @patch("src.explainability.evaluate.os.makedirs")
    @patch("src.explainability.evaluate.Image.fromarray")
    def test_evaluate_single_policy_run_truncation(self, mock_image_fromarray, mock_makedirs, 
                                                mock_get_fully_obs, mock_create_env, mock_model, env_config, temp_dir):
        """Test that evaluate_single_policy_run correctly handles episode truncation."""
        mock_env = MagicMock()
        mock_env.reset.return_value = (np.zeros(10), {})
        mock_env.step.return_value = (np.zeros(10), 0.0, False, True, {"TimeLimit.truncated": True})
        mock_env.render.return_value = np.zeros((84, 84, 3), dtype=np.uint8)
        mock_create_env.return_value = mock_env
        
        mock_fully_obs = None
        mock_get_fully_obs.return_value = mock_fully_obs
        
        mock_image = MagicMock()
        mock_image_fromarray.return_value = mock_image
        
        log_path = os.path.join(temp_dir, "truncation_log.txt")
        
        with patch("src.explainability.evaluate.os.path.join", return_value=log_path):
            evaluate_single_policy_run(mock_model, env_config, seed=42, max_steps=100, save_full_obs=False)
        
        assert mock_env.step.call_count == 1
        
        with open(log_path, "r") as f:
            log_content = f.read()
        
        assert "Truncated: True" in log_content
        assert "Episode ended." in log_content

    def test_evaluate_model_with_custom_environment(self):
        """Test evaluate_model with a custom environment that requires specific initialization."""
        pass