"""
Environment module for STACHE.

This module contains environment utilities, wrappers, and extensions
for working with MiniGrid environments.
"""

from src.envs.environment_utils import create_minigrid_env
from src.envs.set_state_extention import (
    set_standard_state_minigrid,
    factorized_symbolic_to_fullobs
)

__all__ = [
    'create_minigrid_env',
    'set_standard_state_minigrid',
    'factorized_symbolic_to_fullobs'
]
