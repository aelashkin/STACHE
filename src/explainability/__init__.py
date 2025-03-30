"""
Explainability module for STACHE.

This module contains tools and algorithms for explaining and understanding
model behavior, including Robustness Regions (RR) calculation.
"""

from src.explainability.bfs-rr import (
    bfs_rr,
    get_neighbors,
    get_symbolic_env,
    generate_rr_images
)

__all__ = [
    'bfs_rr',
    'get_neighbors',
    'get_symbolic_env',
    'generate_rr_images'
]
