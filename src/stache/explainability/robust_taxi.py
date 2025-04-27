import time
from collections import deque
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN

def translate_tuple_to_onehot(env: gym.Env, f: tuple[int, int, int, int]) -> np.ndarray:
    """
    Convert a Taxi-v3 factored state tuple f=(x,y,P,D) to a one-hot vector shape (1,500).
    Raises ValueError if any component is out of range.
    """
    x, y, P, D = f
    # Validate ranges
    if not (0 <= x <= 4):
        raise ValueError(f"x coordinate {x} out of range [0,4]")
    if not (0 <= y <= 4):
        raise ValueError(f"y coordinate {y} out of range [0,4]")
    if not (0 <= P <= 4):
        raise ValueError(f"passenger location {P} out of range [0,4]")
    if not (0 <= D <= 3):
        raise ValueError(f"destination {D} out of range [0,3]")
    # Encode and one-hot
    base_env = getattr(env, 'unwrapped', env)
    s = base_env.encode(x, y, P, D)  # type: ignore[attr-defined]
    vec = np.zeros((env.observation_space.n,), dtype=np.float32)  # type: ignore[attr-defined]
    vec[s] = 1.0
    return vec[None, :]

def get_neighbors_taxi(f: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]]:
    """
    Generate all atomic neighbors of Taxi state f=(x,y,P,D):
      - x ±1 (within [0,4])
      - y ±1 (within [0,4])
      - P changes to any other {0..4}
      - D changes to any other {0..3}
    Return list in stable order: moves, passenger changes, destination changes.
    """
    x, y, P, D = f
    neighbors: list[tuple[int, int, int, int]] = []
    # Row moves
    for dx in (-1, 1):
        nx = x + dx
        if 0 <= nx <= 4:
            neighbors.append((nx, y, P, D))
    # Col moves
    for dy in (-1, 1):
        ny = y + dy
        if 0 <= ny <= 4:
            neighbors.append((x, ny, P, D))
    # Passenger location changes
    for p in range(5):
        if p != P:
            neighbors.append((x, y, p, D))
    # Destination changes
    for d in range(4):
        if d != D:
            neighbors.append((x, y, P, d))
    return neighbors

def compute_rr_taxi(
    seed_f: tuple[int, int, int, int],
    model: DQN,
    env: gym.Env,
    precomputed_sa: dict[int, int] | None = None,
) -> dict[str, any]:
    """
    Breadth-first compute robustness region from seed_f under model's deterministic policy.

    Returns dict with keys:
      - rr_tuple_set: set of tuples in the region
      - rr_depths: dict mapping tuple->BFS depth
      - initial_action: int a* chosen at seed
      - stats: dict with region_size, visited, opened, elapsed
    """
    start = time.perf_counter()
    # Determine initial action
    seed_vec = translate_tuple_to_onehot(env, seed_f)
    a_star, _ = model.predict(seed_vec, deterministic=True)
    # Unwrap numpy
    if isinstance(a_star, (np.ndarray, list, tuple)):
        a_star = int(a_star[0])
    # BFS init
    queue = deque([(seed_f, 0)])
    visited: set[tuple[int, int, int, int]] = {seed_f}
    rr_tuple_set: set[tuple[int, int, int, int]] = set()
    rr_depths: dict[tuple[int, int, int, int], int] = {}
    opened = 0
    # Traverse
    while queue:
        f, d = queue.popleft()
        opened += 1
        # Get action
        s_env = getattr(env, 'unwrapped', env)
        s = s_env.encode(*f)  # type: ignore[attr-defined]
        if precomputed_sa is not None:
            if s not in precomputed_sa:
                raise KeyError(f"State {s} not in precomputed mapping")
            a = precomputed_sa[s]
        else:
            vec = translate_tuple_to_onehot(env, f)
            a_tmp, _ = model.predict(vec, deterministic=True)
            a = int(a_tmp[0]) if isinstance(a_tmp, (np.ndarray, list, tuple)) else int(a_tmp)
        if a == a_star:
            rr_tuple_set.add(f)
            rr_depths[f] = d
            # Expand neighbors
            for n in get_neighbors_taxi(f):
                if n not in visited:
                    visited.add(n)
                    queue.append((n, d + 1))
    elapsed = time.perf_counter() - start
    stats = {
        "region_size": len(rr_tuple_set),
        "visited": len(visited),
        "opened": opened,
        "elapsed": elapsed,
    }
    return {
        "rr_tuple_set": rr_tuple_set,
        "rr_depths": rr_depths,
        "initial_action": a_star,
        "stats": stats,
    }
