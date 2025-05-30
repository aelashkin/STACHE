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
    # Use base_env.observation_space.n for the size of the one-hot vector
    vec = np.zeros((base_env.observation_space.n,), dtype=np.float32)  # type: ignore[attr-defined]
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
    env: gym.Env, # This env is expected to be the base_env for translate_tuple_to_onehot calls within
    precomputed_sa: dict[int, int] | None = None,
) -> dict[str, any]:
    """
    Breadth-first compute robustness region from seed_f under model's deterministic policy.
    Also identifies counterfactual states encountered during the search.

    Returns dict with keys:
      - rr_tuple_set: set of tuples in the region
      - rr_depths: dict mapping tuple->BFS depth
      - counterfactuals: list of (tuple_state, depth) for states where action != initial_action
      - initial_action: int a* chosen at seed
      - stats: dict with region_size, visited, opened, elapsed
    """
    start = time.perf_counter()
    # Determine initial action
    # translate_tuple_to_onehot expects base_env, model.predict expects one-hot vector
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
    counterfactuals: list[tuple[tuple[int, int, int, int], int]] = [] # Added
    opened = 0
    # Traverse
    while queue:
        f, d = queue.popleft()
        opened += 1
        # Get action
        # env here is base_env, s_env will also be base_env
        s_env = getattr(env, 'unwrapped', env)
        s = s_env.encode(*f)  # type: ignore[attr-defined]
        current_action: int
        if precomputed_sa is not None:
            if s not in precomputed_sa:
                # It's possible a state is visited that's outside the precomputed set if precomputation was partial.
                # In this case, we must use the model.
                vec = translate_tuple_to_onehot(env, f)
                a_tmp, _ = model.predict(vec, deterministic=True)
                current_action = int(a_tmp[0]) if isinstance(a_tmp, (np.ndarray, list, tuple)) else int(a_tmp)
            else:
                current_action = precomputed_sa[s]
        else:
            # translate_tuple_to_onehot expects base_env
            vec = translate_tuple_to_onehot(env, f)
            a_tmp, _ = model.predict(vec, deterministic=True)
            current_action = int(a_tmp[0]) if isinstance(a_tmp, (np.ndarray, list, tuple)) else int(a_tmp)

        if current_action == a_star:
            rr_tuple_set.add(f)
            rr_depths[f] = d
            # Expand neighbors only if part of the RR
            for n_neighbor in get_neighbors_taxi(f):
                if n_neighbor not in visited:
                    visited.add(n_neighbor)
                    queue.append((n_neighbor, d + 1))
        else:
            # This state f is a counterfactual because its action is different from a_star
            counterfactuals.append((f, d))
            pass

    elapsed = time.perf_counter() - start
    stats = {
        "region_size": len(rr_tuple_set),
        "visited": len(visited), # Total unique states put on queue
        "opened": opened, # Total states popped from queue
        "elapsed": elapsed,
    }
    return {
        "rr_tuple_set": rr_tuple_set,
        "rr_depths": rr_depths,
        "counterfactuals": counterfactuals, # Added
        "initial_action": a_star,
        "stats": stats,
    }
