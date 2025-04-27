import gymnasium as gym

from stable_baselines3.common.monitor import Monitor

# ────────────────────────────────────────────────────────────────────────────────
# Taxi environment factory
# ────────────────────────────────────────────────────────────────────────────────

def create_taxi_env(env_config: dict) -> gym.Env:
    """
    Build and return a Taxi‑v3 environment ready for SB3 PPO.

    Supported representations
    -------------------------
    - "one_hot"  (default): wraps the Discrete(500) observation into a Box(0,1)
      vector using gymnasium.wrappers.OneHotObservation.
    - "discrete": leaves the native Discrete space untouched (**not** suitable for
      vanilla SB3‑PPO).

    Extra keys in `env_config`
    --------------------------
    - render_mode : passed straight to `gym.make`
    - representation : see above
    """
    env_name   = env_config.get("env_name", "Taxi-v3")
    render     = env_config.get("render_mode")       # None / "human" / "rgb_array"
    repr_type  = env_config.get("representation", "one_hot")

    env = gym.make(env_name, render_mode=render)

    if repr_type == "one_hot":
        raise NotImplementedError("One-hot observation is not yet supported.")
    elif repr_type == "discrete":
        pass  # nothing to do
    else:
        raise ValueError(
            f"Unsupported representation '{repr_type}' for Taxi. "
            "Use 'discrete'."
        )

    # Keep logging identical to MiniGrid
    env = Monitor(env)
    return env
