import gymnasium as gym

from stache.envs.minigrid_ext.factory import create_minigrid_env
from stache.envs.taxi_ext.factory import create_taxi_env
# ────────────────────────────────────────────────────────────────────────────────
# Generic factory entry‑point for *all* envs
# ────────────────────────────────────────────────────────────────────────────────

def create_env(env_config: dict) -> gym.Env:
    """
    Generic environment factory. Decides which specialised builder to call based
    on `env_name`.  New domains can be added here with zero impact on callers.
    """
    env_name = env_config.get("env_name", "").lower()

    if "minigrid" in env_name:
        return create_minigrid_env(env_config)
    elif "taxi" in env_name:
        return create_taxi_env(env_config)
    else:  # Fallback – build raw Gymnasium env
        print(f"[WARN] No dedicated builder for '{env_name}', using gym.make")
        raise NotImplementedError(f"Environment '{env_name}' not supported. ")
        #TODO: return gym.make(env_config["env_name"], render_mode=env_config.get("render_mode"))
    
