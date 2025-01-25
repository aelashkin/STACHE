
import optuna


def sample_a2c_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for A2C hyperparameters."""
    gamma = 1.0 - trial.suggest_float("gamma", 0.0001, 0.1, log=True)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 5.0, log=True)
    gae_lambda = 1.0 - trial.suggest_float("gae_lambda", 0.001, 0.2, log=True)
    n_steps = 2 ** trial.suggest_int("exponent_n_steps", 3, 10)
    learning_rate = trial.suggest_float("lr", 1e-5, 1, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    ortho_init = trial.suggest_categorical("ortho_init", [False, True])
    net_arch = trial.suggest_categorical("net_arch", ["tiny", "small"])
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    trial.set_user_attr("gamma_", gamma)
    trial.set_user_attr("gae_lambda_", gae_lambda)
    trial.set_user_attr("n_steps", n_steps)

    net_arch = {"pi": [64], "vf": [64]} if net_arch == "tiny" else {"pi": [64, 64], "vf": [64, 64]}
    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn]

    return {
        "n_steps": n_steps,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "max_grad_norm": max_grad_norm,
        "policy_kwargs": {
            "net_arch": net_arch,
            "activation_fn": activation_fn,
            "ortho_init": ortho_init,
        },
    }

def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for PPO hyperparameters."""
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)

    n_steps = 2 ** trial.suggest_int("exponent_n_steps", 3, 10)
    # Filter batch_size options to ensure compatibility with n_steps
    valid_batch_sizes = [b for b in [32, 64, 128, 256] if n_steps % b == 0]
    if not valid_batch_sizes:
        raise ValueError(f"No valid batch_size for n_steps={n_steps}. Adjust ranges.")
    
    # Sample compatible batch_size
    batch_size = trial.suggest_categorical("batch_size", valid_batch_sizes)

    n_epochs = trial.suggest_int("n_epochs", 3, 12)

    gamma = 1.0 - trial.suggest_float("gamma", 0.0001, 0.2, log=True)
    gae_lambda = 1.0 - trial.suggest_float("gae_lambda", 0.05, 0.2, log=True) #екн keep 0.95 as default

    ent_coef = trial.suggest_float("ent_coef", 1e-8, 1e-2, log=True)
    vf_coef = trial.suggest_float("vf_coef", 0.1, 1.0)

    #check on these

    trial.set_user_attr("n_steps", n_steps)    
    trial.set_user_attr("gamma_", gamma)
    trial.set_user_attr("gae_lambda_", gae_lambda)


    return {
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
    }
