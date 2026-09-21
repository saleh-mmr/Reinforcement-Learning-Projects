import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from learning.trainer import Trainer



hyperparams = {
    "discount_factor": 0.99,
    "batch_size": 1000,
    "warmup_size": 1000,
    "network_size": 40,
    "max_steps_per_episode": 100,
    "max_episodes": 8000,
    "epsilon_max": 1.0,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.00001,
    "memory_capacity": 10000,
    "g_ap": 18.0,
    "g_p": 15.0,
    "shift_parameter": 20,
    "g_bias": 30.0,
    "noise_stddev": 0.001,
    "CP_pole_length_1": 5.0,
    "CP_pole_mass_1": 1.0,
    "CP_pole_length_2": 10.0,
    "CP_pole_mass_2": 2.0,
    "CP_pole_length_3": 20.0,
    "CP_pole_mass_3": 5.0,
    "regularization_C": 0.00000001,
}

if __name__ == "__main__":
    base_folder = SCRIPT_DIR / "three_problems"
    run_folder = "run_2026-06-05_08-05-50"

    folder = base_folder / run_folder

    weigh_step = 377060
    cartpole = 0
    correspond_weight = 0

    if correspond_weight == 0:
        keyword = "MC1"
    elif correspond_weight == 1:
        keyword = "MC2"
    else:
        keyword = "MC3"

    path = folder / f"{keyword}_{weigh_step}.pth"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    num_tests = 5000
    trainer = Trainer(hyperparams, seed=None, folder=folder)

    test_log = trainer.test(
        model_path=path,
        num_tests=num_tests,
        cartpole=cartpole,
    )

    result_path = f"test_log_{keyword}.csv"
    test_log.to_csv(folder / result_path, index=False)
