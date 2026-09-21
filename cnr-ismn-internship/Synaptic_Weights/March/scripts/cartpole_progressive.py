import sys
from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from learning.trainer import Trainer


def load_hyperparams(folder):
    row = pd.read_csv(folder / "details_log.csv").iloc[0]

    return {
        "discount_factor": float(row["discount_factor"]),
        "batch_size": int(row["batch_size"]),
        "warmup_size": int(row["warmup_size"]),
        "network_size": int(row["network_size"]),
        "max_steps_per_episode": int(row["max_steps_per_episode"]),
        "max_episodes": int(row["max_episodes"]),
        "epsilon_max": 1.0,
        "epsilon_min": 0.01,
        "epsilon_decay": float(row["epsilon_decay"]),
        "memory_capacity": int(row["memory_size"]),

        "g_bias": float(row["G_bias_coefficient"]),
        "g_ap": float(row["G_ap_coefficient"]),
        "g_p": float(row["G_p_coefficient"]),
        "shift_parameter": float(row["shift parameter"]),
        "noise_stddev": float(row["noise_stddev"]),

        "CP_pole_length_1": float(row["CP_pole_length_1"]),
        "CP_pole_mass_1": float(row["CP_pole_mass_1"]),
        "CP_pole_length_2": float(row["CP_pole_length_2"]),
        "CP_pole_mass_2": float(row["CP_pole_mass_2"]),
        "CP_pole_length_3": float(row["CP_pole_length_3"]),
        "CP_pole_mass_3": float(row["CP_pole_mass_3"]),

        "regularization_C": 0.00000001,
    }


def get_step_from_checkpoint(path):
    """
    Example:
        MC1_76329.pth -> 76329
    """
    return int(path.stem.split("_")[-1])


def get_checkpoint_path(folder, weight_selector, step):
    """
    weight_selector:
        0 -> MC1
        1 -> MC2
        2 -> MC3
    """
    keyword = ["MC1", "MC2", "MC3"][weight_selector]
    return folder / f"{keyword}_{step}.pth"


def run_one_checkpoint_test(
    trainer,
    checkpoint_path,
    num_tests,
    cartpole_selector,
    weight_selector,
    threshold_reward,
):
    """
    Tests one checkpoint and returns a summary dictionary.
    """

    test_log = trainer.test(
        model_path=checkpoint_path,
        num_tests=num_tests,
        cartpole=cartpole_selector,
    )

    mean_reward = test_log["reward"].mean()
    std_reward = test_log["reward"].std(ddof=0)
    min_reward = test_log["reward"].min()
    max_reward = test_log["reward"].max()

    passed = mean_reward >= threshold_reward

    step = get_step_from_checkpoint(checkpoint_path)

    print(
        f"Step {step} | "
        f"cartpole_{cartpole_selector} with weight_{weight_selector} | "
        f"mean={mean_reward:.6f} | "
        f"std={std_reward:.6f} | "
        f"threshold={threshold_reward:.2f} | "
        f"passed={passed}"
    )

    return {
        "step": step,
        "checkpoint": checkpoint_path.name,
        "cartpole_selector": cartpole_selector,
        "weight_selector": weight_selector,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "min_reward": min_reward,
        "max_reward": max_reward,
        "threshold_reward": threshold_reward,
        "passed": passed,
    }


def main():
    # ---------------------------------------------------------
    # Settings
    # ---------------------------------------------------------
    folder_name = "run_2026-06-26_00-54-50"
    folder = SCRIPT_DIR / "three_problems" / folder_name

    num_tests = 200
    pass_ratio = 0.80

    hyperparams = load_hyperparams(folder)

    max_steps_per_episode = hyperparams["max_steps_per_episode"]
    threshold_reward = pass_ratio * max_steps_per_episode

    print("\n===================================")
    print("Progressive 80% testing")
    print("===================================")
    print(f"Folder: {folder}")
    print(f"Num tests per checkpoint: {num_tests}")
    print(f"Max steps per episode: {max_steps_per_episode}")
    print(f"Threshold reward: {threshold_reward}")
    print("===================================\n")

    trainer = Trainer(hyperparams, seed=None, folder=folder)

    # ---------------------------------------------------------
    # Stage 1:
    # Test cartpole_0 with weight_0, meaning MC1 checkpoints
    # ---------------------------------------------------------
    print("\n========== Stage 1 ==========")
    print("Testing cartpole_0 with weight_0 / MC1")
    print("=============================\n")

    mc1_checkpoints = sorted(
        folder.glob("MC1_*.pth"),
        key=get_step_from_checkpoint,
    )

    stage1_results = []
    stage1_passed_steps = []

    for checkpoint_path in mc1_checkpoints:
        result = run_one_checkpoint_test(
            trainer=trainer,
            checkpoint_path=checkpoint_path,
            num_tests=num_tests,
            cartpole_selector=0,
            weight_selector=0,
            threshold_reward=threshold_reward,
        )

        stage1_results.append(result)

        if result["passed"]:
            stage1_passed_steps.append(result["step"])

    stage1_df = pd.DataFrame(stage1_results)
    stage1_df.to_csv(folder / "stage1_cartpole0_weight0_results.csv", index=False)

    print("\nStage 1 passed steps:")
    print(stage1_passed_steps)

    # ---------------------------------------------------------
    # Stage 2:
    # Only use steps that passed Stage 1.
    # Test cartpole_1 with weight_1, meaning MC2 at same steps.
    # ---------------------------------------------------------
    print("\n========== Stage 2 ==========")
    print("Testing cartpole_1 with weight_1 / MC2 only for Stage 1 passed steps")
    print("=============================\n")

    stage2_results = []
    stage2_passed_steps = []

    for step in stage1_passed_steps:
        checkpoint_path = get_checkpoint_path(
            folder=folder,
            weight_selector=1,
            step=step,
        )

        if not checkpoint_path.exists():
            print(f"Skipping step {step}: {checkpoint_path.name} does not exist.")
            continue

        result = run_one_checkpoint_test(
            trainer=trainer,
            checkpoint_path=checkpoint_path,
            num_tests=num_tests,
            cartpole_selector=1,
            weight_selector=1,
            threshold_reward=threshold_reward,
        )

        result["passed_stage1"] = True
        stage2_results.append(result)

        if result["passed"]:
            stage2_passed_steps.append(result["step"])

    stage2_df = pd.DataFrame(stage2_results)
    stage2_df.to_csv(folder / "stage2_cartpole1_weight1_results.csv", index=False)

    print("\nSteps that passed both Stage 1 and Stage 2:")
    print(stage2_passed_steps)

    # ---------------------------------------------------------
    # Stage 3:
    # Only use steps that passed Stage 1 and Stage 2.
    # Test cartpole_2 with weight_2, meaning MC3 at same steps.
    # ---------------------------------------------------------
    print("\n========== Stage 3 ==========")
    print("Testing cartpole_2 with weight_2 / MC3 only for Stage 1 + Stage 2 passed steps")
    print("=============================\n")

    stage3_results = []
    final_passed_steps = []

    for step in stage2_passed_steps:
        checkpoint_path = get_checkpoint_path(
            folder=folder,
            weight_selector=2,
            step=step,
        )

        if not checkpoint_path.exists():
            print(f"Skipping step {step}: {checkpoint_path.name} does not exist.")
            continue

        result = run_one_checkpoint_test(
            trainer=trainer,
            checkpoint_path=checkpoint_path,
            num_tests=num_tests,
            cartpole_selector=2,
            weight_selector=2,
            threshold_reward=threshold_reward,
        )

        result["passed_stage1"] = True
        result["passed_stage2"] = True
        stage3_results.append(result)

        if result["passed"]:
            final_passed_steps.append(result["step"])

    stage3_df = pd.DataFrame(stage3_results)
    stage3_df.to_csv(folder / "stage3_cartpole2_weight2_results.csv", index=False)

    print("\n===================================")
    print("Final steps that passed all 3 stages")
    print("===================================")
    print(final_passed_steps)

    # ---------------------------------------------------------
    # Final summary table
    # ---------------------------------------------------------
    final_summary = pd.DataFrame({
        "step": final_passed_steps,
        "passed_cartpole0_weight0": True,
        "passed_cartpole1_weight1": True,
        "passed_cartpole2_weight2": True,
    })

    final_summary.to_csv(folder / "final_passed_all_three_cartpoles.csv", index=False)

    print(f"\nSaved results in folder:")
    print(folder)
    print("\nCSV files created:")
    print("stage1_cartpole0_weight0_results.csv")
    print("stage2_cartpole1_weight1_results.csv")
    print("stage3_cartpole2_weight2_results.csv")
    print("final_passed_all_three_cartpoles.csv")


if __name__ == "__main__":
    main()