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


def get_checkpoint_path(folder, weight_selector, step):
    """
    weight_selector:
        0 -> MC1
        1 -> MC2
        2 -> MC3
    """
    keyword = ["MC1", "MC2", "MC3"][weight_selector]
    return folder / f"{keyword}_{step}.pth"


def run_checkpoint_test(
    trainer,
    folder,
    step,
    cartpole_selector,
    weight_selector,
    num_tests,
):
    checkpoint_path = get_checkpoint_path(
        folder=folder,
        weight_selector=weight_selector,
        step=step,
    )

    if not checkpoint_path.exists():
        print(f"Missing checkpoint: {checkpoint_path}")
        return None

    test_log = trainer.test(
        model_path=checkpoint_path,
        num_tests=num_tests,
        cartpole=cartpole_selector,
    )

    mean_reward = test_log["reward"].mean()
    std_reward = test_log["reward"].std(ddof=0)
    min_reward = test_log["reward"].min()
    max_reward = test_log["reward"].max()

    is_correct_pair = cartpole_selector == weight_selector

    print(
        f"Step {step} | "
        f"cartpole_{cartpole_selector} with weight_{weight_selector} | "
        f"mean={mean_reward:.6f} | "
        f"std={std_reward:.6f} | "
        f"correct_pair={is_correct_pair}"
    )

    return {
        "step": step,
        "cartpole_selector": cartpole_selector,
        "weight_selector": weight_selector,
        "checkpoint": checkpoint_path.name,
        "is_correct_pair": is_correct_pair,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "min_reward": min_reward,
        "max_reward": max_reward,
    }


def main():
    # ---------------------------------------------------------
    # Settings
    # ---------------------------------------------------------
    folder_name = "run_2026-06-26_00-54-50"
    folder = SCRIPT_DIR / "three_problems" / folder_name

    num_tests = 200

    # Minimum required decrease when using swapped weights.
    # 0.0 means correct weight only needs to be greater than swapped weights.
    # Example: 5.0 means correct weight must be at least 5 reward points better.
    required_drop = 0.0

    # This CSV should come from your previous progressive script.
    passed_steps_csv = folder / "final_passed_all_three_cartpoles.csv"

    # ---------------------------------------------------------
    # Load passed steps
    # ---------------------------------------------------------
    passed_df = pd.read_csv(passed_steps_csv)
    candidate_steps = passed_df["step"].astype(int).tolist()

    print("\n===================================")
    print("Swap weight testing")
    print("===================================")
    print(f"Folder: {folder}")
    print(f"Candidate steps: {candidate_steps}")
    print(f"Number of candidate steps: {len(candidate_steps)}")
    print(f"Num tests per pair: {num_tests}")
    print(f"Required drop: {required_drop}")
    print("===================================\n")

    hyperparams = load_hyperparams(folder)
    trainer = Trainer(hyperparams, seed=None, folder=folder)

    all_results = []
    summary_rows = []
    satisfying_steps = []

    # ---------------------------------------------------------
    # For every candidate step, test all 9 combinations:
    #
    # cartpole_0 with weight_0, weight_1, weight_2
    # cartpole_1 with weight_0, weight_1, weight_2
    # cartpole_2 with weight_0, weight_1, weight_2
    # ---------------------------------------------------------
    for step in candidate_steps:
        print("\n===================================")
        print(f"Testing step {step}")
        print("===================================\n")

        step_results = {}

        for cartpole_selector in [0, 1, 2]:
            for weight_selector in [0, 1, 2]:
                result = run_checkpoint_test(
                    trainer=trainer,
                    folder=folder,
                    step=step,
                    cartpole_selector=cartpole_selector,
                    weight_selector=weight_selector,
                    num_tests=num_tests,
                )

                if result is None:
                    continue

                all_results.append(result)
                step_results[(cartpole_selector, weight_selector)] = result["mean_reward"]

        # -----------------------------------------------------
        # Skip incomplete steps
        # -----------------------------------------------------
        expected_pairs = [
            (0, 0), (0, 1), (0, 2),
            (1, 0), (1, 1), (1, 2),
            (2, 0), (2, 1), (2, 2),
        ]

        if not all(pair in step_results for pair in expected_pairs):
            print(f"Step {step} skipped because some results are missing.")
            continue

        # -----------------------------------------------------
        # Correct means
        # -----------------------------------------------------
        c0_w0 = step_results[(0, 0)]
        c1_w1 = step_results[(1, 1)]
        c2_w2 = step_results[(2, 2)]

        # -----------------------------------------------------
        # Swapped means
        # -----------------------------------------------------
        c0_w1 = step_results[(0, 1)]
        c0_w2 = step_results[(0, 2)]

        c1_w0 = step_results[(1, 0)]
        c1_w2 = step_results[(1, 2)]

        c2_w0 = step_results[(2, 0)]
        c2_w1 = step_results[(2, 1)]

        # -----------------------------------------------------
        # Drops: correct performance - swapped performance
        # Positive drop means swapped weight is worse.
        # -----------------------------------------------------
        drop_c0_w1 = c0_w0 - c0_w1
        drop_c0_w2 = c0_w0 - c0_w2

        drop_c1_w0 = c1_w1 - c1_w0
        drop_c1_w2 = c1_w1 - c1_w2

        drop_c2_w0 = c2_w2 - c2_w0
        drop_c2_w1 = c2_w2 - c2_w1

        # -----------------------------------------------------
        # Constraints
        # -----------------------------------------------------
        cartpole0_pass = (
            drop_c0_w1 >= required_drop and
            drop_c0_w2 >= required_drop
        )

        cartpole1_pass = (
            drop_c1_w0 >= required_drop and
            drop_c1_w2 >= required_drop
        )

        cartpole2_pass = (
            drop_c2_w0 >= required_drop and
            drop_c2_w1 >= required_drop
        )

        all_constraints_pass = (
            cartpole0_pass and
            cartpole1_pass and
            cartpole2_pass
        )

        if all_constraints_pass:
            satisfying_steps.append(step)

        summary_row = {
            "step": step,

            "cartpole0_weight0_correct_mean": c0_w0,
            "cartpole0_weight1_swap_mean": c0_w1,
            "cartpole0_weight2_swap_mean": c0_w2,
            "drop_cartpole0_w0_minus_w1": drop_c0_w1,
            "drop_cartpole0_w0_minus_w2": drop_c0_w2,
            "cartpole0_pass": cartpole0_pass,

            "cartpole1_weight1_correct_mean": c1_w1,
            "cartpole1_weight0_swap_mean": c1_w0,
            "cartpole1_weight2_swap_mean": c1_w2,
            "drop_cartpole1_w1_minus_w0": drop_c1_w0,
            "drop_cartpole1_w1_minus_w2": drop_c1_w2,
            "cartpole1_pass": cartpole1_pass,

            "cartpole2_weight2_correct_mean": c2_w2,
            "cartpole2_weight0_swap_mean": c2_w0,
            "cartpole2_weight1_swap_mean": c2_w1,
            "drop_cartpole2_w2_minus_w0": drop_c2_w0,
            "drop_cartpole2_w2_minus_w1": drop_c2_w1,
            "cartpole2_pass": cartpole2_pass,

            "required_drop": required_drop,
            "all_constraints_pass": all_constraints_pass,
        }

        summary_rows.append(summary_row)

        print("\nStep summary:")
        print(f"cartpole_0 correct weight_0 mean: {c0_w0:.6f}")
        print(f"cartpole_0 swap weight_1 mean:    {c0_w1:.6f} | drop={drop_c0_w1:.6f}")
        print(f"cartpole_0 swap weight_2 mean:    {c0_w2:.6f} | drop={drop_c0_w2:.6f}")
        print(f"cartpole_0 pass: {cartpole0_pass}")

        print(f"cartpole_1 correct weight_1 mean: {c1_w1:.6f}")
        print(f"cartpole_1 swap weight_0 mean:    {c1_w0:.6f} | drop={drop_c1_w0:.6f}")
        print(f"cartpole_1 swap weight_2 mean:    {c1_w2:.6f} | drop={drop_c1_w2:.6f}")
        print(f"cartpole_1 pass: {cartpole1_pass}")

        print(f"cartpole_2 correct weight_2 mean: {c2_w2:.6f}")
        print(f"cartpole_2 swap weight_0 mean:    {c2_w0:.6f} | drop={drop_c2_w0:.6f}")
        print(f"cartpole_2 swap weight_1 mean:    {c2_w1:.6f} | drop={drop_c2_w1:.6f}")
        print(f"cartpole_2 pass: {cartpole2_pass}")

        print(f"All constraints pass: {all_constraints_pass}")

    # ---------------------------------------------------------
    # Save logs
    # ---------------------------------------------------------
    all_results_df = pd.DataFrame(all_results)
    summary_df = pd.DataFrame(summary_rows)

    all_results_path = folder / "swap_all_pair_results.csv"
    summary_path = folder / "swap_summary_results.csv"
    satisfying_path = folder / "swap_satisfying_steps.csv"

    all_results_df.to_csv(all_results_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    satisfying_df = pd.DataFrame({
        "step": satisfying_steps,
        "passed_swap_specificity_constraints": True,
    })
    satisfying_df.to_csv(satisfying_path, index=False)

    print("\n===================================")
    print("Swap testing finished")
    print("===================================")
    print(f"Satisfying steps: {satisfying_steps}")
    print(f"Saved all pair results to: {all_results_path}")
    print(f"Saved summary results to: {summary_path}")
    print(f"Saved satisfying steps to: {satisfying_path}")


if __name__ == "__main__":
    main()