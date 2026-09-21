from pathlib import Path

import pandas as pd
import torch
import matplotlib.pyplot as plt


plt.rcParams.update({"font.size": 14})

# ---------------------------------------------------------
# Change only these
# ---------------------------------------------------------
RUN_FOLDER = "run_2026-06-26_00-54-50"
STEP = 4659159
LAYER = "FC.2"
BINS = 50

# Folder:
# /Users/salehmmrezaei/Desktop/CNR/Synaptic_Weights/March/scripts/three_problems
BASE_DIR = Path(__file__).resolve().parent
BASE_FOLDER = BASE_DIR / "three_problems"


def load_weights(path, key):
    state_dict = torch.load(path, map_location="cpu")

    if key not in state_dict:
        raise KeyError(f"{key} not found. Available keys: {list(state_dict.keys())}")

    return state_dict[key].detach().cpu().numpy().flatten()


def main():
    folder = (BASE_FOLDER / RUN_FOLDER).resolve()
    weight_key = f"{LAYER}.weight"

    print("Looking in:", folder)

    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    log_path = folder / "details_log.csv"
    if not log_path.exists():
        raise FileNotFoundError(f"details_log.csv not found: {log_path}")

    paths = [
        folder / f"MC1_{STEP}.pth",
        folder / f"MC2_{STEP}.pth",
        folder / f"MC3_{STEP}.pth",
    ]

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

    log = pd.read_csv(log_path).iloc[0]

    weights = [
        ("MC1", load_weights(paths[0], weight_key), log.get("CP_pole_length_1", "N/A"), log.get("CP_pole_mass_1", "N/A")),
        ("MC2", load_weights(paths[1], weight_key), log.get("CP_pole_length_2", "N/A"), log.get("CP_pole_mass_2", "N/A")),
        ("MC3", load_weights(paths[2], weight_key), log.get("CP_pole_length_3", "N/A"), log.get("CP_pole_mass_3", "N/A")),
    ]

    global_min = min(w.min() for _, w, _, _ in weights)
    global_max = max(w.max() for _, w, _, _ in weights)

    fig, axes = plt.subplots(1, 3, figsize=(22, 6), sharex=True, sharey=True)

    for ax, (name, w, pole_length, pole_mass) in zip(axes, weights):
        ax.hist(
            w,
            bins=BINS,
            range=(global_min, global_max),
            edgecolor="black",
            alpha=0.75,
        )

        ax.set_title(f"{weight_key} - {name}\nL={pole_length}, M={pole_mass}")
        ax.set_xlabel("Weight value")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()