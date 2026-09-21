from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns


plt.rcParams.update({"font.size": 14})

# ---------------------------------------------------------
# Change only these
# ---------------------------------------------------------
RUN_FOLDER = "run_2026-06-26_00-54-50"
STEP = 4659159
LAYER = "FC.2"

BASE_DIR = Path(__file__).resolve().parent
BASE_FOLDER = BASE_DIR / "three_problems"


def load_state_dict(path):
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    return torch.load(path, map_location="cpu")


def get_tensor(state_dict, key):
    if key not in state_dict:
        raise KeyError(f"{key} not found. Available keys: {list(state_dict.keys())}")

    return state_dict[key].detach().cpu().numpy()


def plot_heatmaps(items, key, title_prefix, xlabel, ylabel, figsize):
    max_abs = max(np.max(np.abs(values)) for _, values, _, _ in items)

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)

    for ax, (name, values, pole_length, pole_mass) in zip(axes, items):
        sns.heatmap(
            values,
            ax=ax,
            cmap="seismic",
            center=0,
            vmin=-max_abs,
            vmax=max_abs,
            xticklabels=5,
            yticklabels=5 if values.shape[0] > 1 else False,
            cbar=True,
        )

        ax.set_title(f"{title_prefix} {key} - {name}\nL={pole_length}, M={pole_mass}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    plt.show()


def main():
    folder = (BASE_FOLDER / RUN_FOLDER).resolve()
    print("Looking in:", folder)

    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    log_path = folder / "details_log.csv"
    if not log_path.exists():
        raise FileNotFoundError(f"details_log.csv not found: {log_path}")

    weight_key = f"{LAYER}.weight"
    bias_key = f"{LAYER}.bias"

    paths = [
        folder / f"MC1_{STEP}.pth",
        folder / f"MC2_{STEP}.pth",
        folder / f"MC3_{STEP}.pth",
    ]

    log = pd.read_csv(log_path).iloc[0]

    state_dicts = [load_state_dict(path) for path in paths]

    weights = [get_tensor(sd, weight_key) for sd in state_dicts]
    biases = [get_tensor(sd, bias_key).reshape(1, -1) for sd in state_dicts]

    pole_lengths = [
        log.get("CP_pole_length_1", "N/A"),
        log.get("CP_pole_length_2", "N/A"),
        log.get("CP_pole_length_3", "N/A"),
    ]

    pole_masses = [
        log.get("CP_pole_mass_1", "N/A"),
        log.get("CP_pole_mass_2", "N/A"),
        log.get("CP_pole_mass_3", "N/A"),
    ]

    model_names = ["MC1", "MC2", "MC3"]

    weight_items = list(zip(model_names, weights, pole_lengths, pole_masses))
    bias_items = list(zip(model_names, biases, pole_lengths, pole_masses))

    plot_heatmaps(
        weight_items,
        key=weight_key,
        title_prefix="",
        xlabel="Input neuron index",
        ylabel="Output neuron index",
        figsize=(22, 6),
    )

    plot_heatmaps(
        bias_items,
        key=bias_key,
        title_prefix="",
        xlabel="Bias neuron index",
        ylabel="",
        figsize=(22, 4),
    )

    weight_diff_items = [
        ("MC1 - MC2", weights[0] - weights[1], f"{pole_lengths[0]} - {pole_lengths[1]}", ""),
        ("MC1 - MC3", weights[0] - weights[2], f"{pole_lengths[0]} - {pole_lengths[2]}", ""),
        ("MC2 - MC3", weights[1] - weights[2], f"{pole_lengths[1]} - {pole_lengths[2]}", ""),
    ]

    plot_heatmaps(
        weight_diff_items,
        key=weight_key,
        title_prefix="Difference",
        xlabel="Input neuron index",
        ylabel="Output neuron index",
        figsize=(22, 6),
    )

    bias_diff_items = [
        ("MC1 - MC2", biases[0] - biases[1], f"{pole_lengths[0]} - {pole_lengths[1]}", ""),
        ("MC1 - MC3", biases[0] - biases[2], f"{pole_lengths[0]} - {pole_lengths[2]}", ""),
        ("MC2 - MC3", biases[1] - biases[2], f"{pole_lengths[1]} - {pole_lengths[2]}", ""),
    ]

    plot_heatmaps(
        bias_diff_items,
        key=bias_key,
        title_prefix="Difference",
        xlabel="Bias neuron index",
        ylabel="",
        figsize=(22, 4),
    )


if __name__ == "__main__":
    main()