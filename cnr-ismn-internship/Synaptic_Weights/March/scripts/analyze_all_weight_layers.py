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
RUN_FOLDER = "run_2026-06-12_20-44-23"
STEP = 686478

BINS_HISTOGRAM = 50
N_BINS_ENTROPY = 50

MODEL_NAMES = ["MC1", "MC2", "MC3"]

# Only these layers will get eigenvalue / SVD analysis
SQUARE_LAYERS_FOR_MATRIX_ANALYSIS = ["FC.2.weight", "FC.4.weight"]

SAVE_DIFFERENCE_HEATMAPS = True

BASE_DIR = Path(__file__).resolve().parent
BASE_FOLDER = BASE_DIR / "three_problems"


# ---------------------------------------------------------
# Loading
# ---------------------------------------------------------
def load_state_dict(path):
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    obj = torch.load(path, map_location="cpu")

    if isinstance(obj, dict):
        for key in ["state_dict", "model_state_dict", "net", "model"]:
            if key in obj and isinstance(obj[key], dict):
                return obj[key]

    return obj


def get_tensor(state_dict, key):
    if key not in state_dict:
        raise KeyError(f"{key} not found. Available keys: {list(state_dict.keys())}")

    return state_dict[key].detach().cpu().numpy()


def get_weight_layer_keys(state_dicts):
    common_keys = set.intersection(*[set(sd.keys()) for sd in state_dicts])
    weight_keys = sorted([key for key in common_keys if key.endswith(".weight")])

    if not weight_keys:
        raise ValueError("No common '.weight' layers found.")

    return weight_keys


def safe_name(name):
    return name.replace(".", "_").replace("/", "_")


# ---------------------------------------------------------
# Information content / entropy
# ---------------------------------------------------------
def entropy_bits(values, n_bins=50, value_range=None):
    x = np.asarray(values).ravel()
    x = x[np.isfinite(x)]

    if x.size == 0:
        return np.nan

    counts, _ = np.histogram(x, bins=n_bins, range=value_range)
    probs = counts[counts > 0] / counts.sum()

    return float(-np.sum(probs * np.log2(probs)))


def normalized_entropy_bits(values, n_bins=50, value_range=None):
    h = entropy_bits(values, n_bins=n_bins, value_range=value_range)

    if not np.isfinite(h) or n_bins <= 1:
        return np.nan

    return float(h / np.log2(n_bins))


# ---------------------------------------------------------
# Eigenvalue and SVD analysis
# ---------------------------------------------------------
def matrix_analysis(W):
    """
    Calculates eigenvalue and singular-value summaries for square matrices.

    Eigenvalues can be complex, so we summarize their absolute values.
    SVD singular values are always real and non-negative.
    """
    W = np.asarray(W)

    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        return {}

    eigenvalues = np.linalg.eigvals(W)
    eig_abs = np.abs(eigenvalues)

    singular_values = np.linalg.svd(W, compute_uv=False)
    sv_sum = singular_values.sum()

    if sv_sum > 0:
        sv_probs = singular_values / sv_sum
        sv_entropy = float(-np.sum(sv_probs[sv_probs > 0] * np.log2(sv_probs[sv_probs > 0])))
        sv_effective_rank = float(2 ** sv_entropy)
    else:
        sv_entropy = np.nan
        sv_effective_rank = np.nan

    if singular_values[-1] > 0:
        condition_number = float(singular_values[0] / singular_values[-1])
    else:
        condition_number = np.inf

    return {
        "eig_abs_mean": float(np.mean(eig_abs)),
        "eig_abs_std": float(np.std(eig_abs)),
        "eig_abs_min": float(np.min(eig_abs)),
        "eig_abs_max": float(np.max(eig_abs)),
        "spectral_radius": float(np.max(eig_abs)),
        "sv_mean": float(np.mean(singular_values)),
        "sv_std": float(np.std(singular_values)),
        "sv_min": float(np.min(singular_values)),
        "sv_max": float(np.max(singular_values)),
        "sv_entropy_bits": sv_entropy,
        "sv_effective_rank": sv_effective_rank,
        "condition_number": condition_number,
    }


def save_eigenvalues_and_singular_values(W, model, layer_key, output_dir):
    """
    Saves the raw eigenvalues and singular values for FC.2 and FC.4.
    """
    W = np.asarray(W)

    eigenvalues = np.linalg.eigvals(W)
    singular_values = np.linalg.svd(W, compute_uv=False)

    df_eig = pd.DataFrame(
        {
            "model": model,
            "layer": layer_key,
            "eigenvalue_real": np.real(eigenvalues),
            "eigenvalue_imag": np.imag(eigenvalues),
            "eigenvalue_abs": np.abs(eigenvalues),
        }
    )

    df_sv = pd.DataFrame(
        {
            "model": model,
            "layer": layer_key,
            "singular_value_index": np.arange(len(singular_values)),
            "singular_value": singular_values,
        }
    )

    eig_path = output_dir / f"{safe_name(layer_key)}_{model}_eigenvalues_step_{STEP}.csv"
    sv_path = output_dir / f"{safe_name(layer_key)}_{model}_singular_values_step_{STEP}.csv"

    df_eig.to_csv(eig_path, index=False)
    df_sv.to_csv(sv_path, index=False)

    print(f"Saved eigenvalues: {eig_path}")
    print(f"Saved singular values: {sv_path}")


# ---------------------------------------------------------
# Plotting
# ---------------------------------------------------------
def plot_weight_heatmaps(items, layer_key, output_dir):
    max_abs = max(np.max(np.abs(values)) for _, values, _, _ in items)

    if max_abs == 0:
        max_abs = 1.0

    fig, axes = plt.subplots(1, len(items), figsize=(7 * len(items), 6), sharey=True)

    for ax, (model, values, pole_length, pole_mass) in zip(axes, items):
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

        ax.set_title(f"{layer_key} - {model}\nL={pole_length}, M={pole_mass}")
        ax.set_xlabel("Input neuron index")
        ax.set_ylabel("Output neuron index")
        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()

    output_path = output_dir / f"{safe_name(layer_key)}_heatmap_step_{STEP}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved heatmap: {output_path}")

    plt.show()
    plt.close(fig)


def plot_weight_difference_heatmaps(items, layer_key, output_dir):
    if len(items) != 3:
        return

    model_1, w1, l1, m1 = items[0]
    model_2, w2, l2, m2 = items[1]
    model_3, w3, l3, m3 = items[2]

    if not (w1.shape == w2.shape == w3.shape):
        print(f"Skipping difference heatmap for {layer_key}: shapes differ.")
        return

    diff_items = [
        (f"{model_1} - {model_2}", w1 - w2, f"{l1} - {l2}", f"{m1} - {m2}"),
        (f"{model_1} - {model_3}", w1 - w3, f"{l1} - {l3}", f"{m1} - {m3}"),
        (f"{model_2} - {model_3}", w2 - w3, f"{l2} - {l3}", f"{m2} - {m3}"),
    ]

    max_abs = max(np.max(np.abs(values)) for _, values, _, _ in diff_items)

    if max_abs == 0:
        max_abs = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(21, 6), sharey=True)

    for ax, (name, values, pole_length, pole_mass) in zip(axes, diff_items):
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

        ax.set_title(f"Difference {layer_key}\n{name}\nL={pole_length}, M={pole_mass}")
        ax.set_xlabel("Input neuron index")
        ax.set_ylabel("Output neuron index")
        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()

    output_path = output_dir / f"{safe_name(layer_key)}_difference_heatmap_step_{STEP}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved difference heatmap: {output_path}")

    plt.show()
    plt.close(fig)


def plot_weight_histograms(items, layer_key, output_dir):
    flattened = [np.asarray(values).ravel() for _, values, _, _ in items]

    global_min = min(w.min() for w in flattened)
    global_max = max(w.max() for w in flattened)

    if global_min == global_max:
        global_min -= 1.0
        global_max += 1.0

    fig, axes = plt.subplots(
        1,
        len(items),
        figsize=(7 * len(items), 6),
        sharex=True,
        sharey=True,
    )

    for ax, (model, values, pole_length, pole_mass) in zip(axes, items):
        w = np.asarray(values).ravel()

        ax.hist(
            w,
            bins=BINS_HISTOGRAM,
            range=(global_min, global_max),
            edgecolor="black",
            alpha=0.75,
        )

        ax.set_title(f"{layer_key} - {model}\nL={pole_length}, M={pole_mass}")
        ax.set_xlabel("Weight value")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / f"{safe_name(layer_key)}_histogram_step_{STEP}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved histogram: {output_path}")

    plt.show()
    plt.close(fig)


def plot_information_table(summary_df, output_dir):
    display_df = summary_df.copy()

    numeric_cols = display_df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        display_df[col] = display_df[col].map(
            lambda x: "inf" if np.isinf(x) else f"{x:.6f}" if pd.notna(x) else "nan"
        )

    fig_height = max(5, 0.45 * (len(display_df) + 1))
    fig_width = 28

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.4)

    ax.set_title(
        f"Information content, eigenvalues, and SVD of weight layers - step {STEP}",
        fontsize=16,
        pad=20,
    )

    plt.tight_layout()

    output_path = output_dir / f"all_weight_layers_information_matrix_table_step_{STEP}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved table image: {output_path}")

    plt.show()
    plt.close(fig)


def plot_eigenvalue_spectrum(all_eig_df, output_dir):
    for layer_key in SQUARE_LAYERS_FOR_MATRIX_ANALYSIS:
        df = all_eig_df[all_eig_df["layer"] == layer_key]

        if df.empty:
            continue

        plt.figure(figsize=(8, 7))

        for model in MODEL_NAMES:
            d = df[df["model"] == model]
            plt.scatter(
                d["eigenvalue_real"],
                d["eigenvalue_imag"],
                label=model,
                alpha=0.75,
            )

        plt.axhline(0, color="black", linewidth=1)
        plt.axvline(0, color="black", linewidth=1)
        plt.title(f"Eigenvalue spectrum: {layer_key}")
        plt.xlabel("Real part")
        plt.ylabel("Imaginary part")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = output_dir / f"{safe_name(layer_key)}_eigenvalue_spectrum_step_{STEP}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved eigenvalue spectrum: {output_path}")

        plt.show()
        plt.close()


def plot_singular_values(all_sv_df, output_dir):
    for layer_key in SQUARE_LAYERS_FOR_MATRIX_ANALYSIS:
        df = all_sv_df[all_sv_df["layer"] == layer_key]

        if df.empty:
            continue

        plt.figure(figsize=(10, 6))

        for model in MODEL_NAMES:
            d = df[df["model"] == model]
            plt.plot(
                d["singular_value_index"],
                d["singular_value"],
                marker="o",
                label=model,
            )

        plt.title(f"Singular values: {layer_key}")
        plt.xlabel("Singular value index")
        plt.ylabel("Singular value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = output_dir / f"{safe_name(layer_key)}_singular_values_step_{STEP}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved singular-value plot: {output_path}")

        plt.show()
        plt.close()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    folder = (BASE_FOLDER / RUN_FOLDER).resolve()
    print("Looking in:", folder)

    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    log_path = folder / "details_log.csv"

    if not log_path.exists():
        raise FileNotFoundError(f"details_log.csv not found: {log_path}")

    output_dir = folder / f"all_weight_layers_analysis_step_{STEP}"
    output_dir.mkdir(exist_ok=True)

    paths = [
        folder / f"MC1_{STEP}.pth",
        folder / f"MC2_{STEP}.pth",
        folder / f"MC3_{STEP}.pth",
    ]

    log = pd.read_csv(log_path).iloc[0]

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

    state_dicts = [load_state_dict(path) for path in paths]
    weight_layer_keys = get_weight_layer_keys(state_dicts)

    print("\nFound weight layers:")
    for key in weight_layer_keys:
        print("  ", key)

    summary_rows = []
    all_eig_rows = []
    all_sv_rows = []

    for layer_key in weight_layer_keys:
        print(f"\nProcessing layer: {layer_key}")

        weights = [get_tensor(sd, layer_key) for sd in state_dicts]
        items = list(zip(MODEL_NAMES, weights, pole_lengths, pole_masses))

        layer_min = min(np.min(w) for w in weights)
        layer_max = max(np.max(w) for w in weights)

        if layer_min == layer_max:
            layer_min -= 1.0
            layer_max += 1.0

        entropy_range = (layer_min, layer_max)

        plot_weight_heatmaps(items, layer_key, output_dir)

        if SAVE_DIFFERENCE_HEATMAPS:
            plot_weight_difference_heatmaps(items, layer_key, output_dir)

        plot_weight_histograms(items, layer_key, output_dir)

        for model, values, pole_length, pole_mass in items:
            values_flat = np.asarray(values).ravel()

            row = {
                "model": model,
                "layer": layer_key,
                "shape": str(tuple(values.shape)),
                "n_parameters": int(values_flat.size),
                "mean": float(np.mean(values_flat)),
                "std": float(np.std(values_flat)),
                "min": float(np.min(values_flat)),
                "max": float(np.max(values_flat)),
                "H_bits": entropy_bits(
                    values_flat,
                    n_bins=N_BINS_ENTROPY,
                    value_range=entropy_range,
                ),
                "H_normalized": normalized_entropy_bits(
                    values_flat,
                    n_bins=N_BINS_ENTROPY,
                    value_range=entropy_range,
                ),
                "n_bins": N_BINS_ENTROPY,
                "pole_length": pole_length,
                "pole_mass": pole_mass,
            }

            if layer_key in SQUARE_LAYERS_FOR_MATRIX_ANALYSIS:
                row.update(matrix_analysis(values))

                eigenvalues = np.linalg.eigvals(values)
                singular_values = np.linalg.svd(values, compute_uv=False)

                for eig in eigenvalues:
                    all_eig_rows.append(
                        {
                            "model": model,
                            "layer": layer_key,
                            "eigenvalue_real": float(np.real(eig)),
                            "eigenvalue_imag": float(np.imag(eig)),
                            "eigenvalue_abs": float(np.abs(eig)),
                            "pole_length": pole_length,
                            "pole_mass": pole_mass,
                        }
                    )

                for i, sv in enumerate(singular_values):
                    all_sv_rows.append(
                        {
                            "model": model,
                            "layer": layer_key,
                            "singular_value_index": i,
                            "singular_value": float(sv),
                            "pole_length": pole_length,
                            "pole_mass": pole_mass,
                        }
                    )

                save_eigenvalues_and_singular_values(
                    values,
                    model,
                    layer_key,
                    output_dir,
                )

            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    eig_df = pd.DataFrame(all_eig_rows)
    sv_df = pd.DataFrame(all_sv_rows)

    summary_csv = output_dir / f"all_weight_layers_information_matrix_summary_step_{STEP}.csv"
    eig_csv = output_dir / f"FC2_FC4_eigenvalues_step_{STEP}.csv"
    sv_csv = output_dir / f"FC2_FC4_singular_values_step_{STEP}.csv"

    summary_df.to_csv(summary_csv, index=False)
    eig_df.to_csv(eig_csv, index=False)
    sv_df.to_csv(sv_csv, index=False)

    print("\nSaved summary CSV:")
    print(summary_csv)

    print("\nSaved combined eigenvalue CSV:")
    print(eig_csv)

    print("\nSaved combined singular-value CSV:")
    print(sv_csv)

    print("\nSummary:")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 260)
    print(summary_df.to_string(index=False))

    plot_information_table(summary_df, output_dir)
    plot_eigenvalue_spectrum(eig_df, output_dir)
    plot_singular_values(sv_df, output_dir)

    print("\nDone.")
    print("All images and CSV files were saved in:")
    print(output_dir)


if __name__ == "__main__":
    main()