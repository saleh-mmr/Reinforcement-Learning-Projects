from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main():
    base_dir = Path(__file__).resolve().parent
    folder = base_dir / "three_problems/run_2026-06-13_07-38-17"
    file = folder / "details_log.csv"

    df = pd.read_csv(file)

    table_df = df.iloc[0].reset_index()
    table_df.columns = ["Parameter", "Value"]

    fig, ax = plt.subplots(figsize=(7, len(table_df) * 0.55))
    ax.axis("off")

    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=[0.45, 0.55]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1, 1.6)

    for (row, col), cell in table.get_celld().items():
        cell.set_text_props(ha="center", va="center")

        if row == 0:
            cell.set_facecolor("#D9EAF7")
            cell.set_text_props(weight="bold", ha="center", va="center")

    output_file = folder / "parameters_table.png"
    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    plt.close()

if __name__ == "__main__":
    main()