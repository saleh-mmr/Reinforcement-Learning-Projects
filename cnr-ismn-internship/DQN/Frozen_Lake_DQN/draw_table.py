import csv
import matplotlib.pyplot as plt

CSV_PATH = "tracked_neuron_history.csv"
MAX_ROWS = 40


def draw_csv_table(csv_path, max_rows):
    # ---- read csv ----
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = []

        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            rows.append(row)

    if len(rows) == 0:
        raise RuntimeError("CSV contains no data")

    # ---- create figure ----
    fig_height = max(6, 0.25 * len(rows))  # dynamic height
    fig, ax = plt.subplots(figsize=(8, fig_height))
    ax.axis("off")

    # ---- draw table ----
    table = ax.table(
        cellText=rows,
        colLabels=header,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)

    ax.set_title(f"Tracked Neuron History (First {len(rows)} Rows)", pad=20)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    draw_csv_table(CSV_PATH, MAX_ROWS)
