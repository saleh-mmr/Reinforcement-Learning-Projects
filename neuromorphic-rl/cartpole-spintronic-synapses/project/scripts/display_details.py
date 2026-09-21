from csv import DictReader
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib.table import Table


def _format_value(value: str) -> str:
	value = "" if value is None else str(value)
	if value.endswith(".0"):
		try:
			as_float = float(value)
			if as_float.is_integer():
				return str(int(as_float))
		except ValueError:
			pass
	return value

def main():
	# Paths to your models
	base_dir = Path(__file__).resolve().parent
	folder = base_dir / "run_2026-05-27_00-20-33"
	file = folder / "details_log.csv"

	if not file.exists():
		raise FileNotFoundError(f"Could not find details log at: {file}")

	# Load the CSV file
	with file.open(newline="", encoding="utf-8") as handle:
		reader = DictReader(handle)
		rows = list(reader)
		columns = reader.fieldnames or []

	if not columns:
		raise ValueError(f"No columns found in CSV: {file}")
	if not rows:
		raise ValueError(f"No data rows found in CSV: {file}")

	row = rows[0]
	table_data = [[column, _format_value(row.get(column, ""))] for column in columns]
	output_path = folder / "details_table.png"

	fig_height = max(2.0, 0.45 * (len(table_data) + 1))
	fig, ax = plt.subplots(figsize=(6.5, fig_height))
	ax.axis("off")
	table = Table(ax, bbox=Bbox.from_bounds(0, 0, 1, 1))
	row_height = 1.0 / (len(table_data) + 1)
	col_widths = [0.42, 0.50]

	for col_index, header in enumerate(["parameter", "value"]):
		cell = table.add_cell(0, col_index, width=col_widths[col_index], height=row_height, text=header, loc="left")
		cell.set_text_props(weight="bold")
		cell.set_facecolor("#d9eaf7")
		cell.set_edgecolor("#cccccc")

	for row_index, (parameter, value) in enumerate(table_data, start=1):
		for col_index, cell_value in enumerate([parameter, value]):
			cell = table.add_cell(row_index, col_index, width=col_widths[col_index], height=row_height, text=cell_value, loc="left")
			cell.set_facecolor("#f7f7f7" if row_index % 2 == 0 else "white")
			cell.set_edgecolor("#cccccc")

	table.auto_set_font_size(False)
	table.set_fontsize(10)
	ax.add_table(table)

	fig.tight_layout()
	fig.savefig(output_path, dpi=300, bbox_inches="tight")
	plt.close(fig)

	print(f"Saved details table to: {output_path}")


if __name__ == "__main__":
	main()

