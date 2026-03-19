from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Node:
	id: int
	place: str
	type: str
	elevation: int
	winter_gear: str
	summer_gear: str
    


def _is_int(value: str) -> bool:
	try:
		int(value)
		return True
	except (TypeError, ValueError):
		return False


def _clean(cell: str) -> str:
	return (cell or "").strip()


def _find_matrix_header(rows: list[list[str]]) -> tuple[int, int, list[int]]:
	for row_index, row in enumerate(rows):
		cleaned = [_clean(cell) for cell in row]
		for start_col in range(len(cleaned) - 2):
			if _is_int(cleaned[start_col]) and _is_int(cleaned[start_col + 1]) and _is_int(cleaned[start_col + 2]):
				labels = []
				cursor = start_col
				while cursor < len(cleaned) and _is_int(cleaned[cursor]):
					labels.append(int(cleaned[cursor]))
					cursor += 1
				if len(labels) >= 3:
					return row_index, start_col, labels
	raise ValueError("No se pudo localizar el encabezado de la matriz de conectividad en el CSV.")


def _parse_time_to_minutes(value: str) -> int:
	value = _clean(value)
	if not value or value.upper() == "X":
		return 0

	if "'" in value:
		hours, minutes = value.split("'", maxsplit=1)
		if _is_int(hours) and _is_int(minutes):
			return int(hours) * 60 + int(minutes)

	if _is_int(value):
		return int(value)

	return 0


def extract_nodes(rows: list[list[str]]) -> list[Node]:
	nodes: list[Node] = []
	for row in rows:
		if len(row) < 6:
			continue

		node_id = _clean(row[0])
		place = _clean(row[1])
		node_type = _clean(row[2])
		elevation = _clean(row[3])
		winter_gear = _clean(row[4])
		summer_gear = _clean(row[5])

		if not (_is_int(node_id) and place):
			continue

		if not _is_int(elevation):
			continue

		nodes.append(
			Node(
				id=int(node_id),
				place=place,
				type=node_type,
				elevation=int(elevation),
				winter_gear=winter_gear,
				summer_gear=summer_gear,
			)
		)

	nodes.sort(key=lambda node: node.id)
	return nodes


def extract_connectivity_matrix(rows: list[list[str]]) -> tuple[list[int], list[list[int]]]:
	header_row_index, start_col, column_labels = _find_matrix_header(rows)
	size = len(column_labels)

	candidate_cols = [start_col]
	if start_col > 0:
		candidate_cols.append(start_col - 1)

	row_label_col = max(
		candidate_cols,
		key=lambda col: sum(
			1
			for row in rows[header_row_index + 1 :]
			if len(row) > col and _is_int(_clean(row[col])) and int(_clean(row[col])) in column_labels
		),
	)
	data_start_col = row_label_col + 1

	matrix_by_row_label: dict[int, list[int]] = {}

	for row in rows[header_row_index + 1 :]:
		if len(row) <= row_label_col:
			continue

		row_label = _clean(row[row_label_col])
		if not _is_int(row_label):
			continue

		row_id = int(row_label)
		if row_id not in column_labels:
			continue

		cells = row[data_start_col : data_start_col + size]
		if len(cells) < size:
			cells = cells + [""] * (size - len(cells))

		parsed_row: list[int] = []
		for cell in cells:
			parsed_row.append(_parse_time_to_minutes(cell))

		matrix_by_row_label[row_id] = parsed_row
		if len(matrix_by_row_label) == size:
			break

	matrix = [matrix_by_row_label.get(label, [0] * size) for label in column_labels]
	return column_labels, matrix


def convert_to_sparse(labels: list[int], matrix: list[list[int]] | np.ndarray) -> np.ndarray:
	matrix_np = np.asarray(matrix, dtype=np.int32)
	labels_np = np.asarray(labels, dtype=np.int32)

	if matrix_np.ndim != 2:
		raise ValueError("La matriz de conectividad debe tener dos dimensiones.")

	if matrix_np.shape[0] != matrix_np.shape[1]:
		raise ValueError("La matriz debe ser cuadrada.")

	if len(labels_np) != matrix_np.shape[0]:
		raise ValueError("El número de etiquetas debe coincidir con el número de filas de la matriz.")

	row_indices, col_indices = np.nonzero(matrix_np > 0)
	if row_indices.size == 0:
		return np.empty((0, 3), dtype=np.int32)

	minutes = matrix_np[row_indices, col_indices]

	return np.column_stack((labels_np[row_indices], labels_np[col_indices], minutes)).astype(np.int32, copy=False)




def load_graph_data(csv_path: str | Path) -> tuple[list[Node], list[int], list[list[int]]]:
	csv_path = Path(csv_path)
	with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
		rows = list(csv.reader(csv_file))

	nodes = extract_nodes(rows)
	labels, connectivity_matrix = extract_connectivity_matrix(rows)
	return nodes, labels, connectivity_matrix