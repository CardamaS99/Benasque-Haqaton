from __future__ import annotations

from pathlib import Path

import numpy as np

try:
	from preprocessing import INF, load_graph_data
except ModuleNotFoundError:
	from .preprocessing import INF, load_graph_data


def build_weight_matrix(connectivity_minutes: list[list[int]] | np.ndarray) -> np.ndarray:
	connectivity_np = np.asarray(connectivity_minutes, dtype=np.float64)
	if connectivity_np.ndim != 2 or connectivity_np.shape[0] != connectivity_np.shape[1]:
		raise ValueError("La matriz de conectividad debe ser cuadrada.")

	weights = connectivity_np.copy()
	weights[weights >= INF] = np.inf
	np.fill_diagonal(weights, 0.0)
	return weights


def floyd_warshall_min_times(weight_matrix: np.ndarray) -> np.ndarray:
	distances = weight_matrix.astype(np.float64, copy=True)
	size = distances.shape[0]

	for k in range(size):
		through_k = distances[:, [k]] + distances[[k], :]
		distances = np.minimum(distances, through_k)

	return distances


def solve_classical_warshall(csv_path: str | Path) -> tuple[list[int], np.ndarray]:
	_, labels, connectivity_matrix = load_graph_data(csv_path)
	weights = build_weight_matrix(connectivity_matrix)
	min_times = floyd_warshall_min_times(weights)
	return labels, min_times


def get_min_time_between(labels: list[int], min_times: np.ndarray, source: int, target: int) -> float:
	label_to_index = {label: index for index, label in enumerate(labels)}
	if source not in label_to_index or target not in label_to_index:
		raise ValueError("Los nodos source/target deben existir en labels.")

	i = label_to_index[source]
	j = label_to_index[target]
	return float(min_times[i, j])


def min_times_to_sparse(labels: list[int], min_times: np.ndarray) -> np.ndarray:
	labels_np = np.asarray(labels, dtype=np.int32)
	distances = np.asarray(min_times, dtype=np.float64)

	if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
		raise ValueError("min_times debe ser una matriz cuadrada.")

	if len(labels_np) != distances.shape[0]:
		raise ValueError("El número de labels debe coincidir con el tamaño de min_times.")

	mask = np.isfinite(distances) & (distances > 0)
	row_indices, col_indices = np.nonzero(mask)
	if row_indices.size == 0:
		return np.empty((0, 3), dtype=np.int32)

	minutes = distances[row_indices, col_indices].astype(np.int32)
	return np.column_stack((labels_np[row_indices], labels_np[col_indices], minutes))


if __name__ == "__main__":
	project_root = Path(__file__).resolve().parents[1]
	csv_file = project_root / "data" / "data.csv"

	labels, min_times = solve_classical_warshall(csv_file)
	min_times_sparse = min_times_to_sparse(labels, min_times)

	print("Node labels:")
	print(labels)
	print("\nMinimum-time matrix (minutes) with Floyd-Warshall:")
	print(np.where(np.isinf(min_times), -1, min_times).astype(int))
	print("\nSparse result matrix [source, target, minutes]:")
	print(min_times_sparse)
