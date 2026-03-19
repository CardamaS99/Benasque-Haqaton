from __future__ import annotations

import csv
import numpy as np
from dataclasses import dataclass
from pathlib import Path

# Constante para representar rutas imposibles
INF = 999999

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

def _parse_time_to_minutes(value: str) -> int:
    value = _clean(value)
    
    if not value or value.upper() == "X":
        return INF

    if value == "0":
        return 0

    if "'" in value:
        try:
            parts = value.split("'")
            hours = int(parts[0]) if parts[0] else 0
            minutes = int(parts[1]) if len(parts) > 1 and parts[1] else 0
            return hours * 60 + minutes
        except (ValueError, IndexError):
            return INF

    if _is_int(value):
        return int(value)

    return INF

def _find_matrix_header(rows: list[list[str]]) -> tuple[int, int, list[int]]:
    for row_index, row in enumerate(rows):
        cleaned = [_clean(cell) for cell in row]
        for start_col in range(len(cleaned) - 2):
            if _is_int(cleaned[start_col]) and _is_int(cleaned[start_col + 1]):
                labels = []
                cursor = start_col
                while cursor < len(cleaned) and _is_int(cleaned[cursor]):
                    labels.append(int(cleaned[cursor]))
                    cursor += 1
                if len(labels) >= 2:
                    return row_index, start_col, labels
    raise ValueError("No se pudo localizar el encabezado de la matriz en el CSV.")

def extract_connectivity_matrix(rows: list[list[str]]) -> tuple[list[int], list[list[int]]]:
    header_idx, start_col, column_labels = _find_matrix_header(rows)
    size = len(column_labels)
    
    row_label_col = start_col - 1 if start_col > 0 else start_col
    data_start_col = start_col

    matrix_by_row_label: dict[int, list[int]] = {}

    for row in rows[header_idx + 1 :]:
        if len(row) <= row_label_col: continue
        row_label = _clean(row[row_label_col])
        if not _is_int(row_label): continue
        
        row_id = int(row_label)
        if row_id not in column_labels: continue

        cells = row[data_start_col : data_start_col + size]
        if len(cells) < size:
            cells += ["X"] * (size - len(cells))

        matrix_by_row_label[row_id] = [_parse_time_to_minutes(c) for c in cells]

    matrix = [matrix_by_row_label.get(lbl, [INF] * size) for lbl in column_labels]
    return column_labels, matrix

def convert_to_sparse(labels: list[int], matrix: list[list[int]]) -> np.ndarray:
    matrix_np = np.array(matrix, dtype=np.int32)
    labels_np = np.array(labels, dtype=np.int32)

    # Solo rutas posibles (menor que INF) y excluyendo diagonal
    row_indices, col_indices = np.where(matrix_np < INF)
    mask = row_indices != col_indices
    row_indices, col_indices = row_indices[mask], col_indices[mask]

    if row_indices.size == 0:
        return np.empty((0, 3), dtype=np.int32)

    weights = matrix_np[row_indices, col_indices]
    return np.column_stack((labels_np[row_indices], labels_np[col_indices], weights))

def get_sparse_connectivity(csv_path: str | Path) -> np.ndarray:
    """
    Función principal: Lee el CSV y devuelve únicamente la matriz dispersa (sparse).
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"No se encuentra el archivo en {path}")

    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    labels, matrix = extract_connectivity_matrix(rows)
    return convert_to_sparse(labels, matrix)

if __name__ == "__main__":
    # Ejemplo de uso: solo imprime la sparse resultante
    try:
        sparse_matrix = get_sparse_connectivity("./data/data.csv")
        print(sparse_matrix)
    except Exception as e:
        print(f"Error: {e}")