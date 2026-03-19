import csv
import numpy as np
import pandas as pd
from pathlib import Path

# 1. IMPORTACIÓN DESDE TU OTRO SCRIPT
# Importamos la función que creamos en preprocessing2.py
try:
    from routes_creator import get_sparse_connectivity, INF
except ImportError:
    print("❌ Error: No se pudo importar 'get_sparse_connectivity' de preprocessing2.py")
    print("Asegúrate de que ambos archivos estén en la misma carpeta.")
    INF = 999999

# Configuración de Pandas para que la tabla se vea bien en consola
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def get_user_preferences(available_nodes):
    print("\n" + "="*50)
    print("🌲 CONFIGURACIÓN DE TU RUTA PERSONALIZADA 🌲")
    print("="*50)
    
    # Mostrar lista para elegir inicio
    print("\n📍 Lugares disponibles:")
    for n in available_nodes:
        print(f"[{n['id']}] {n['name']} ({n['elev']}m)")
    
    try:
        start_id = int(input("\n👉 Introduce el ID de tu PUNTO DE PARTIDA: "))
    except ValueError:
        start_id = 1 # Por defecto Benasque o el ID 1

    try:
        max_elev = int(input("🏔️ ¿Altitud máxima permitida? (Ej: 2200): "))
    except ValueError:
        max_elev = 3404 # Altura del Aneto por defecto
        
    print("\n🛡️ ¿Qué equipo llevas? (U: Urban, T: Trail, M: Mountain, S: Snow)")
    opcion_gear = input("Selecciona (ej: UTM): ").upper()
    
    gear_map = {'U': 'Urban', 'T': 'Trail', 'M': 'Mountain', 'S': 'Snow'}
    allowed_gears = [gear_map[letter] for letter in opcion_gear if letter in gear_map]

    return start_id, max_elev, allowed_gears

def main():
    # --- DEFINICIÓN DE LA RUTA DEL CSV ---
    # Aquí definimos dónde está el archivo realmente
    csv_path = Path(__file__).resolve().parent.parent / "data" / "data.csv" # Ajusta según tu estructura: "data/data.csv"
    
    if not csv_path.exists():
        print(f"❌ Error: No se encuentra el archivo CSV en: {csv_path.absolute()}")
        return

    # 1. Llamada a preproccessing2 pasándole la ruta del CSV
    try:
        # Aquí es donde le 'decimos' a la otra función dónde está el archivo
        sparse_matrix = get_sparse_connectivity(csv_path)

        # Append at the end of sparse_matrix the transpose of the original sparse_matrix (to ensure simetría)
        if sparse_matrix.size > 0:
            sparse_matrix = np.vstack((sparse_matrix, sparse_matrix[:, [1, 0, 2]]))

    except Exception as e:
        print(f"❌ Error al procesar la sparse en preproccessing2: {e}")
        return

    # 2. Leer metadatos de nodos para el filtrado
    all_nodes = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = list(csv.reader(f))
        for row in reader:
            if row and row[0].isdigit():
                all_nodes.append({
                    'id': int(row[0]),
                    'name': row[1],
                    'elev': int(row[3]),
                    'gear': row[5]
                })

    # 3. Pedir preferencias
    start_id, max_elev, allowed_gears = get_user_preferences(all_nodes)

    # 4. Filtrar nodos (siempre incluimos el ID de inicio)
    valid_nodes = [
        n for n in all_nodes 
        if n['id'] == start_id or (n['elev'] <= max_elev and n['gear'] in allowed_gears)
    ]
    
    n_sub = len(valid_nodes)
    if n_sub < 1:
        print("❌ No hay nodos que cumplan los criterios.")
        return

    ids_sub = [n['id'] for n in valid_nodes]
    elevs_sub = [n['elev'] for n in valid_nodes]
    
    # 5. Reconstruir Matriz Densa desde la Sparse
    id_to_idx = {node['id']: i for i, node in enumerate(valid_nodes)}
    valid_ids_set = set(id_to_idx.keys())

    sub_matrix = np.full((n_sub, n_sub), float(INF))
    np.fill_diagonal(sub_matrix, 0)

    for row in sparse_matrix:
        u, v, w = int(row[0]), int(row[1]), int(row[2])
        if u in valid_ids_set and v in valid_ids_set:
            sub_matrix[id_to_idx[u], id_to_idx[v]] = w

    elevs_array = np.array(elevs_sub)
    elev_diff_matrix = np.abs(elevs_array[None, :] - elevs_array[:, None])

    # --- CREACIÓN DE DATAFRAMES CON IDs ---
    df_tiempos = pd.DataFrame(sub_matrix, index=ids_sub, columns=ids_sub)
    df_desniveles = pd.DataFrame(elev_diff_matrix, index=ids_sub, columns=ids_sub)
    print("\n✅ Matriz de tiempos (en minutos):")
    print(df_tiempos)
    print("\n✅ Matriz de desniveles (en metros):")
    print(df_desniveles)

    return df_tiempos, df_desniveles


if __name__ == "__main__":
    main()