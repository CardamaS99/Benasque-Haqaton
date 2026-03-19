import csv
import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    from preprocessing import INF, load_graph_data, convert_to_sparse
except ImportError:
    from .preprocessing import INF, load_graph_data, convert_to_sparse


@st.cache_data
def load_data():
    """Load graph data from CSV"""
    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / "data" / "data.csv"

    nodes, labels, connectivity_matrix = load_graph_data(csv_path)
    
    node_dict = {node.id: {"name": node.place, "elev": node.elevation, "gear": node.summer_gear} for node in nodes}
    
    return labels, connectivity_matrix, node_dict


def format_matrix_display(matrix, inf_value=999999, replacement="∞"):
    """Format matrix for display, replacing inf with ∞"""
    display_matrix = np.where(
        (matrix == inf_value) | np.isinf(matrix),
        replacement,
        matrix.astype(int)
    )
    return display_matrix


def build_submatrix(connectivity_matrix, labels, valid_ids):
    """Build submatrix for valid node IDs"""
    id_to_idx = {label: i for i, label in enumerate(labels)}
    valid_ids_set = set(valid_ids)
    
    n_sub = len(valid_ids)
    sub_matrix = np.full((n_sub, n_sub), INF, dtype=float)
    np.fill_diagonal(sub_matrix, 0)
    
    connectivity_np = np.asarray(connectivity_matrix, dtype=float)
    
    for i, u in enumerate(valid_ids):
        for j, v in enumerate(valid_ids):
            if u in id_to_idx and v in id_to_idx:
                u_idx = id_to_idx[u]
                v_idx = id_to_idx[v]
                val = connectivity_np[u_idx, v_idx]
                if val != INF:
                    sub_matrix[i, j] = val
    
    return sub_matrix


def sparse_to_csv(sparse_array):
    """Convert sparse array to CSV string"""
    output = io.StringIO()
    output.write("source,target,minutes\n")
    for row in sparse_array:
        output.write(f"{int(row[0])},{int(row[1])},{int(row[2])}\n")
    return output.getvalue()


def dense_to_csv(matrix, node_ids):
    """Convert dense matrix to CSV string"""
    output = io.StringIO()
    output.write("," + ",".join(str(n) for n in node_ids) + "\n")
    for i, row_id in enumerate(node_ids):
        row_vals = [str(int(val)) if val != INF else str(int(INF)) for val in matrix[i]]
        output.write(f"{row_id}," + ",".join(row_vals) + "\n")
    return output.getvalue()


def main():
    st.set_page_config(page_title="🌲 Route Creator", layout="wide")
    
    st.title("🌲 Smart Route Creator")
    st.markdown("Build personalized hiking routes with custom preferences")
    
    # Load data
    labels, connectivity_matrix, node_dict = load_data()
    all_nodes = [{"id": label, "name": node_dict[label]["name"], "elev": node_dict[label]["elev"], "gear": node_dict[label]["gear"]} for label in labels]
    
    # Sidebar for preferences
    st.sidebar.header("⚙️ Route Preferences")
    
    st.sidebar.subheader("📍 Starting Point")
    start_id = st.sidebar.selectbox(
        "Select starting location:",
        options=[n["id"] for n in all_nodes],
        format_func=lambda x: f"[{x}] {node_dict[x]['name']} ({node_dict[x]['elev']}m)"
    )
    
    st.sidebar.subheader("🏔️ Elevation Constraints")
    max_elev = st.sidebar.slider(
        "Maximum elevation allowed (m):",
        min_value=1000,
        max_value=3500,
        value=3404,
        step=100
    )
    
    st.sidebar.subheader("🛡️ Available Gear")
    gear_options = st.sidebar.multiselect(
        "Select gear types:",
        options=["Urban", "Trail", "Mountain", "Snow"],
        default=["Urban", "Trail", "Mountain", "Snow"]
    )
    
    # Filter nodes
    valid_nodes = [
        n for n in all_nodes
        if n["id"] == start_id or (n["elev"] <= max_elev and n["gear"] in gear_options)
    ]
    
    if len(valid_nodes) < 1:
        st.error("❌ No locations match your criteria.")
        return
    
    valid_ids = [n["id"] for n in valid_nodes]
    elevs = [n["elev"] for n in valid_nodes]
    
    # Build submatrices
    sub_matrix_times = build_submatrix(connectivity_matrix, labels, valid_ids)
    
    # Elevation differences matrix
    elevs_array = np.array(elevs)
    elev_diff_matrix = np.abs(elevs_array[None, :] - elevs_array[:, None])
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["⏱️ Travel Times", "📊 Elevation Differences", "💾 Downloads"])
    
    with tab1:
        st.subheader("Travel Time Matrix (minutes)")
        st.markdown(f"**{len(valid_ids)}** locations selected • Starting point: **[{start_id}] {node_dict[start_id]['name']}**")
        
        # Display matrix with proper formatting
        display_times = format_matrix_display(sub_matrix_times, inf_value=INF)
        df_times = pd.DataFrame(display_times, index=valid_ids, columns=valid_ids)
        
        # Style the dataframe
        def highlight_infinite(val):
            if val == "∞":
                return "background-color: #ff0000; color: black"
            elif val == 0:
                return "background-color: #000000; color: white"
            else:
                return "background-color: #000000; color: white"
        
        styled_df = df_times.style.applymap(highlight_infinite)
        st.dataframe(styled_df, use_container_width=True)
            
    with tab2:
        st.subheader("Elevation Difference Matrix (meters)")
        df_elev = pd.DataFrame(elev_diff_matrix, index=valid_ids, columns=valid_ids)
        
        def highlight_elev(val):
            if val == 0:
                return "background-color: #000000; color: white"
            elif val > 500:
                return "background-color: #000000; color: white"
            return "background-color: #000000; color: white"
        
        styled_elev = df_elev.style.applymap(highlight_elev)
        st.dataframe(styled_elev, use_container_width=True)
    
    with tab3:
        st.subheader("📥 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Travel Times**")
            
            # Sparse CSV
            sparse_times = convert_to_sparse(valid_ids, sub_matrix_times)
            sparse_csv = sparse_to_csv(sparse_times)
            st.download_button(
                label="⬇️ Sparse CSV",
                data=sparse_csv,
                file_name="travel_times_sparse.csv",
                mime="text/csv",
                key="sparse_times"
            )
            
            # Dense CSV
            dense_csv = dense_to_csv(sub_matrix_times, valid_ids)
            st.download_button(
                label="⬇️ Dense CSV",
                data=dense_csv,
                file_name="travel_times_dense.csv",
                mime="text/csv",
                key="dense_times"
            )
        
        with col2:
            st.markdown("**Elevation Differences**")
            elev_dense_csv = dense_to_csv(elev_diff_matrix, valid_ids)
            st.download_button(
                label="⬇️ Dense CSV",
                data=elev_dense_csv,
                file_name="elevation_diff_dense.csv",
                mime="text/csv",
                key="dense_elev"
            )


if __name__ == "__main__":
    main()
