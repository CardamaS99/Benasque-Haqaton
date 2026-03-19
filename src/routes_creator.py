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


def normalize_finite_matrix(matrix, invalid_mask):
    """Normalize matrix values in [0, 1] over valid entries only."""
    matrix_np = np.asarray(matrix, dtype=float)
    valid_values = matrix_np[~invalid_mask]
    if valid_values.size == 0:
        return np.zeros_like(matrix_np, dtype=float)

    min_val = float(np.min(valid_values))
    max_val = float(np.max(valid_values))
    if max_val == min_val:
        normalized = np.zeros_like(matrix_np, dtype=float)
    else:
        normalized = (matrix_np - min_val) / (max_val - min_val)

    normalized[invalid_mask] = 0.0
    return normalized


def build_combination_matrix(times_matrix, diff_matrix, alpha, meters_per_minute):
    """Build combined normalized matrix: alpha*time + (1-alpha)*difference."""
    times_np = np.asarray(times_matrix, dtype=float)
    diff_np = np.asarray(diff_matrix, dtype=float)

    invalid_mask = times_np == INF
    diff_as_time = diff_np / max(float(meters_per_minute), 1e-9)

    norm_time = normalize_finite_matrix(times_np, invalid_mask)
    norm_diff = normalize_finite_matrix(diff_as_time, invalid_mask)

    combined = alpha * norm_time + (1.0 - alpha) * norm_diff
    combined[invalid_mask] = INF
    np.fill_diagonal(combined, 0.0)
    return combined


def build_graphviz_dot(
    valid_ids,
    adjacency_matrix,
    node_dict,
    label_mode="Name",
    skip_inf=False,
    availability_matrix=None,
):
    """Build a Graphviz DOT graph with weighted edges."""
    labels = {}
    for node_id in valid_ids:
        if label_mode == "Name":
            labels[node_id] = node_dict[node_id]["name"]
        else:
            labels[node_id] = str(node_id)

    lines = [
        "digraph RouteGraph {",
        "rankdir=LR;",
        "bgcolor=white;",
        "node [shape=circle, style=filled, fillcolor=black, fontcolor=white, color=black];",
    ]

    for node_id in valid_ids:
        node_label = labels[node_id].replace('"', "\\\"")
        lines.append(f'"{node_id}" [label="{node_label}"];')

    matrix_np = np.asarray(adjacency_matrix)
    availability_np = None if availability_matrix is None else np.asarray(availability_matrix)

    def is_available(i, j):
        if availability_np is None:
            return True
        return availability_np[i, j] != INF

    positive_weights = []
    for i, source in enumerate(valid_ids):
        for j, target in enumerate(valid_ids):
            if source == target:
                continue
            if not is_available(i, j):
                continue
            weight = matrix_np[i, j]
            if skip_inf and weight == INF:
                continue
            if weight != 0:
                positive_weights.append(float(weight))

    min_w = min(positive_weights) if positive_weights else 1.0
    max_w = max(positive_weights) if positive_weights else 1.0

    def normalize(weight: float) -> float:
        if max_w == min_w:
            return 1.0
        return (weight - min_w) / (max_w - min_w)

    def gradient_green_to_black(weight: float) -> str:
        t = normalize(weight)
        start_r, start_g, start_b = (0, 170, 0)
        end_r, end_g, end_b = (0, 0, 0)
        r = int(start_r + (end_r - start_r) * t)
        g = int(start_g + (end_g - start_g) * t)
        b = int(start_b + (end_b - start_b) * t)
        return f"#{r:02x}{g:02x}{b:02x}"

    for i, source in enumerate(valid_ids):
        for j, target in enumerate(valid_ids):
            weight = matrix_np[i, j]
            if source == target:
                continue
            if not is_available(i, j):
                continue
            if skip_inf and weight == INF:
                continue
            if weight != INF:
                if weight == 0:
                    lines.append(
                        f'"{source}" -> "{target}" [label="0", style="dashed", color="#2e7d32", fontcolor="#2e7d32", penwidth="1.0"];'
                    )
                else:
                    edge_color = gradient_green_to_black(float(weight))
                    penwidth = 1.2 + 2.8 * normalize(float(weight))
                    if float(weight).is_integer():
                        weight_label = f"{int(weight)}"
                    else:
                        weight_label = f"{float(weight):.3f}"
                    lines.append(
                        f'"{source}" -> "{target}" [label="{weight_label}", color="{edge_color}", fontcolor="{edge_color}", penwidth="{penwidth:.2f}"];'
                    )

    lines.append("}")
    return "\n".join(lines)


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
    elev_diff_matrix = elevs_array[None, :] - elevs_array[:, None]
    elev_diff_abs_matrix = np.abs(elev_diff_matrix)
    elev_diff_todor_matrix = np.maximum(elev_diff_matrix, 0)
    
    # Main content tabs
    tab1, tab2, tab26, tab25, tab4, tab3 = st.tabs([
        "⏱️ Travel Times",
        "📊 Elevation Differences",
        "📈 Elevation Absolute Difference",
        "Elevation Todor",
        "🕸️ Graph",
        "💾 Downloads",
    ])
    
    with tab1:
        st.subheader("Travel Time Matrix (minutes)")
        st.markdown(f"**{len(valid_ids)}** locations selected • Starting point: **[{start_id}] {node_dict[start_id]['name']}**")
        
        # Display matrix with proper formatting
        display_times = format_matrix_display(sub_matrix_times, inf_value=INF)

        # i > j -> lower triangular as 0
        for i in range(display_times.shape[0]):
            for j in range(display_times.shape[1]):
                if i > j: 
                    display_times[i, j] = display_times[j, i]
        
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

    with tab26:
        st.subheader("Elevation Absolute Difference Matrix (meters)")
        df_elev_abs = pd.DataFrame(elev_diff_abs_matrix, index=valid_ids, columns=valid_ids)

        def highlight_elev_abs(val):
            if val == 0:
                return "background-color: #000000; color: white"
            return "background-color: #000000; color: white"

        styled_elev_abs = df_elev_abs.style.applymap(highlight_elev_abs)
        st.dataframe(styled_elev_abs, use_container_width=True)

    with tab25:
        st.subheader("Elevation Difference Matrix (meters) - Todor Style")
        df_elev_todor = pd.DataFrame(elev_diff_todor_matrix, index=valid_ids, columns=valid_ids)
        
        def color_elev(val):
            if val == 0:
                return "background-color: #000000; color: white"
            else:
                return "background-color: #000000; color: white"
        
        styled_elev_todor = df_elev_todor.style.applymap(color_elev)
        st.dataframe(styled_elev_todor, use_container_width=True)

    with tab4:
        st.subheader("Graph View")
        label_mode = st.radio("Node labels:", ["Name", "ID"], horizontal=True)
        matrix_mode = st.radio(
            "Adjacency matrix:",
            ["Times", "Difference", "Absolute difference", "Todor difference", "Combination"],
            horizontal=True,
        )

        combination_matrix = None
        if matrix_mode == "Combination":
            st.markdown("**Combination settings**")
            diff_for_combination = st.radio(
                "Difference matrix for combination:",
                ["Difference", "Absolute difference", "Todor difference"],
                horizontal=True,
            )
            alpha = st.slider("Alpha (time weight)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
            meters_per_minute = st.number_input(
                "Equivalence (meters for 1 minute)",
                min_value=1.0,
                max_value=2000.0,
                value=100.0,
                step=5.0,
                help="Used to convert elevation difference into time-equivalent units before normalization.",
            )

            diff_matrix_map = {
                "Difference": elev_diff_matrix,
                "Absolute difference": elev_diff_abs_matrix,
                "Todor difference": elev_diff_todor_matrix,
            }
            selected_diff_matrix = diff_matrix_map[diff_for_combination]
            combination_matrix = build_combination_matrix(
                sub_matrix_times,
                selected_diff_matrix,
                alpha=alpha,
                meters_per_minute=meters_per_minute,
            )

        matrix_options = {
            "Times": (sub_matrix_times, True, "Edge labels are travel times in minutes."),
            "Difference": (elev_diff_matrix, False, "Edge labels are signed elevation differences in meters (only for finite-time routes)."),
            "Absolute difference": (elev_diff_abs_matrix, False, "Edge labels are absolute elevation differences in meters (only for finite-time routes)."),
            "Todor difference": (elev_diff_todor_matrix, False, "Edge labels are Todor elevation differences in meters (only for finite-time routes)."),
            "Combination": (combination_matrix, False, "Edge labels are combined normalized effort: alpha*time + (1-alpha)*difference."),
        }
        selected_matrix, skip_inf, caption = matrix_options[matrix_mode]
        st.caption(caption)

        availability_matrix = None if matrix_mode == "Times" else sub_matrix_times

        dot_graph = build_graphviz_dot(
            valid_ids,
            selected_matrix,
            node_dict,
            label_mode=label_mode,
            skip_inf=skip_inf,
            availability_matrix=availability_matrix,
        )
        st.graphviz_chart(dot_graph)
    
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
