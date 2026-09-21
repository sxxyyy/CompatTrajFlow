import logging
import math
import os
import pickle

import geopandas as gpd
import networkx as nx
import numpy as np
from scipy import sparse

from utils.trajectory import Trajectory

logger = logging.getLogger(__name__)


def get_or_build_grid_mapping(
    location: str,
    height_km: float,
    width_km: float,
    trajectories: list[Trajectory] | None = None,
    return_num_tokens: bool = False,
) -> dict | tuple[dict, int, tuple[int, int]]:
    """
    Build a dense mapping from raw edge IDs to grid IDs.

    When trajectories are supplied, the grid boundary is derived from exactly the
    edges used by those trajectories. This is intentionally rebuilt per run
    because generated anomaly trajectories can change the required boundary.
    """
    mapping_path = (
        f"data/{location}/processed/grid_mapping_{height_km}_{width_km}_rect.pkl"
    )
    if trajectories is None and not return_num_tokens and os.path.exists(mapping_path):
        with open(mapping_path, "rb") as f:
            return pickle.load(f)

    logger.info(
        "Building dense grid mapping for %s with grid size %f km x %f km...",
        location,
        height_km,
        width_km,
    )
    edges_file = f"data/{location}/raw/edges.shp"
    gdf = gpd.read_file(edges_file)

    if trajectories is not None:
        trajectory_edges = {
            edge_id for trajectory in trajectories for edge_id in trajectory.path
        }
        if not trajectory_edges:
            raise ValueError("Cannot build grid mapping from empty trajectories.")
        gdf = gdf[gdf["fid"].isin(trajectory_edges)].copy()
        if gdf.empty:
            raise ValueError(
                "None of the trajectory edge IDs were found in the road network."
            )
        logger.info(
            "Using trajectory-derived grid boundary from %d trajectories and %d unique edges.",
            len(trajectories),
            len(trajectory_edges),
        )

    gdf = gdf.to_crs(gdf.estimate_utm_crs())

    min_x, min_y, max_x, max_y = gdf.total_bounds
    cell_width_m = width_km * 1000
    cell_height_m = height_km * 1000
    num_cols = max(1, math.ceil((max_x - min_x) / cell_width_m))
    num_rows = max(1, math.ceil((max_y - min_y) / cell_height_m))
    num_grid_tokens = num_rows * num_cols

    # Assign each edge to a cell by its centroid, while the grid boundary itself
    # is split from the full min/max geometry extent.
    gdf["centroid"] = gdf.geometry.centroid
    gdf["grid_x"] = ((gdf["centroid"].x - min_x) / cell_width_m).astype(int)
    gdf["grid_y"] = ((gdf["centroid"].y - min_y) / cell_height_m).astype(int)
    gdf["grid_x"] = gdf["grid_x"].clip(lower=0, upper=num_cols - 1)
    gdf["grid_y"] = gdf["grid_y"].clip(lower=0, upper=num_rows - 1)
    gdf["grid_id"] = gdf["grid_y"] * num_cols + gdf["grid_x"]

    logger.info(
        "Grid boundary for %s: min=(%.3f, %.3f), max=(%.3f, %.3f).",
        location,
        min_x,
        min_y,
        max_x,
        max_y,
    )
    logger.info(
        "Grid cell size for %s: %.3f km height x %.3f km width.",
        location,
        height_km,
        width_km,
    )
    logger.info(
        "Grid dimensions for %s: %d rows x %d columns = %d tokens.",
        location,
        num_rows,
        num_cols,
        num_grid_tokens,
    )

    raw_to_dense = dict(zip(gdf["fid"], gdf["grid_id"]))

    os.makedirs(f"data/{location}/processed", exist_ok=True)
    if trajectories is None:
        with open(mapping_path, "wb") as f:
            pickle.dump(raw_to_dense, f)

    if return_num_tokens:
        return raw_to_dense, num_grid_tokens, (num_rows, num_cols)
    return raw_to_dense


def get_or_build_edge_mapping(location: str) -> dict:
    """
    Build a dense mapping from raw edge IDs to 0..N.
    """
    mapping_path = f"data/{location}/processed/edge_mapping.pkl"
    if os.path.exists(mapping_path):
        with open(mapping_path, "rb") as f:
            return pickle.load(f)

    logger.info(
        "Building dense edge mapping for %s to reduce vocabulary size...", location
    )
    with open(f"data/{location}/processed/all_trajectories.pkl", "rb") as f:
        trajectories: list[Trajectory] = pickle.load(f)

    unique_edges = set()
    for t in trajectories:
        unique_edges.update(t.path)

    raw_to_dense = {
        raw_id: dense_id for dense_id, raw_id in enumerate(sorted(unique_edges))
    }

    with open(mapping_path, "wb") as f:
        pickle.dump(raw_to_dense, f)

    return raw_to_dense


def apply_edge_mapping(trajectories: list[Trajectory], mapping: dict):
    """
    In-memory map of trajectory paths to dense IDs.
    """
    for t in trajectories:
        new_path = []
        for raw_id in t.path:
            if raw_id in mapping:
                new_path.append(mapping[raw_id])
            else:
                logger.warning(
                    "Found unmapped edge ID %d in trajectory. Adding to mapping.",
                    raw_id,
                )
                new_id = len(set(mapping.values()))
                mapping[raw_id] = new_id
                new_path.append(new_id)
        t.path = new_path


def build_grid_adj_dnorm(
    num_rows: int, num_cols: int
) -> tuple[sparse.coo_matrix, sparse.csr_matrix]:
    """Build adjacency (A+I) and normalised degree (D^{-1/2}) matrices for a 2-D grid.

    Uses ``nx.grid_2d_graph`` for clean 4-connected neighbour construction and
    matches the ``A+I`` / ``D^{-1/2}`` convention from ``RoadNetwork.get_a_d_matrix``.

    Args:
        num_rows: Number of grid rows.
        num_cols: Number of grid columns.

    Returns:
        tuple: (adjacency matrix ``A+I`` in COO format, normalised degree matrix
        ``D^{-1/2}`` in CSR format).
    """

    G = nx.grid_2d_graph(num_rows, num_cols, periodic=False)
    # Fix row-major ordering so grid IDs are reproducible.
    nodelist = sorted(G.nodes())
    adj_no_self = nx.adjacency_matrix(G, nodelist=nodelist).astype(int).tocoo()
    identity = sparse.identity(num_rows * num_cols, dtype=int, format="coo")
    adj = (adj_no_self + identity).tocoo()

    degree_list = np.array(adj.sum(axis=1)).flatten()
    inv_sqrt_degree = 1.0 / (np.sqrt(degree_list) + 1e-10)
    inv_sqrt_degree[np.isinf(inv_sqrt_degree)] = 0.0
    d_norm = sparse.diags(inv_sqrt_degree, format="csr")

    logger.info(
        "Built grid adjacency: %d cells, %d edges (incl. self-loops).",
        num_rows * num_cols,
        adj.nnz,
    )

    return adj, d_norm
