"""Edge-to-GPS coordinate mapping for trajectory visualization.

Builds a mapping from **dense** edge IDs (as used in trajectory paths after
``apply_edge_mapping``) to full LineString GPS coordinates using the edges
shapefile and the edge-remapping dictionary produced by ``get_or_build_edge_mapping``.

Only supports edge-ID-based trajectories.  Grid-token mode is explicitly rejected
because the MLLM visual pipeline requires real-world GPS coordinates.
"""

import logging
import os
import pickle

import geopandas as gpd

logger = logging.getLogger(__name__)


class EdgeGeometryMapper:
    """Maps dense edge identifiers to full LineString (lon, lat) coordinates.

    Parameters
    ----------
    location : str
        City name (``"porto"`` or ``"xian"``).
    edge_mapping : dict[int, int] | None
        The ``raw_id → dense_id`` mapping built by
        :func:`~utils.edge_remapper.get_or_build_edge_mapping`.  When omitted
        the geometry mapping is built assuming identity (dense_id == raw_id).
    use_grid_tokens : bool
        If ``True``, an error is raised immediately because the MLLM visual
        pipeline cannot operate on abstract grid cells.
    grid_height_km : float, optional
    grid_width_km : float, optional
    """

    def __init__(
        self,
        location: str,
        edge_mapping: dict[int, int] | None = None,
        use_grid_tokens: bool = False,
        grid_height_km: float = 0.1,
        grid_width_km: float = 0.1,
    ) -> None:
        self._location = location
        self._edges_path = f"data/{location}/raw/edges.shp"
        self._cache_path = f"data/{location}/processed/edge_geom_mapping.pkl"
        # raw_id (= shapefile FID) → dense_id (= index in trajectory path).
        self._raw_to_dense: dict[int, int] = edge_mapping or {}

        if use_grid_tokens:
            raise ValueError(
                "Traj-MLLM runner does not support --use_grid_tokens. "
                "Grid tokens are abstract cell indices without real-world GPS "
                "coordinates, which are required for generating map-based "
                "visualisations. Please re-run without --use_grid_tokens, or "
                f"use a smaller grid (currently {grid_height_km}km x {grid_width_km}km) "
                "that still maps to edge IDs."
            )

        self._mapping: dict[int, list[tuple[float, float]]] = {}

    def build(self) -> dict[int, list[tuple[float, float]]]:
        """Build (or load cached) dense-edge-ID → full-LineString-coords mapping.

        The mapping is keyed by **dense** edge ID (the integer stored in
        ``Trajectory.path`` after ``apply_edge_mapping``) so that trajectory
        lookups are O(1) without any indirection.
        """
        if self._mapping:
            return self._mapping

        if os.path.exists(self._cache_path):
            logger.info(
                "Loading cached edge geometry mapping from %s", self._cache_path
            )
            with open(self._cache_path, "rb") as f:
                self._mapping = pickle.load(f)
            logger.info("Loaded %d edge geometry entries.", len(self._mapping))
            return self._mapping

        logger.info("Building edge geometry mapping from %s ...", self._edges_path)
        if not os.path.exists(self._edges_path):
            raise FileNotFoundError(
                f"Edges shapefile not found: {self._edges_path}. "
                "Run preprocessing first to generate the road network."
            )

        edges_gdf = gpd.read_file(self._edges_path)

        # Store the full LineString coordinates for smooth trajectory rendering.
        # Key by **dense** edge ID so trajectory lookups are direct (O(1)).
        has_remap = bool(self._raw_to_dense)
        for _, row in edges_gdf.iterrows():
            raw_fid = int(row["fid"])
            # If we have a remapping, use it to convert raw→dense; skip unused edges.
            if has_remap:
                dense_id = self._raw_to_dense.get(raw_fid)
                if dense_id is None:
                    continue  # edge not present in any trajectory
            else:
                dense_id = raw_fid

            geom = row["geometry"]
            # geom.coords returns [(lon, lat), ...] for EPSG:4326
            coords: list[tuple[float, float]] = [
                (round(c[0], 7), round(c[1], 7)) for c in geom.coords
            ]
            self._mapping[dense_id] = coords

        # Persist for subsequent runs.
        os.makedirs(os.path.dirname(self._cache_path), exist_ok=True)
        with open(self._cache_path, "wb") as f:
            pickle.dump(self._mapping, f)
        logger.info(
            "Built and cached %d edge geometry entries to %s.",
            len(self._mapping),
            self._cache_path,
        )
        return self._mapping

    def trajectory_to_gps(self, path: list[int]) -> list[tuple[float, float]]:
        """Convert a trajectory path (list of edge IDs) to GPS coordinates.

        Uses the full LineString geometry of each edge to produce a smooth
        road-following path.  Consecutive duplicate coordinates (shared edge
        endpoints) are removed.

        Parameters
        ----------
        path : list[int]
            Sequence of edge IDs.

        Returns
        -------
        list[tuple[float, float]]
            List of ``(longitude, latitude)`` tuples.  Missing edges are
            skipped with a warning.
        """
        mapping = self.build()
        coords: list[tuple[float, float]] = []
        for edge_id in path:
            edge_coords = mapping.get(edge_id)
            if edge_coords is None:
                logger.warning(
                    "Edge ID %d not found in edge geometry mapping; skipping.",
                    edge_id,
                )
                continue
            for pt in edge_coords:
                # Skip if identical to the previous point (shared endpoint).
                if not coords or pt != coords[-1]:
                    coords.append(pt)
        if not coords:
            raise ValueError(
                "No valid GPS coordinates could be derived from the trajectory "
                "path. All edge IDs were missing from the mapping."
            )
        return coords
