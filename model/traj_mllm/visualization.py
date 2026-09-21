"""Trajectory visualisation pipeline for MLLM-based anomaly detection.

Generates trajectory JSON files (with GPS coordinates derived from edge-centre
mapping), segments trajectories, converts road-network shapefiles to GeoJSON,
and invokes Node.js/Puppeteer to render PNG images.

Directory layout produced for a single run::

    <work_dir>/
        jsons/                  # per-trajectory JSON (global + 4 segment)
        images_poi/             # POI-view PNGs
        images_road/            # Road-Network-Structure PNGs
        html_poi/               # intermediate HTML (POI)
        html_road/              # intermediate HTML (Road Network)
        nodes.geojson           # road-network nodes (generated once)
        edges.geojson           # road-network edges (generated once)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

logger = logging.getLogger(__name__)

NUM_SEGMENTS = 4

_JS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "js_rendering")
_RENDER_POI = os.path.join(_JS_DIR, "render_poi.js")
_RENDER_ROAD = os.path.join(_JS_DIR, "render_road_structure.js")


def _cleanup_orphan_chrome() -> None:
    """Kill leftover headless Chrome processes from a previous Puppeteer run."""
    try:
        subprocess.run(
            ["pkill", "-f", "puppeteer_dev_chrome_profile"],
            capture_output=True,
            timeout=5,
        )
    except Exception:
        pass


class TrajectoryVisualizer:
    """Orchestrate JSON generation and PNG rendering for a set of trajectories.

    Parameters
    ----------
    work_dir : str
        Root directory where all intermediate and output files are placed.
    location : str
        City name (``"porto"`` or ``"xian"``).
    edge_mapper : EdgeGeometryMapper
        Pre-built edge-ID → GPS-coordinate mapper.
    concurrency : int
        Number of concurrent Puppeteer pages.
    subdir : str or None
        Optional sub-directory name to isolate outputs per anomaly type.
        When provided, JSON/PNG/HTML directories are placed under
        ``<work_dir>/<subdir>/``.  Road-network GeoJSON is always stored at
        the root ``work_dir`` level (shared across subdirs).
    """

    def __init__(
        self,
        work_dir: str,
        location: str,
        edge_mapper,
        poi_concurrency: int = 4,
        road_concurrency: int = 8,
        subdir: str | None = None,
    ) -> None:
        self._work_dir = work_dir
        self._location = location
        self._edge_mapper = edge_mapper
        self._poi_concurrency = poi_concurrency
        self._road_concurrency = road_concurrency

        base = os.path.join(work_dir, subdir) if subdir else work_dir

        self._json_dir = os.path.join(base, "jsons")
        self._poi_image_dir = os.path.join(base, "images_poi")
        self._road_image_dir = os.path.join(base, "images_road")
        self._poi_html_dir = os.path.join(base, "html_poi")
        self._road_html_dir = os.path.join(base, "html_road")
        self._nodes_geojson = os.path.join(work_dir, "nodes.geojson")
        self._edges_geojson = os.path.join(work_dir, "edges.geojson")

        for d in (
            self._json_dir,
            self._poi_image_dir,
            self._road_image_dir,
            self._poi_html_dir,
            self._road_html_dir,
        ):
            os.makedirs(d, exist_ok=True)

    def prepare_road_network_geojson(self) -> None:
        """Convert road-network shapefiles to GeoJSON (once per location)."""
        if os.path.exists(self._nodes_geojson) and os.path.exists(self._edges_geojson):
            logger.info("Road-network GeoJSON files already exist; skipping.")
            return

        import geopandas as gpd  # local import to keep the module lightweight

        for shp_key, geojson_path in [
            ("nodes", self._nodes_geojson),
            ("edges", self._edges_geojson),
        ]:
            shp_path = f"data/{self._location}/raw/{shp_key}.shp"
            if not os.path.exists(shp_path):
                raise FileNotFoundError(
                    f"Cannot generate GeoJSON: {shp_path} not found. "
                    "Run preprocessing first."
                )
            gdf = gpd.read_file(shp_path)
            cols = (
                ["fid", "u", "v", "geometry"]
                if shp_key == "edges"
                else ["osmid", "geometry"]
            )
            gdf = gdf[[c for c in cols if c in gdf.columns]]
            gdf.to_file(geojson_path, driver="GeoJSON")
            logger.info("Generated %s (%d features).", geojson_path, len(gdf))

    def generate_jsons(
        self,
        traj_index: int,
        path: list[int],
        timestamps: list[int],
    ) -> dict[str, str]:
        """Generate global + segmented JSON files for one trajectory.

        Parameters
        ----------
        traj_index : int
            Unique integer index used as the trajectory identifier (``devid``).
        path : list[int]
            Sequence of edge IDs forming the trajectory.
        timestamps : list[int]
            Unix timestamps corresponding to each point in *path*.

        Returns
        -------
        dict[str, str]
            Mapping ``{"global": <path>, "seg_0": <path>, ...}``.
        """
        traj_id = str(traj_index)

        global_path = os.path.join(self._json_dir, f"{traj_id}.json")
        result: dict[str, str] = {}

        # If the global JSON already exists, assume all segments exist too.
        if os.path.exists(global_path):
            result["global"] = global_path
            for seg_idx in range(NUM_SEGMENTS):
                seg_path = os.path.join(
                    self._json_dir, f"{traj_id}_segment_{seg_idx}.json"
                )
                if os.path.exists(seg_path):
                    result[f"seg_{seg_idx}"] = seg_path
            if len(result) == NUM_SEGMENTS + 1:
                return result

        gps_coords = self._edge_mapper.trajectory_to_gps(path)

        global_json: dict[str, Any] = {
            "devid": traj_id,
            "o_geo": gps_coords,  # [[lon, lat], ...]
            "times": timestamps,
            "road_ids": path,
        }
        self._write_json(global_path, global_json)
        result["global"] = global_path

        seg_length = max(2, len(gps_coords) // NUM_SEGMENTS)
        for seg_idx in range(NUM_SEGMENTS):
            start = seg_idx * seg_length
            end = min(len(gps_coords), start + seg_length)
            if start >= len(gps_coords):
                break
            seg_json: dict[str, Any] = {
                "devid": traj_id,
                "o_geo": gps_coords[start:end],
                "times": timestamps[start:end],
                "road_ids": path[start:end],
            }
            seg_path = os.path.join(self._json_dir, f"{traj_id}_segment_{seg_idx}.json")
            self._write_json(seg_path, seg_json)
            result[f"seg_{seg_idx}"] = seg_path

        return result

    def render_poi_images(self) -> list[str]:
        """Call Node.js to render POI-view PNGs for all JSON files.

        Returns
        -------
        list[str]
            Absolute paths to the generated PNG files.
        """
        self._run_node(
            _RENDER_POI,
            [
                "--json-dir",
                self._json_dir,
                "--image-dir",
                self._poi_image_dir,
                "--html-dir",
                self._poi_html_dir,
                "--concurrency",
                str(self._poi_concurrency),
            ],
        )
        return sorted(
            os.path.join(self._poi_image_dir, f)
            for f in os.listdir(self._poi_image_dir)
            if f.endswith(".png")
        )

    def render_road_structure_images(self) -> list[str]:
        """Call Node.js to render Road-Network-Structure PNGs.

        Returns
        -------
        list[str]
            Absolute paths to the generated PNG files.
        """
        self._run_node(
            _RENDER_ROAD,
            [
                "--json-dir",
                self._json_dir,
                "--image-dir",
                self._road_image_dir,
                "--html-dir",
                self._road_html_dir,
                "--nodes-geojson",
                self._nodes_geojson,
                "--edges-geojson",
                self._edges_geojson,
                "--concurrency",
                str(self._road_concurrency),
            ],
        )
        return sorted(
            os.path.join(self._road_image_dir, f)
            for f in os.listdir(self._road_image_dir)
            if f.endswith(".png")
        )

    def render_all(self) -> tuple[list[str], list[str]]:
        """Run POI and Road-Network rendering in parallel.

        Returns
        -------
        tuple[list[str], list[str]]
            ``(poi_image_paths, road_image_paths)``.
        """
        poi_result: list[str] = []
        road_result: list[str] = []
        errors: list[str] = []

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {
                pool.submit(self.render_poi_images): "poi",
                pool.submit(self.render_road_structure_images): "road",
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    if name == "poi":
                        poi_result = result
                        logger.info("POI rendering done: %d images.", len(poi_result))
                    else:
                        road_result = result
                        logger.info("Road rendering done: %d images.", len(road_result))
                except Exception as exc:
                    errors.append(name)
                    logger.error("%s rendering failed: %s", name, exc)

        if errors:
            raise RuntimeError(f"Rendering failed for: {', '.join(errors)}")
        return poi_result, road_result

    def get_images_for_trajectory(self, traj_index: int) -> dict[str, list[str]]:
        """Return image paths for a specific trajectory.

        Parameters
        ----------
        traj_index : int
            Trajectory identifier.

        Returns
        -------
        dict[str, list[str]]
            ``{"poi": [...], "road": [...]}`` where each list contains
            the global + segment PNG paths.
        """
        traj_id = str(traj_index)

        # Use precise prefix matching to avoid collisions (e.g. "1" matching "10").
        def _belongs_to_traj(fname: str) -> bool:
            return fname == f"{traj_id}.png" or fname.startswith(f"{traj_id}_")

        poi_imgs = sorted(
            os.path.join(self._poi_image_dir, f)
            for f in os.listdir(self._poi_image_dir)
            if _belongs_to_traj(f)
        )
        road_imgs = sorted(
            os.path.join(self._road_image_dir, f)
            for f in os.listdir(self._road_image_dir)
            if _belongs_to_traj(f)
        )
        return {"poi": poi_imgs, "road": road_imgs}

    def clear_jsons(self) -> None:
        """Remove JSON files to free disk space after rendering."""
        if os.path.isdir(self._json_dir):
            shutil.rmtree(self._json_dir)
            logger.info("Cleared JSON directory: %s", self._json_dir)

    @staticmethod
    def _write_json(filepath: str, data: dict[str, Any]) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @staticmethod
    def _find_node() -> str:
        """Locate a working Node.js binary.

        Node.js ≥23 enables experimental ESM/CJS detection by default which
        breaks CJS packages like puppeteer's yargs dependency.  We therefore
        prefer a system-installed Node (typically older and CJS-safe) over a
        conda-installed bleeding-edge version.
        """
        candidates: list[str] = []
        for p in ("/usr/bin/node", "/usr/local/bin/node"):
            if os.path.isfile(p):
                candidates.append(p)
        path_node = shutil.which("node")
        if path_node and path_node not in candidates:
            candidates.append(path_node)

        for node_bin in candidates:
            try:
                ver = subprocess.run(
                    [node_bin, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                major = int(ver.stdout.strip().lstrip("v").split(".")[0])
                if major < 23:
                    logger.info("Using Node.js %s (%s)", ver.stdout.strip(), node_bin)
                    return node_bin
            except Exception:
                continue

        if path_node:
            logger.warning(
                "Node.js ≥23 detected; rendering may fail due to ESM/CJS "
                "compatibility issues. Consider installing Node.js <23."
            )
            return path_node

        raise RuntimeError(
            "Node.js is not installed or not on PATH. "
            "The Traj-MLLM visual pipeline requires Node.js + Puppeteer. "
            "Install with: conda install nodejs && npm install puppeteer"
        )

    @staticmethod
    def _run_node(script: str, args: list[str]) -> None:
        """Execute a Node.js script via subprocess, streaming output live."""
        if not os.path.isfile(script):
            raise FileNotFoundError(f"JS script not found: {script}")

        node_bin = TrajectoryVisualizer._find_node()
        cmd = [node_bin, script] + args
        logger.info("Running: %s", " ".join(cmd))
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip("\n")
                if line:
                    logger.info("[node] %s", line)
            proc.wait()

            assert proc.stderr is not None
            stderr_text = proc.stderr.read()
        except Exception:
            _cleanup_orphan_chrome()
            raise

        if stderr_text:
            for line in stderr_text.strip().splitlines():
                logger.warning("[node:stderr] %s", line)
        if proc.returncode != 0:
            _cleanup_orphan_chrome()
            raise RuntimeError(
                f"Node.js script {os.path.basename(script)} exited with "
                f"code {proc.returncode}.\nSTDERR:\n{stderr_text}"
            )
        logger.info(
            "Node.js script %s completed successfully.", os.path.basename(script)
        )
