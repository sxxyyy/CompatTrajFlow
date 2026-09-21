"""Traj-MLLM runner – zero-shot MLLM-based anomaly detection.

Wraps the Traj-MLLM visual pipeline (edge-ID → GPS → map images → MLLM API)
as a runner compatible with the ``AbstractRunner`` interface so it can be
benchmarked alongside the other 8 models.

Key characteristics
-------------------
* **Zero-shot** – ``train_step`` is a no-op; no model weights are learned.
* **Image pipeline** – trajectories are rendered as POI + Road-Network PNGs
  via Node.js/Puppeteer, then sent to an MLLM.
* **Discrete predictions** – the MLLM returns *Normal* / *Abnormal* labels.
  ``average_precision_score`` and ``roc_auc_score`` are computed on these
  binary labels.
* **Caching** – JSON, PNG, and MLLM responses are cached, so interrupted
  runs can be resumed without re-rendering or re-incurring API costs.

Requirements
------------
* Node.js + ``puppeteer`` npm package (for PNG rendering).
* ``openai`` Python package (for API calls).
* ``geopandas`` (for GeoJSON generation – already a project dependency).
"""

from __future__ import annotations

import json
import logging
import os
import random
from argparse import Namespace
from typing import Any

import numpy as np
from numpy import ndarray
from torch import Generator
from torch.utils.data import DataLoader

from model.traj_mllm.edge_geometry import EdgeGeometryMapper
from model.traj_mllm.mllm_client import MLLMClient
from model.traj_mllm.prompts import SYSTEM_PROMPT, build_user_content
from model.traj_mllm.visualization import TrajectoryVisualizer
from runner.abstract_runner import AbstractRunner

logger = logging.getLogger(__name__)


class TrajMLLMRunner(AbstractRunner):
    """Zero-shot MLLM anomaly-detection runner.

    Parameters
    ----------
    args : Namespace
        Must contain the standard framework args **plus** the MLLM-specific
        arguments defined in ``args.py``. The API key is read only from the
        ``OPENAI_API_KEY`` environment variable.
    generator : Generator
        Torch random generator (unused but required by the interface).
    """

    def __init__(self, args: Namespace, generator: Generator) -> None:
        super().__init__(args, generator)
        self.name = "TrajMLLM"

        if getattr(args, "use_grid_tokens", False):
            raise ValueError(
                "TrajMLLM runner does not support --use_grid_tokens. "
                "Grid tokens are abstract cell indices without real-world GPS "
                "coordinates, which are required for map-based visualisation."
            )

        # Read credentials only from the environment. Accepting an API key as a
        # command-line argument exposes it in shell history and process listings.
        self._api_key: str = os.environ.get("OPENAI_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "MLLM API key not set. Set the OPENAI_API_KEY environment variable."
            )

        self._model_name: str = getattr(args, "mllm_model_name", "o4-mini")
        self._base_url: str = getattr(
            args, "mllm_base_url", "https://api.openai.com/v1"
        )
        self._max_trajectories: int = getattr(args, "mllm_max_trajectories", 500)
        self._road_workers: int = getattr(args, "mllm_render_workers", 8)
        self._poi_workers: int = min(getattr(args, "mllm_poi_workers", 4), 16)
        self._random_seed: int = args.random_seed
        self._test_rendered = False
        self._test_visualizer: TrajectoryVisualizer | None = None

        # Only include the seed in the path when sampling is enabled,
        # otherwise deterministic runs share the same directory.
        _default_work = f"data/{args.location}/traj_mllm_work"
        if self._max_trajectories > 0:
            _default_work = os.path.join(_default_work, f"seed_{self._random_seed}")
        self._work_dir: str = getattr(args, "mllm_work_dir", None) or _default_work

        self._edge_mapper = EdgeGeometryMapper(
            location=args.location,
            edge_mapping=getattr(args, "edge_mapping", None),
            use_grid_tokens=False,
        )
        self._edge_mapper.build()  # build / load cache eagerly

        logger.info("TrajMLLM runner initialised. Work dir: %s", self._work_dir)
        logger.info("MLLM model: %s  |  base URL: %s", self._model_name, self._base_url)

    def train_step(
        self,
        dataloader: DataLoader,
        val_dataloader: DataLoader | None,
        anomaly_dataloader_dict: dict[str, DataLoader],
    ) -> list[dict[str, float]] | None:
        """MLLM is zero-shot - training is a no-op."""
        logger.info("TrajMLLM is zero-shot; skipping training.")
        return None

    def test_step(
        self, anomaly_dataloader_dict: dict[str, DataLoader]
    ) -> dict[str, float]:  # type: ignore[override] — abstract annotation is imprecise
        """Evaluate the MLLM on every anomaly type.

        Returns a single dict (to match the framework's
        ``AbstractRunner.test_step`` signature).
        """
        return self._get_test_result(anomaly_dataloader_dict)

    def _get_test_result(
        self, anomaly_dataloader_dict: dict[str, DataLoader]
    ) -> dict[str, dict[str, float]]:
        """Like ``AbstractRunner.get_test_result`` but preserves the full
        anomaly-type name for per-type subdirectory isolation."""
        anomaly_result: dict[str, dict[str, float]] = {}
        for anomaly_type, anomaly_dataloader in anomaly_dataloader_dict.items():
            logger.info("Start testing anomaly type: %s", anomaly_type)
            self.anomaly_type = anomaly_type  # full name, e.g. "detour_0.1"
            y_true, y_pred = self.test(anomaly_dataloader)
            metrice_score_dict = self.get_scores_with_different_metrices(y_true, y_pred)
            anomaly_result[anomaly_type] = metrice_score_dict
            for key, value in metrice_score_dict.items():
                logger.info(
                    "Anomaly Type: %s, Metric: %s, Score: %.4f",
                    anomaly_type,
                    key,
                    value,
                )
        return anomaly_result

    def test(self, dataloader: DataLoader) -> tuple[ndarray, ndarray]:
        """Run the full visual + MLLM pipeline on a single anomaly-type dataloader.

        Returns
        -------
        y_true : ndarray
            Ground-truth labels (1 = anomalous, 0 = normal).
        y_pred : ndarray
            Discrete MLLM predictions (1 = Abnormal, 0 = Normal).
        """
        trajectories: list[Any] = []
        labels: list[int] = []
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                for item in batch:
                    if isinstance(item, (list, tuple)) and len(item) == 2:
                        trajectories.append(item[0])
                        labels.append(int(item[1]))
                    else:
                        trajectories.append(item)
                        labels.append(-1)
            else:
                logger.warning("Unexpected batch format: %s", type(batch))
                trajectories.append(batch)
                labels.append(-1)

        n_total = len(trajectories)

        if self._max_trajectories > 0 and self._max_trajectories < n_total:
            normal_idx = [i for i, lb in enumerate(labels) if lb == 0]
            anom_idx = [i for i, lb in enumerate(labels) if lb == 1]
            n_anom = len(anom_idx)

            # Preserve the original anomaly ratio in the sampled subset.
            anom_ratio = n_anom / n_total
            keep_anom = max(1, min(n_anom, round(self._max_trajectories * anom_ratio)))
            keep_normal = self._max_trajectories - keep_anom

            rng = random.Random(self._random_seed)
            selected_normal_global = rng.sample(
                normal_idx, min(keep_normal, len(normal_idx))
            )
            selected_anom = rng.sample(anom_idx, min(keep_anom, n_anom))
            selected = sorted(selected_normal_global + selected_anom)

            # Indices are saved *relative to their source pickle files*:
            #   selected_anomaly_indices  → indices into the anomaly pickle
            #   selected_normal_indices   → indices into test_trajectories.pkl
            # This way any model can load the same pickle files and filter
            # directly, regardless of how it orders the combined dataset.
            subset_dir = os.path.join(self._work_dir, "subsets")
            os.makedirs(subset_dir, exist_ok=True)
            subset_file = os.path.join(
                subset_dir, f"{self.anomaly_type}_selected_indices.json"
            )
            subset_data = {
                "anomaly_type": self.anomaly_type,
                "random_seed": self._random_seed,
                "n_anomaly_total": n_anom,
                "n_normal_total": n_total - n_anom,
                "selected_anomaly_indices": sorted(selected_anom),
                "selected_normal_indices": sorted(
                    [i - n_anom for i in selected_normal_global]
                ),
                "max_trajectories": self._max_trajectories,
            }
            with open(subset_file, "w") as f:
                json.dump(subset_data, f, indent=2)
            logger.info(
                "Saved MLLM subset indices to %s (%d trajectories selected)",
                subset_file,
                len(selected),
            )

            trajectories = [trajectories[i] for i in selected]
            labels = [labels[i] for i in selected]
            n_total = len(trajectories)

        logger.info(
            "TrajMLLM test: %d trajectories (%d normal, %d anomalous).",
            n_total,
            labels.count(0),
            labels.count(1),
        )

        if self._test_visualizer is None:
            self._test_visualizer = TrajectoryVisualizer(
                work_dir=self._work_dir,
                location=self.location,
                edge_mapper=self._edge_mapper,
                poi_concurrency=self._poi_workers,
                road_concurrency=self._road_workers,
                subdir="test",
            )
            self._test_visualizer.prepare_road_network_geojson()

        anom_vis = TrajectoryVisualizer(
            work_dir=self._work_dir,
            location=self.location,
            edge_mapper=self._edge_mapper,
            poi_concurrency=self._poi_workers,
            road_concurrency=self._road_workers,
            subdir=self.anomaly_type,  # e.g. "detour_0.1"
        )

        n_anom = labels.count(1)

        if not self._test_rendered:
            logger.info("Rendering test (normal) trajectories (once)...")
            for idx in range(n_anom, n_total):
                traj = trajectories[idx]
                self._test_visualizer.generate_jsons(
                    traj_index=idx,
                    path=traj.path,
                    timestamps=traj.timestamps,
                )
            try:
                n_poi, n_road = self._test_visualizer.render_all()
                logger.info("Test images: %d POI, %d Road.", len(n_poi), len(n_road))
            except Exception as exc:
                logger.error("Test rendering failed: %s", exc)
            self._test_rendered = True

        for idx in range(n_anom):
            traj = trajectories[idx]
            anom_vis.generate_jsons(
                traj_index=idx,
                path=traj.path,
                timestamps=traj.timestamps,
            )

        try:
            n_poi, n_road = anom_vis.render_all()
            logger.info(
                "Anomaly (%s): %d POI, %d Road images.",
                self.anomaly_type,
                len(n_poi),
                len(n_road),
            )
        except Exception as exc:
            logger.error("Anomaly rendering (%s) failed: %s", self.anomaly_type, exc)

        client = MLLMClient(
            api_key=self._api_key,
            model_name=self._model_name,
            base_url=self._base_url,
            system_prompt=SYSTEM_PROMPT,
            cache_dir=os.path.join(self._work_dir, "mllm_cache"),
        )

        y_true_list: list[int] = []
        y_pred_list: list[int] = []

        for idx in range(n_total):
            traj = trajectories[idx]
            traj_id = str(idx)
            images: list[str] = []

            if idx >= n_anom:
                img_dict = self._test_visualizer.get_images_for_trajectory(idx)
            else:
                img_dict = anom_vis.get_images_for_trajectory(idx)
            images = img_dict.get("poi", []) + img_dict.get("road", [])

            if not images:
                logger.warning(
                    "Trajectory %d: no rendered images found; defaulting to Normal.",
                    idx,
                )
                y_true_list.append(labels[idx])
                y_pred_list.append(0)
                continue

            user_content = build_user_content(traj_id)
            cache_scope = "test" if idx >= n_anom else (self.anomaly_type or "anomaly")
            try:
                pred = client.predict(
                    idx, user_content, images, cache_scope=cache_scope
                )
            except Exception as exc:
                logger.error(
                    "Trajectory %d: MLLM inference failed: %s. Defaulting to Normal.",
                    idx,
                    exc,
                )
                pred = 0

            y_true_list.append(labels[idx])
            y_pred_list.append(pred)

            if (idx + 1) % 10 == 0 or idx == n_total - 1:
                logger.info(
                    "MLLM progress: %d/%d trajectories processed.", idx + 1, n_total
                )

        return np.array(y_true_list, dtype=np.int64), np.array(
            y_pred_list, dtype=np.int64
        )

    def train(self, dataloader: DataLoader) -> float:
        return 0.0

    def create_checkpoint_dir(self) -> None:
        pass

    def save_checkpoint(self) -> None:
        pass

    def load_checkpoint(self) -> None:
        pass

    def is_checkpoint_exists(self) -> bool:
        return False

    def free_vram(self) -> None:
        pass
