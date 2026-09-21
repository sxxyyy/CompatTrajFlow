#!/usr/bin/env python3
"""IBAT (Isolation-Based Anomaly detection for Trajectories) evaluation script.

Uses an inverted index so the inner isolation loop uses set
intersections instead of O(n) list scans.

The train data and index are built once and reused across anomaly settings.

Usage:
    python scripts/run_ibat.py --random_seed "$SEED"
"""

import argparse
import logging
import os
import pickle
import sys
import time
from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from utils.trajectory import Trajectory  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("ibat")


class IBAT:
    """Isolation-Based Anomaly detection for Trajectories.

    Builds an inverted index ``edge -> set(train_idx)`` once so that each
    isolation step becomes a cheap set intersection.
    """

    def __init__(
        self,
        train_paths: list[list[int]],
        *,
        number_of_trials: int = 100,
        sub_sample_size: int = 256,
        random_seed: int,
    ) -> None:
        self.n_train = len(train_paths)
        self.number_of_trials = number_of_trials
        self.sub_sample_size = sub_sample_size
        self._rng = np.random.default_rng(random_seed)

        # Plain sets avoid an extra full copy during index construction.
        logger.info(
            "Building inverted index for %d training trajectories ...", self.n_train
        )
        _edge_to_indices: dict[int, set[int]] = defaultdict(set)
        for idx, p in enumerate(train_paths):
            for edge in p:
                _edge_to_indices[edge].add(idx)
        self._edge_to_indices = dict(_edge_to_indices)  # strip defaultdict wrapper
        logger.info("Inverted index: %d unique edges.", len(self._edge_to_indices))

        n = self.sub_sample_size
        self._c_n = 2.0 * (np.log(max(2, n - 1)) + np.euler_gamma) - 2.0 * (n - 1) / n

    def __call__(self, test_path: list[int]) -> float:
        """Return anomaly score in [0, 1] (higher = more anomalous)."""
        isolation_depths = np.empty(self.number_of_trials, dtype=np.float32)

        for trial in range(self.number_of_trials):
            sampled_idx = self._rng.choice(
                self.n_train, size=self.sub_sample_size, replace=False
            )
            remaining = set(sampled_idx.tolist())

            shuffled = list(test_path)
            self._rng.shuffle(shuffled)

            depth = 0
            for edge in shuffled:
                depth += 1
                candidates = self._edge_to_indices.get(edge)
                if candidates is None:
                    remaining.clear()
                else:
                    remaining.intersection_update(candidates)
                if not remaining:
                    break

            isolation_depths[trial] = depth

        return self._anomaly_score(isolation_depths)

    def _anomaly_score(self, isolation_depths: np.ndarray) -> float:
        """Convert isolation depths to an anomaly score in [0, 1]."""
        e_n_t = float(np.mean(isolation_depths))
        return float(np.power(2.0, -e_n_t / self._c_n))

    def score_batch(self, test_paths: list[list[int]]) -> np.ndarray:
        """Score a batch of test trajectories."""
        return np.array([self(tp) for tp in test_paths], dtype=np.float32)


def _path_from_traj(traj: Trajectory) -> list[int]:
    return traj.path


def load_pickle(path: str):
    logger.info("Loading %s ...", path)
    with open(path, "rb") as f:
        return pickle.load(f)


def load_trajectories(location: str, split: str) -> list[Trajectory]:
    path = f"data/{location}/processed/{split}_trajectories.pkl"
    return load_pickle(path)


def load_anomalies(
    location: str, anomaly_type: str, proportion: float, relax: int | None = None
) -> tuple[list[Trajectory], list[int]]:
    """Load anomaly trajectories + original indices."""
    anomaly_dir = f"data/{location}/anomaly"
    if anomaly_type == "switch" and relax is not None:
        fname = f"switch_{proportion}_relax_{relax}_test.pkl"
    elif anomaly_type == "time_shift":
        gap = relax if relax is not None else 3
        fname = f"time_shift_{proportion}_gap_{gap}_test.pkl"
    else:
        fname = f"{anomaly_type}_{proportion}_test.pkl"

    path = os.path.join(anomaly_dir, fname)
    trajectories, original_indices = load_pickle(path)
    logger.info(
        "  Loaded %d anomaly trajectories (type=%s, proportion=%.1f)",
        len(trajectories),
        anomaly_type,
        proportion,
    )
    return trajectories, original_indices


CONFIG = {
    "location": "xian",
    "anomaly_types": ["switch", "detour", "time_shift"],
    "proportions": [0.1, 0.3],
    "relax_values": {
        "switch": [3],
        "detour": [None],
        "time_shift": [30],
    },
    "partial_ratios": [1.0],
    "n_trials": 100,
    "sub_sample_size": 256,
}


def build_task_list(cfg: dict) -> list[dict]:
    """Flatten the config grid into a list of per-config dicts."""
    tasks = []
    for atype in cfg["anomaly_types"]:
        for prop in cfg["proportions"]:
            for relax in cfg["relax_values"].get(atype, [None]):
                for pr in cfg["partial_ratios"]:
                    tasks.append(
                        {
                            "anomaly_type": atype,
                            "proportion": prop,
                            "relax": relax,
                            "partial_ratio": pr,
                        }
                    )
    return tasks


def main(random_seed: int):
    cfg = CONFIG
    location = cfg["location"]

    t0 = time.time()

    logger.info("=== Loading train trajectories ===")
    train_trajs = load_trajectories(location, "train")
    val_trajs = load_trajectories(location, "val")
    train_paths = [_path_from_traj(t) for t in train_trajs]
    train_paths.extend(_path_from_traj(t) for t in val_trajs)
    del train_trajs, val_trajs  # free Trajectory objects, keep only paths

    logger.info("=== Loading test trajectories ===")
    test_trajs = load_trajectories(location, "test")
    n_test = len(test_trajs)
    base_test_paths = [_path_from_traj(t) for t in test_trajs]
    del test_trajs

    logger.info(
        "Train paths: %d  |  Test paths: %d  |  Load time: %.1f s",
        len(train_paths),
        n_test,
        time.time() - t0,
    )

    t_build = time.time()
    ibat = IBAT(
        train_paths,
        number_of_trials=cfg["n_trials"],
        sub_sample_size=cfg["sub_sample_size"],
        random_seed=random_seed,
    )
    del train_paths  # raw paths no longer needed, index is self-contained
    logger.info("IBAT build time: %.1f s", time.time() - t_build)

    tasks = build_task_list(cfg)
    logger.info("Total config combinations: %d", len(tasks))
    results = []

    for task in tasks:
        t_task = time.time()

        anomaly_trajs, anomaly_indices = load_anomalies(
            location,
            task["anomaly_type"],
            task["proportion"],
            task["relax"],
        )

        test_paths = list(base_test_paths)  # shallow copy of list
        for i, anom in zip(anomaly_indices, anomaly_trajs):
            test_paths[i] = _path_from_traj(anom)
        del anomaly_trajs

        pr = task["partial_ratio"]
        if pr < 1.0:
            test_paths = [p[: max(1, round(len(p) * pr))] for p in test_paths]

        y_true = np.zeros(n_test, dtype=np.int32)
        y_true[list(anomaly_indices)] = 1

        logger.info(
            "Scoring: type=%s proportion=%.1f relax=%s partial=%.1f ...",
            task["anomaly_type"],
            task["proportion"],
            task["relax"],
            pr,
        )
        scores = ibat.score_batch(test_paths)
        del test_paths

        y_pred = (scores >= 0.5).astype(np.int32)
        auroc = roc_auc_score(y_true=y_true, y_score=y_pred)
        ap = average_precision_score(y_true=y_true, y_score=y_pred)

        elapsed = time.time() - t_task
        result = {
            "location": location,
            "anomaly_type": task["anomaly_type"],
            "proportion": task["proportion"],
            "relax": task["relax"],
            "partial_ratio": pr,
            "n_trials": cfg["n_trials"],
            "sub_sample_size": cfg["sub_sample_size"],
            "random_seed": random_seed,
            "n_train": ibat.n_train,
            "n_test": n_test,
            "n_anomalies": len(anomaly_indices),
            "auroc": auroc,
            "avg_precision": ap,
            "elapsed_sec": elapsed,
        }
        results.append(result)
        logger.info("  -> AUROC = %.4f  AP = %.4f  (%.1f s)", auroc, ap, elapsed)

    total_elapsed = time.time() - t0
    print("\n" + "=" * 95)
    print(
        f"{'Type':<20s} {'Prop':>6s} {'Relax':>6s} {'Part.':>6s} "
        f"{'#Anom':>6s} {'AUROC':>8s} {'AP':>8s}  {'Time(s)':>8s}"
    )
    print("-" * 95)
    for r in results:
        print(
            f"{r['anomaly_type']:<20s} "
            f"{r['proportion']:>6.1f} "
            f"{str(r['relax']):>6s} "
            f"{r['partial_ratio']:>6.1f} "
            f"{r['n_anomalies']:>6d} "
            f"{r['auroc']:>8.4f} "
            f"{r['avg_precision']:>8.4f}  "
            f"{r['elapsed_sec']:>8.1f}"
        )
    print("=" * 95)
    logger.info("Total elapsed: %.1f s", total_elapsed)

    import pandas as pd

    os.makedirs("result/ibat", exist_ok=True)
    out_path = f"result/ibat/{location}_ibat_results.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, required=True)
    cli_args = parser.parse_args()
    main(cli_args.random_seed)
