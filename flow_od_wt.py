import glob
import json
import logging
import math
import os
import pickle
import random
import sys
from argparse import ArgumentParser
from collections import defaultdict

import lightning as L
import torch
import torch.nn.functional as F
import wandb
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    RichModelSummary,
    RichProgressBar,
    WeightAveraging,
)
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor, nn, optim
from torch.nn.utils.rnn import pad_sequence
from torch.optim import lr_scheduler
from torch.optim.swa_utils import get_ema_multi_avg_fn
from torch.utils.data import DataLoader, Dataset
from torchmetrics import AUROC, AveragePrecision
from tqdm import tqdm

from dit import DiT
from dit_od import ODConditionedDiT
from utils.edge_remapper import apply_edge_mapping
from utils.timestamp_converter import TimestampConverter
from utils.trajectories_dataset import TokenTrajectoryDataset
from utils.trajectory import TokenTrajectory, Trajectory


def collate_fn(batch: list[tuple[TokenTrajectory, int]]):
    """
    Custom collate function to handle variable-length sequences.
    """
    paths, times, labels = zip(
        *[(traj.path, traj.time, labels) for traj, labels in batch]
    )

    tgt_path = [torch.tensor(p, dtype=torch.long) + 1 for p in paths]
    tgt_time = [torch.tensor(t, dtype=torch.long) for t in times]

    tgt_path = pad_sequence(tgt_path, batch_first=True, padding_value=0)
    tgt_time = pad_sequence(tgt_time, batch_first=True, padding_value=0)

    masks = tgt_path != 0

    labels = torch.tensor(labels, dtype=torch.long)
    return (tgt_path, tgt_time, masks), labels


class ConditionPairDataset(Dataset):
    """Attach globally selected hard conditions to targets.

    ``extra_target_indices`` optionally repeats selected targets without
    discarding any item from the base dataset. It is used only for training
    to make sparse X|T hard pairs visible often enough to learn from.
    """

    def __init__(
        self,
        base: TokenTrajectoryDataset,
        hard_time_indices: list[int],
        hard_path_indices: list[int],
        extra_target_indices: list[int] | None = None,
    ):
        if not (len(base) == len(hard_time_indices) == len(hard_path_indices)):
            raise ValueError("Condition-pair maps must match the base dataset length.")
        if any(
            hard_index == target_index
            for target_index, hard_index in enumerate(hard_time_indices)
        ) or any(
            hard_index == target_index
            for target_index, hard_index in enumerate(hard_path_indices)
        ):
            raise ValueError("Hard condition-pair maps must not contain self-matches.")
        self.base = base
        self.hard_time_indices = hard_time_indices
        self.hard_path_indices = hard_path_indices
        self.extra_target_indices = extra_target_indices or []
        if any(not 0 <= index < len(base) for index in self.extra_target_indices):
            raise ValueError("Extra condition-pair indices must index the base dataset.")

    def __len__(self):
        return len(self.base) + len(self.extra_target_indices)

    def __getitem__(self, index: int):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index >= len(self.base):
            index = self.extra_target_indices[index - len(self.base)]
        target, label = self.base[index]
        hard_time_index = self.hard_time_indices[index]
        hard_path_index = self.hard_path_indices[index]
        hard_time = (
            self.base.token_trajectories[hard_time_index]
            if hard_time_index >= 0
            else target
        )
        hard_path = (
            self.base.token_trajectories[hard_path_index]
            if hard_path_index >= 0
            else target
        )
        return (
            target,
            label,
            hard_time,
            hard_time_index >= 0,
            hard_path,
            hard_path_index >= 0,
        )

def condition_pair_collate_fn(batch):
    """Collate targets plus exact-length hard time/path conditions."""
    target_batch, labels = collate_fn([(item[0], item[1]) for item in batch])
    target_path, target_time, _ = target_batch
    hard_time = pad_sequence(
        [torch.tensor(item[2].time, dtype=torch.long) for item in batch],
        batch_first=True,
        padding_value=0,
    )
    hard_path = pad_sequence(
        [torch.tensor(item[4].path, dtype=torch.long) + 1 for item in batch],
        batch_first=True,
        padding_value=0,
    )
    if hard_time.shape != target_time.shape or hard_path.shape != target_path.shape:
        raise RuntimeError("Hard conditions must have exactly the target length.")
    return target_batch, labels, {
        "hard_time": hard_time,
        "hard_time_eligible": torch.tensor(
            [item[3] for item in batch], dtype=torch.bool
        ),
        "hard_path": hard_path,
        "hard_path_eligible": torch.tensor(
            [item[5] for item in batch], dtype=torch.bool
        ),
    }


class TokenTrajectoryDataModule(L.LightningDataModule):
    """
    Data module for loading token trajectories.
    """

    def __init__(
        self,
        location: str,
        timestamp_converter: TimestampConverter,
        random_seed: int,
        batch_size: int = 32,
        num_workers: int = 16,
        test_anomaly_type: str = "detour",
        test_anomaly_ratio: float = 0.3,
        test_anomaly_types: list[str] | None = None,
        switch_relax: int = 5,
        shift_time_gap: int = 3,
        edge_mapping: dict | None = None,
        mllm_subset_path: str | None = None,
    ):
        super().__init__()
        self.timestamp_converter = timestamp_converter
        self.random_seed = random_seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.location = location
        self.test_anomaly_type = test_anomaly_type
        self.test_anomaly_ratio = test_anomaly_ratio
        self.test_anomaly_types = test_anomaly_types
        self.switch_relax = switch_relax
        self.shift_time_gap = shift_time_gap
        self.edge_mapping = edge_mapping
        self.mllm_subset_path = mllm_subset_path
        self._num_edges: int | None = None

    @property
    def num_edges(self) -> int:
        """Total vocabulary size (including PAD token)."""
        if self._num_edges is None:
            raise RuntimeError(
                "Edge mapping not built yet. Call prepare_edge_mapping() first."
            )
        return self._num_edges

    def prepare_edge_mapping(self):
        """
        Pre-scan all datasets (train/val/test/anomaly) to build a complete
        edge mapping that covers every edge ID the model will ever see.

        This must be called BEFORE creating the ConditionalFlowMatching model,
        so that nn.Embedding is sized correctly.

        The mapping is cached to disk for reproducibility between train and
        test runs.
        """
        cache_path = f"data/{self.location}/processed/complete_edge_mapping.pkl"
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                self.edge_mapping = pickle.load(f)
            self._num_edges = len(self.edge_mapping) + 1
            logging.info(
                "Loaded cached complete edge mapping: %d unique edges.",
                self._num_edges - 1,
            )
            return self.edge_mapping

        logging.info(
            "Building complete edge mapping for %s from all datasets...",
            self.location,
        )

        all_edges: set[int] = set()
        dataset_paths = [
            f"data/{self.location}/processed/train_trajectories.pkl",
            f"data/{self.location}/processed/val_trajectories.pkl",
            f"data/{self.location}/processed/test_trajectories.pkl",
        ]
        # Include every anomaly variant so all loaders share one vocabulary.
        anomaly_glob_test = f"data/{self.location}/anomaly/*_test.pkl"
        anomaly_glob_val = f"data/{self.location}/anomaly/*_val.pkl"
        anomaly_paths = sorted(
            glob.glob(anomaly_glob_test) + glob.glob(anomaly_glob_val)
        )
        if anomaly_paths:
            logging.info(
                "Found %d anomaly file(s): %s",
                len(anomaly_paths),
                ", ".join(os.path.basename(p) for p in anomaly_paths),
            )
            dataset_paths.extend(anomaly_paths)
        else:
            logging.warning(
                "No anomaly files found in %s",
                f"data/{self.location}/anomaly/",
            )

        for path in dataset_paths:
            if not os.path.exists(path):
                logging.warning("Dataset file not found, skipping: %s", path)
                continue
            with open(path, "rb") as f:
                data = pickle.load(f)
            # Anomaly files store (trajectories, labels); split files store a list.
            trajectories: list = data if isinstance(data, list) else data[0]
            for traj in trajectories:
                all_edges.update(traj.path)

        if not all_edges:
            raise RuntimeError(
                f"No edges found in any dataset for location '{self.location}'."
            )

        self.edge_mapping = {
            raw_id: dense_id for dense_id, raw_id in enumerate(sorted(all_edges))
        }
        self._num_edges = len(self.edge_mapping) + 1  # +1 for PAD token

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(self.edge_mapping, f)

        logging.info(
            "Complete edge mapping built: %d unique edges (vocabulary size: %d).",
            self._num_edges - 1,
            self._num_edges,
        )
        return self.edge_mapping

    def _load_and_tokenize_trajectories(
        self, trajectories: list[Trajectory], desc: str, labels: int
    ) -> TokenTrajectoryDataset:
        """
        Helper method to load and tokenize trajectories.
        """
        token_data = []
        for traj in tqdm(trajectories, file=sys.stdout, desc=desc):
            token_data.append(
                TokenTrajectory(path=traj.path, time=self.timestamp_converter(traj))
            )
        return TokenTrajectoryDataset(token_data, labels=[labels] * len(token_data))

    def load_train_dataset(self):
        with open(f"data/{self.location}/processed/train_trajectories.pkl", "rb") as f:
            data: list[Trajectory] = pickle.load(f)

        assert self.edge_mapping is not None, (
            "prepare_edge_mapping() must be called first"
        )
        apply_edge_mapping(data, self.edge_mapping)
        return self._load_and_tokenize_trajectories(
            data, "Loading train trajectories", 0
        )

    def load_val_dataset(self):
        with open(f"data/{self.location}/processed/val_trajectories.pkl", "rb") as f:
            data: list[Trajectory] = pickle.load(f)

        assert self.edge_mapping is not None, (
            "prepare_edge_mapping() must be called first"
        )
        apply_edge_mapping(data, self.edge_mapping)
        return self._load_and_tokenize_trajectories(
            data, "Loading validation trajectories", 0
        )

    def _load_mllm_subset(self) -> dict | None:
        """Load MLLM-selected subset indices for the current anomaly type/ratio.

        Returns a dict with keys ``"anomaly_indices"`` and ``"normal_indices"``,
        each a list of ints (indices into the respective source pickle files),
        or ``None`` if no subset file is found.
        """
        if not self.mllm_subset_path:
            return None
        key = f"{self.test_anomaly_type}_{self.test_anomaly_ratio}"
        subset_file = os.path.join(
            self.mllm_subset_path, f"{key}_selected_indices.json"
        )
        if not os.path.exists(subset_file):
            logging.warning("MLLM subset file not found: %s", subset_file)
            return None
        with open(subset_file, "r") as f:
            data = json.load(f)
        logging.info(
            "MLLM subset loaded: %d anomaly + %d normal (from %s)",
            len(data["selected_anomaly_indices"]),
            len(data["selected_normal_indices"]),
            subset_file,
        )
        return {
            "anomaly_indices": data["selected_anomaly_indices"],
            "normal_indices": data["selected_normal_indices"],
        }

    def load_test_dataset_mllm_subset(self) -> TokenTrajectoryDataset:
        """Like :meth:`load_test_dataset`, but filters trajectories to the
        MLLM-selected subset **before** tokenization.

        Requires ``self.mllm_subset_path`` to be set.

        NOTE: Only supports single-anomaly-type mode (``test_anomaly_types``
        must be ``None``).
        """
        if self.test_anomaly_types:
            raise NotImplementedError(
                "MLLM subset mode does not yet support combined anomaly types."
            )

        subset = self._load_mllm_subset()
        if subset is None:
            raise FileNotFoundError(
                f"No MLLM subset file found under {self.mllm_subset_path} "
                f"for {self.test_anomaly_type}_{self.test_anomaly_ratio}"
            )

        with open(f"data/{self.location}/processed/test_trajectories.pkl", "rb") as f:
            test_data: list[Trajectory] = pickle.load(f)

        if self.test_anomaly_type == "switch":
            anomaly_file = f"data/{self.location}/anomaly/switch_{self.test_anomaly_ratio}_relax_{self.switch_relax}_test.pkl"
        elif self.test_anomaly_type == "time_shift":
            anomaly_file = f"data/{self.location}/anomaly/time_shift_{self.test_anomaly_ratio}_gap_{self.shift_time_gap}_test.pkl"
        else:
            anomaly_file = f"data/{self.location}/anomaly/{self.test_anomaly_type}_{self.test_anomaly_ratio}_test.pkl"

        with open(anomaly_file, "rb") as f:
            anomaly_data: tuple[list[Trajectory], list[int]] = pickle.load(f)
        anomaly_trajs: list[Trajectory] = anomaly_data[0]

        assert self.edge_mapping is not None, (
            "prepare_edge_mapping() must be called first"
        )
        apply_edge_mapping(test_data, self.edge_mapping)
        apply_edge_mapping(anomaly_trajs, self.edge_mapping)

        n_test_before = len(test_data)
        n_anom_before = len(anomaly_trajs)
        test_data = [test_data[i] for i in subset["normal_indices"]]
        anomaly_trajs = [anomaly_trajs[i] for i in subset["anomaly_indices"]]
        logging.info(
            "MLLM subset filter: test %d→%d, anomaly %d→%d",
            n_test_before,
            len(test_data),
            n_anom_before,
            len(anomaly_trajs),
        )

        test_dataset = self._load_and_tokenize_trajectories(
            test_data, "Loading test trajectories (subset)", 0
        )
        anomaly_dataset = self._load_and_tokenize_trajectories(
            anomaly_trajs, "Loading anomaly trajectories (subset)", 1
        )

        return TokenTrajectoryDataset(
            test_dataset.token_trajectories + anomaly_dataset.token_trajectories,
            labels=test_dataset.labels + anomaly_dataset.labels,
        )

    def load_test_dataset(self) -> TokenTrajectoryDataset:
        """
        Load test trajectories from a pickle file.
        When test_anomaly_types is set, load multiple anomaly types equally divided
        with subsampling so the total anomaly count stays within test_anomaly_ratio.
        """
        with open(f"data/{self.location}/processed/test_trajectories.pkl", "rb") as f:
            data: list[Trajectory] = pickle.load(f)

        assert self.edge_mapping is not None, (
            "prepare_edge_mapping() must be called first"
        )
        apply_edge_mapping(data, self.edge_mapping)
        test_dataset = self._load_and_tokenize_trajectories(
            data, "Loading test trajectories", 0
        )

        if self.test_anomaly_types:
            num_types = len(self.test_anomaly_types)
            per_type_count = int(len(data) * self.test_anomaly_ratio / num_types)
            all_anomaly_trajs: list[TokenTrajectory] = []
            all_anomaly_labels: list[int] = []

            for atype in self.test_anomaly_types:
                if atype == "switch":
                    anomaly_file = f"data/{self.location}/anomaly/switch_0.3_relax_{self.switch_relax}_test.pkl"
                elif atype == "time_shift":
                    anomaly_file = f"data/{self.location}/anomaly/time_shift_0.3_gap_{self.shift_time_gap}_test.pkl"
                else:
                    anomaly_file = f"data/{self.location}/anomaly/{atype}_0.3_test.pkl"

                with open(anomaly_file, "rb") as f:
                    anomaly_data: tuple[list[Trajectory], list[int]] = pickle.load(f)

                apply_edge_mapping(anomaly_data[0], self.edge_mapping)

                rng = random.Random(self.random_seed)
                if len(anomaly_data[0]) > per_type_count:
                    sampled_indices = rng.sample(
                        range(len(anomaly_data[0])), per_type_count
                    )
                    sampled_trajs = [anomaly_data[0][i] for i in sampled_indices]
                else:
                    sampled_trajs = anomaly_data[0]

                anomaly_dataset = self._load_and_tokenize_trajectories(
                    sampled_trajs,
                    f"Loading {atype} anomaly trajectories",
                    1,
                )
                all_anomaly_trajs.extend(anomaly_dataset.token_trajectories)
                all_anomaly_labels.extend(anomaly_dataset.labels)

            print(
                f"Combined {num_types} anomaly types: "
                f"{len(all_anomaly_trajs)} anomalies + {len(test_dataset.token_trajectories)} normal "
                f"= {len(test_dataset.token_trajectories) + len(all_anomaly_trajs)} total"
            )
            return TokenTrajectoryDataset(
                test_dataset.token_trajectories + all_anomaly_trajs,
                labels=test_dataset.labels + all_anomaly_labels,
            )

        if self.test_anomaly_type == "switch":
            anomaly_file = f"data/{self.location}/anomaly/switch_{self.test_anomaly_ratio}_relax_{self.switch_relax}_test.pkl"
        elif self.test_anomaly_type == "time_shift":
            anomaly_file = f"data/{self.location}/anomaly/time_shift_{self.test_anomaly_ratio}_gap_{self.shift_time_gap}_test.pkl"
        else:
            anomaly_file = f"data/{self.location}/anomaly/{self.test_anomaly_type}_{self.test_anomaly_ratio}_test.pkl"

        with open(anomaly_file, "rb") as f:
            anomaly_data: tuple[list[Trajectory], list[int]] = pickle.load(f)

        apply_edge_mapping(anomaly_data[0], self.edge_mapping)
        anomaly_dataset = self._load_and_tokenize_trajectories(
            anomaly_data[0], "Loading anomaly trajectories", 1
        )

        return TokenTrajectoryDataset(
            test_dataset.token_trajectories + anomaly_dataset.token_trajectories,
            labels=test_dataset.labels + anomaly_dataset.labels,
        )

    def load_val_anomaly_dataset(self) -> TokenTrajectoryDataset:
        """
        Load validation trajectories + validation anomaly trajectories.
        Mirrors :meth:`load_test_dataset` but uses val trajectories and
        ``_val.pkl`` anomaly files.
        """
        with open(f"data/{self.location}/processed/val_trajectories.pkl", "rb") as f:
            data: list[Trajectory] = pickle.load(f)

        assert self.edge_mapping is not None, (
            "prepare_edge_mapping() must be called first"
        )
        apply_edge_mapping(data, self.edge_mapping)
        val_dataset = self._load_and_tokenize_trajectories(
            data, "Loading validation trajectories", 0
        )

        if self.test_anomaly_types:
            num_types = len(self.test_anomaly_types)
            per_type_count = int(len(data) * self.test_anomaly_ratio / num_types)
            all_anomaly_trajs: list[TokenTrajectory] = []
            all_anomaly_labels: list[int] = []

            for atype in self.test_anomaly_types:
                if atype == "switch":
                    anomaly_file = (
                        f"data/{self.location}/anomaly/"
                        f"switch_{self.test_anomaly_ratio}_relax_{self.switch_relax}_val.pkl"
                    )
                elif atype == "time_shift":
                    anomaly_file = (
                        f"data/{self.location}/anomaly/"
                        f"time_shift_{self.test_anomaly_ratio}_gap_{self.shift_time_gap}_val.pkl"
                    )
                else:
                    anomaly_file = (
                        f"data/{self.location}/anomaly/"
                        f"{atype}_{self.test_anomaly_ratio}_val.pkl"
                    )

                with open(anomaly_file, "rb") as f:
                    anomaly_data: tuple[list[Trajectory], list[int]] = pickle.load(f)

                apply_edge_mapping(anomaly_data[0], self.edge_mapping)

                rng = random.Random(self.random_seed)
                if len(anomaly_data[0]) > per_type_count:
                    sampled_indices = rng.sample(
                        range(len(anomaly_data[0])), per_type_count
                    )
                    sampled_trajs = [anomaly_data[0][i] for i in sampled_indices]
                else:
                    sampled_trajs = anomaly_data[0]

                anomaly_dataset = self._load_and_tokenize_trajectories(
                    sampled_trajs,
                    f"Loading {atype} validation anomaly trajectories",
                    1,
                )
                all_anomaly_trajs.extend(anomaly_dataset.token_trajectories)
                all_anomaly_labels.extend(anomaly_dataset.labels)

            print(
                f"Combined {num_types} val anomaly types: "
                f"{len(all_anomaly_trajs)} anomalies + {len(val_dataset.token_trajectories)} normal "
                f"= {len(val_dataset.token_trajectories) + len(all_anomaly_trajs)} total"
            )
            return TokenTrajectoryDataset(
                val_dataset.token_trajectories + all_anomaly_trajs,
                labels=val_dataset.labels + all_anomaly_labels,
            )

        if self.test_anomaly_type == "switch":
            anomaly_file = (
                f"data/{self.location}/anomaly/"
                f"switch_{self.test_anomaly_ratio}_relax_{self.switch_relax}_val.pkl"
            )
        elif self.test_anomaly_type == "time_shift":
            anomaly_file = (
                f"data/{self.location}/anomaly/"
                f"time_shift_{self.test_anomaly_ratio}_gap_{self.shift_time_gap}_val.pkl"
            )
        else:
            anomaly_file = (
                f"data/{self.location}/anomaly/"
                f"{self.test_anomaly_type}_{self.test_anomaly_ratio}_val.pkl"
            )

        with open(anomaly_file, "rb") as f:
            anomaly_data: tuple[list[Trajectory], list[int]] = pickle.load(f)

        apply_edge_mapping(anomaly_data[0], self.edge_mapping)
        anomaly_dataset = self._load_and_tokenize_trajectories(
            anomaly_data[0], "Loading validation anomaly trajectories", 1
        )

        return TokenTrajectoryDataset(
            val_dataset.token_trajectories + anomaly_dataset.token_trajectories,
            labels=val_dataset.labels + anomaly_dataset.labels,
        )

    def _build_condition_pair_dataset(
        self, dataset: TokenTrajectoryDataset
    ) -> ConditionPairDataset:
        """Build deterministic, global same-OD/exact-length hard pairs."""
        trajectories = dataset.token_trajectories
        groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        for index, trajectory in enumerate(trajectories):
            groups[(trajectory.path[0], trajectory.path[-1], len(trajectory.path))].append(
                index
            )

        hard_time_indices = [-1] * len(trajectories)
        hard_path_indices = [-1] * len(trajectories)
        for indices in groups.values():
            if len(indices) < 2:
                continue

            # At most 24 representatives are sufficient for the four-hour
            # cyclic departure-time constraint. This keeps construction linear
            # in group size instead of scanning every pair in a large OD group.
            first_by_hour: dict[int, int] = {}
            for candidate_index in indices:
                hour = (
                    trajectories[candidate_index].time[0] % (24 * 60 * 60)
                ) // 3600
                first_by_hour.setdefault(hour, candidate_index)

            first_path_index = indices[0]
            first_path = trajectories[first_path_index].path
            different_path_index = next(
                (
                    candidate_index
                    for candidate_index in indices[1:]
                    if trajectories[candidate_index].path != first_path
                ),
                -1,
            )

            for target_index in indices:
                target = trajectories[target_index]
                target_hour = (target.time[0] % (24 * 60 * 60)) // 3600
                hard_time_indices[target_index] = min(
                    (
                        candidate_index
                        for hour, candidate_index in first_by_hour.items()
                        if min(abs(target_hour - hour), 24 - abs(target_hour - hour))
                        >= 4
                    ),
                    default=-1,
                )
                if target.path != first_path:
                    hard_path_indices[target_index] = first_path_index
                else:
                    hard_path_indices[target_index] = different_path_index

        hard_time_count = sum(index >= 0 for index in hard_time_indices)
        hard_path_count = sum(index >= 0 for index in hard_path_indices)
        logging.info(
            "Global hard-condition coverage: X|T,OD=%d/%d (%.2f%%), "
            "T|X=%d/%d (%.2f%%)",
            hard_time_count,
            len(dataset),
            100.0 * hard_time_count / max(len(dataset), 1),
            hard_path_count,
            len(dataset),
            100.0 * hard_path_count / max(len(dataset), 1),
        )
        return ConditionPairDataset(
            dataset, hard_time_indices, hard_path_indices
        )

    def _create_dataloader(
        self, dataset, shuffle: bool, batch_collate_fn=collate_fn
    ) -> DataLoader:
        """Factory method for DataLoader creation."""
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=batch_collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = self.load_train_dataset()
            self.val_dataset = self.load_val_dataset()
            self.val_condition_dataset = self._build_condition_pair_dataset(
                self.val_dataset
            )
        elif stage == "validate":
            self.val_dataset = self.load_val_dataset()
            self.val_condition_dataset = self._build_condition_pair_dataset(
                self.val_dataset
            )
        elif stage == "test":
            self.test_dataset = self.load_test_dataset()
        elif stage == "val_anomaly":
            self.val_anomaly_dataset = self.load_val_anomaly_dataset()
        elif stage == "mllm_subset":
            self.test_dataset = self.load_test_dataset_mllm_subset()
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def train_dataloader(self):
        return self._create_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._create_dataloader(
            self.val_condition_dataset,
            shuffle=False,
            batch_collate_fn=condition_pair_collate_fn,
        )

    def test_dataloader(self):
        return self._create_dataloader(self.test_dataset, shuffle=False)

    def val_anomaly_dataloader(self):
        return self._create_dataloader(self.val_anomaly_dataset, shuffle=False)


class TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.is_workday_emb = nn.Embedding(
            num_embeddings=2 + 1, embedding_dim=embedding_dim, padding_idx=0
        )
        self.hour_emb = nn.Embedding(
            num_embeddings=24 + 1, embedding_dim=embedding_dim, padding_idx=0
        )
        self.minute_emb = nn.Embedding(
            num_embeddings=60 + 1, embedding_dim=embedding_dim, padding_idx=0
        )
        self.second_emb = nn.Embedding(
            num_embeddings=60 + 1, embedding_dim=embedding_dim, padding_idx=0
        )

    def forward(self, time_features: Tensor) -> Tensor:
        is_workday = time_features[:, :, 0]
        hour = time_features[:, :, 1]
        minute = time_features[:, :, 2]
        second = time_features[:, :, 3]

        is_workday_emb = self.is_workday_emb(is_workday)
        hour_emb = self.hour_emb(hour)
        minute_emb = self.minute_emb(minute)
        second_emb = self.second_emb(second)
        time_emb = is_workday_emb + hour_emb + minute_emb + second_emb
        return time_emb


class ConditionalFlowMatching(L.LightningModule):
    def __init__(
        self,
        num_edges: int,
        emb_dim: int,
        P_mean: float,
        P_std: float,
        cfg_interval: tuple[float, float],
        t_eps: float,
        cfg_drop_prob: float,
        cfg_warmup_epochs: int,
        guidance_scale: float,
        learning_rate: float,
        dit_config: dict,
        score_cache_path: str | None = None,
        test_t: float = 0.1,
        warmup_steps: int = 1,
        lambda_ce: float = 0.25,
        condition_val_t: float = 0.1,
        condition_val_ts: tuple[float, ...] | list[float] | None = None,
        condition_val_max_batches: int = 32,
        t_sampling: str = "logit_normal",
    ):
        super().__init__()
        self.save_hyperparameters()
        if t_sampling not in {"uniform", "logit_normal"}:
            raise ValueError(
                "t_sampling must be either 'uniform' or 'logit_normal', "
                f"received {t_sampling!r}"
            )
        self.P_mean = P_mean
        self.P_std = P_std
        self.t_sampling = t_sampling
        self.t_eps = t_eps
        self.cfg_interval = cfg_interval
        self.emb_dim = emb_dim
        self.cfg_drop_prob = cfg_drop_prob
        self.cfg_warmup_epochs = cfg_warmup_epochs
        self.guidance_scale = guidance_scale
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.dit_config = dit_config
        self.score_cache_path = score_cache_path
        if not 0 < test_t < 1:
            raise ValueError("test_t must lie strictly between zero and one")
        self.test_t = test_t
        if not 0 < condition_val_t < 1:
            raise ValueError("condition_val_t must lie strictly between zero and one")
        if condition_val_ts is None:
            condition_val_ts = (condition_val_t,)
        else:
            condition_val_ts = tuple(float(t) for t in condition_val_ts)
            if not condition_val_ts:
                raise ValueError("condition_val_ts must contain at least one value")
            if any(not 0 < t < 1 for t in condition_val_ts):
                raise ValueError(
                    "every condition_val_ts value must lie strictly between zero and one"
                )
            if len(set(condition_val_ts)) != len(condition_val_ts):
                raise ValueError("condition_val_ts values must be unique")
        if condition_val_max_batches < 0:
            raise ValueError("condition_val_max_batches must be non-negative")
        self.condition_val_t = condition_val_t
        self.condition_val_ts = condition_val_ts
        self.condition_val_max_batches = condition_val_max_batches

        self._test_path_scores: list[Tensor] = []
        self._test_time_scores: list[Tensor] = []
        self._test_labels: list[Tensor] = []

        self.path_embedding = nn.Embedding(
            num_embeddings=num_edges, embedding_dim=emb_dim, padding_idx=0
        )
        self.time_embedding = TimeEmbedding(embedding_dim=emb_dim)

        self.time_embedding_weights = [
            self.time_embedding.is_workday_emb,
            self.time_embedding.hour_emb,
            self.time_embedding.minute_emb,
            self.time_embedding.second_emb,
        ]

        self.uncond_path_emb = nn.Parameter(torch.randn(emb_dim))
        # OD is the complete Path-side CFG condition in this factorization.
        self.uncond_od_emb = nn.Parameter(torch.randn(2 * emb_dim))

        self.lambda_ce = lambda_ce
        self.path_temperature = nn.Parameter(torch.tensor(0.07))
        self.time_temperature = nn.Parameter(torch.tensor(0.07))

        self.path_net = ODConditionedDiT(**self.dit_config)
        self.time_net = DiT(**self.dit_config)

        self.ce_loss = nn.CrossEntropyLoss(reduction="mean", ignore_index=0)
        self.mse_loss = nn.MSELoss(reduction="mean")

        self.auroc = AUROC(task="binary")
        self.ap = AveragePrecision(task="binary")

    def _compute_tied_logits(
        self, latents: Tensor, embedding_layer: nn.Embedding, temperature: Tensor
    ) -> Tensor:
        # Normalize representations for cosine similarity
        latents_norm = F.normalize(latents, p=2, dim=-1)
        weight_norm = F.normalize(embedding_layer.weight, p=2, dim=-1)

        # Compute cosine similarity and scale by temperature
        cos_sim = torch.matmul(latents_norm, weight_norm.t())
        logits = cos_sim / temperature
        return logits

    def _apply_classifier_free_dropout(
        self,
        cond_emb: Tensor,
        uncond_emb: Tensor,
        return_drop_mask: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        current_drop_prob = (
            0.0 if self.current_epoch < self.cfg_warmup_epochs else self.cfg_drop_prob
        )
        drop = torch.rand(cond_emb.shape[0], device=cond_emb.device) < current_drop_prob
        uncond_emb_reshaped = uncond_emb.view(
            *([1] * (cond_emb.ndim - 1)), -1
        ).expand_as(cond_emb)
        out = torch.where(
            drop.view(-1, *([1] * (cond_emb.ndim - 1))), uncond_emb_reshaped, cond_emb
        )
        if return_drop_mask:
            return out, drop
        return out

    def _od_context(self, path_emb: Tensor, masks: Tensor) -> Tensor:
        """Return untouched first/last path-edge embeddings as OD context."""
        last_index = masks.sum(dim=1).long().clamp_min(1) - 1
        batch_index = torch.arange(path_emb.size(0), device=path_emb.device)
        first = path_emb[:, 0]
        last = path_emb[batch_index, last_index]
        return torch.cat([first, last], dim=-1)

    def _path_condition(
        self, time_emb: Tensor, path_emb: Tensor, masks: Tensor
    ) -> Tensor:
        """Build the OD-only condition for path generation."""
        valid = masks.unsqueeze(-1).to(dtype=time_emb.dtype)
        od = self._od_context(path_emb, masks).unsqueeze(1)
        return od.expand(-1, time_emb.size(1), -1) * valid

    def _unconditional_path_condition(
        self, path_emb: Tensor, masks: Tensor
    ) -> Tensor:
        """Return the OD-only reference condition used by Path-side CFG."""
        condition = self.uncond_od_emb.view(1, 1, -1).expand(
            path_emb.size(0), path_emb.size(1), -1
        )
        return condition * masks.unsqueeze(-1).to(dtype=path_emb.dtype)

    def _path_nll_per_sample(
        self, prediction: Tensor, path: Tensor, masks: Tensor
    ) -> Tensor:
        logits = self._compute_tied_logits(
            prediction, self.path_embedding, self.path_temperature
        )
        token_nll = -torch.gather(
            F.log_softmax(logits, dim=-1), -1, path.unsqueeze(-1)
        ).squeeze(-1)
        valid_length = masks.sum(dim=1).clamp_min(1)
        return (token_nll * masks).sum(dim=1) / valid_length

    def _time_nll_per_sample(
        self, prediction: Tensor, time: Tensor, masks: Tensor
    ) -> Tensor:
        total_nll = torch.zeros_like(masks, dtype=prediction.dtype)
        for feature, embedding_layer in enumerate(self.time_embedding_weights):
            logits = self._compute_tied_logits(
                prediction, embedding_layer, self.time_temperature
            )
            feature_nll = -torch.gather(
                F.log_softmax(logits, dim=-1),
                -1,
                time[:, :, feature].unsqueeze(-1),
            ).squeeze(-1)
            total_nll += feature_nll
        valid_length = masks.sum(dim=1).clamp_min(1)
        return (total_nll * masks).sum(dim=1) / (4.0 * valid_length)

    def _sample_t(self, n: int, device: torch.device) -> Tensor:
        if self.t_sampling == "uniform":
            return torch.rand(n, device=device)
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return F.sigmoid(z)

    def _transform_time(self, raw_time: Tensor, masks: Tensor) -> Tensor:
        day_seconds = 24 * 60 * 60

        # Keep padded tokens as zero by applying the path mask after conversion.
        is_workday = (raw_time < day_seconds).long()
        timestamp_rem = raw_time % day_seconds
        hour = timestamp_rem // 3600
        minute = (timestamp_rem % 3600) // 60
        second = timestamp_rem % 60

        time_features = torch.stack([is_workday, hour, minute, second], dim=-1) + 1
        return time_features * masks.unsqueeze(-1).long()

    def _forward_sample(
        self, net, z: Tensor, t: Tensor, cond: Tensor, uncond_emb: Tensor
    ) -> Tensor:
        x_cond = net(z, t.flatten(), cond)
        v_cond = (x_cond - z) / (1 - t).clamp_min(self.t_eps)

        if uncond_emb.ndim == cond.ndim:
            uncond_emb_reshaped = uncond_emb
        else:
            uncond_emb_reshaped = uncond_emb.view(
                *([1] * (cond.ndim - 1)), -1
            ).expand_as(cond)
        x_uncond = net(z, t.flatten(), uncond_emb_reshaped)
        v_uncond = (x_uncond - z) / (1 - t).clamp_min(self.t_eps)

        low, high = self.cfg_interval
        interval_mask = (t < high) & (t >= low)
        cfg_scale_interval = torch.where(interval_mask, self.guidance_scale, 1.0)

        return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    @torch.no_grad()
    def _euler_step(
        self,
        z: Tensor,
        t: Tensor,
        t_next: Tensor,
        cond: Tensor,
        uncond_emb: Tensor,
        net,
    ) -> Tensor:
        v_pred = self._forward_sample(net, z, t, cond=cond, uncond_emb=uncond_emb)
        z_next = z + v_pred * (t_next - t)
        return z_next

    def masked_mse_loss(self, pred: Tensor, target: Tensor, masks: Tensor) -> Tensor:
        valid_pred = pred[masks]
        valid_target = target[masks]
        return self.mse_loss(valid_pred, valid_target)

    def _compute_logit_entropy(self, logits: Tensor, masks: Tensor) -> Tensor:
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
        return (entropy * masks.float()).sum() / masks.sum().clamp_min(1)

    def _compute_ce_loss(
        self,
        pred_latents: Tensor,
        target_idx: Tensor,
        embedding_layer: nn.Embedding,
        temperature: Tensor,
    ) -> Tensor:
        """
        Computes Cross-Entropy loss using weight tying and cosine similarity.
        """

        logits = self._compute_tied_logits(pred_latents, embedding_layer, temperature)
        return self.ce_loss(
            logits.view(-1, logits.size(-1)), target_idx.view(-1).long()
        )

    def on_validation_epoch_start(self) -> None:
        names = (
            "path_correct_sum",
            "path_uncond_sum",
            "path_hard_sum",
            "path_hard_win_sum",
            "path_sensitivity_sum",
            "path_count",
            "path_hard_count",
            "time_correct_sum",
            "time_uncond_sum",
            "time_hard_sum",
            "time_hard_win_sum",
            "time_sensitivity_sum",
            "time_count",
            "time_hard_count",
        )
        self._condition_val_stats = {
            f"{validation_t:g}": {
                name: torch.zeros((), device=self.device, dtype=torch.float64)
                for name in names
            }
            for validation_t in self.condition_val_ts
        }

    def _condition_validation_enabled(self, batch_idx: int) -> bool:
        """Use a fixed global condition-diagnostic budget under DDP."""
        trainer = getattr(self, "_trainer", None)
        if trainer is not None and trainer.sanity_checking:
            return False
        if self.condition_val_max_batches == 0:
            return True

        world_size = 1
        global_rank = 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            global_rank = torch.distributed.get_rank()

        base_batches, remainder = divmod(
            self.condition_val_max_batches, world_size
        )
        local_max_batches = base_batches + int(global_rank < remainder)
        return batch_idx < local_max_batches

    def _accumulate_condition_stats(
        self,
        validation_t: float,
        direction: str,
        correct_nll: Tensor,
        unconditional_nll: Tensor,
        hard_gain: Tensor,
        sensitivity: Tensor,
    ) -> None:
        stats = self._condition_val_stats[f"{validation_t:g}"]
        stats[f"{direction}_correct_sum"] += correct_nll.double().sum()
        stats[f"{direction}_uncond_sum"] += (
            unconditional_nll - correct_nll
        ).double().sum()
        stats[f"{direction}_count"] += correct_nll.numel()
        stats[f"{direction}_hard_sum"] += hard_gain.double().sum()
        stats[f"{direction}_hard_win_sum"] += (hard_gain > 0).double().sum()
        stats[f"{direction}_sensitivity_sum"] += sensitivity.double().sum()
        stats[f"{direction}_hard_count"] += hard_gain.numel()

    def on_validation_epoch_end(self) -> None:
        if not hasattr(self, "_condition_val_stats"):
            return
        metrics = {}
        for validation_t in self.condition_val_ts:
            t_key = f"{validation_t:g}"
            names = list(self._condition_val_stats[t_key])
            values = torch.stack(
                [self._condition_val_stats[t_key][name] for name in names]
            )
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(
                    values, op=torch.distributed.ReduceOp.SUM
                )
            stats = dict(zip(names, values))
            for direction in ("path", "time"):
                count = stats[f"{direction}_count"]
                hard_count = stats[f"{direction}_hard_count"]
                if count.item() == 0:
                    continue
                prefix = f"val/{direction}_cond_t{t_key}"
                direction_metrics = {
                    "nll_correct": (
                        stats[f"{direction}_correct_sum"] / count
                    ).float(),
                    "uncond_gain": (
                        stats[f"{direction}_uncond_sum"] / count
                    ).float(),
                    "sample_count": count.float(),
                    "hard_count": hard_count.float(),
                    "hard_coverage": (hard_count / count).float(),
                }
                if hard_count.item() > 0:
                    direction_metrics.update(
                        {
                            "hard_gain": (
                                stats[f"{direction}_hard_sum"] / hard_count
                            ).float(),
                            "hard_win": (
                                stats[f"{direction}_hard_win_sum"] / hard_count
                            ).float(),
                            "hard_sensitivity": (
                                stats[f"{direction}_sensitivity_sum"] / hard_count
                            ).float(),
                        }
                    )
                metrics.update(
                    {f"{prefix}_{suffix}": value for suffix, value in direction_metrics.items()}
                )
                if math.isclose(validation_t, self.condition_val_t):
                    legacy_prefix = f"val/{direction}_cond"
                    metrics.update(
                        {
                            f"{legacy_prefix}_{suffix}": value
                            for suffix, value in direction_metrics.items()
                        }
                    )
        if metrics:
            # Values have already been globally summed above. Every rank now
            # owns the same metrics, so Lightning's distributed mean leaves
            # them unchanged while recording them in a DDP-safe way.
            self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)

    @torch.no_grad()
    def _log_path_condition_quality(
        self,
        path: Tensor,
        path_emb: Tensor,
        time_emb: Tensor,
        hard_time_emb: Tensor,
        hard_eligible: Tensor,
        masks: Tensor,
        noise: Tensor,
        batch_idx: int,
    ) -> None:
        """Evaluate X|OD at fixed t values with shared validation noise."""
        if not self._condition_validation_enabled(batch_idx):
            return
        batch_size = path.size(0)
        correct_condition = self._path_condition(time_emb, path_emb, masks)
        unconditional_condition = self._unconditional_path_condition(
            path_emb, masks
        )
        # Use a non-self OD permutation with identical target, t, and noise.
        if batch_size > 1:
            permutation = torch.roll(
                torch.arange(batch_size, device=path.device), shifts=1
            )
            valid = masks.unsqueeze(-1).to(dtype=path_emb.dtype)
            wrong_od = self._od_context(
                path_emb[permutation], masks[permutation]
            ).unsqueeze(1)
            hard_condition = wrong_od.expand(-1, path.size(1), -1) * valid
            eligible = torch.ones(
                batch_size, dtype=torch.bool, device=path.device
            )
        else:
            eligible = torch.zeros(
                batch_size, dtype=torch.bool, device=path.device
            )
            hard_condition = None

        for validation_t in self.condition_val_ts:
            t = torch.full(
                (batch_size,), validation_t, device=path.device, dtype=path_emb.dtype
            )
            t_expanded = t.view(-1, *([1] * (path_emb.ndim - 1)))
            noisy_path = t_expanded * path_emb + (1 - t_expanded) * noise
            correct_prediction = self.path_net(
                noisy_path, t, cond=correct_condition
            )
            unconditional_prediction = self.path_net(
                noisy_path, t, cond=unconditional_condition
            )
            correct_nll = self._path_nll_per_sample(
                correct_prediction, path, masks
            )
            unconditional_nll = self._path_nll_per_sample(
                unconditional_prediction, path, masks
            )

            if eligible.any():
                hard_prediction = self.path_net(
                    noisy_path[eligible],
                    t[eligible],
                    cond=hard_condition[eligible],
                )
                hard_nll = self._path_nll_per_sample(
                    hard_prediction, path[eligible], masks[eligible]
                )
                hard_gain = hard_nll - correct_nll[eligible]
                difference = (
                    hard_prediction - correct_prediction[eligible]
                ).square()
                sensitivity = (
                    (difference * masks[eligible].unsqueeze(-1)).sum(dim=(1, 2))
                    / (
                        masks[eligible].sum(dim=1) * difference.size(-1)
                    ).clamp_min(1)
                ).sqrt()
            else:
                hard_gain = correct_nll.new_empty(0)
                sensitivity = correct_nll.new_empty(0)
            self._accumulate_condition_stats(
                validation_t,
                "path",
                correct_nll,
                unconditional_nll,
                hard_gain,
                sensitivity,
            )

    @torch.no_grad()
    def _log_time_condition_quality(
        self,
        time: Tensor,
        time_emb: Tensor,
        path_emb: Tensor,
        hard_path_emb: Tensor,
        hard_eligible: Tensor,
        masks: Tensor,
        noise: Tensor,
        batch_idx: int,
    ) -> None:
        """Evaluate T|X at fixed t values with shared validation noise."""
        if not self._condition_validation_enabled(batch_idx):
            return
        batch_size = time.size(0)
        unconditional_condition = self.uncond_path_emb.view(
            1, 1, -1
        ).expand_as(path_emb)
        unconditional_condition = (
            unconditional_condition * masks.unsqueeze(-1)
        )
        eligible = hard_eligible.bool()

        for validation_t in self.condition_val_ts:
            t = torch.full(
                (batch_size,), validation_t, device=time.device, dtype=time_emb.dtype
            )
            t_expanded = t.view(-1, *([1] * (time_emb.ndim - 1)))
            noisy_time = t_expanded * time_emb + (1 - t_expanded) * noise
            correct_prediction = self.time_net(
                noisy_time, t, cond=path_emb
            )
            unconditional_prediction = self.time_net(
                noisy_time, t, cond=unconditional_condition
            )
            correct_nll = self._time_nll_per_sample(
                correct_prediction, time, masks
            )
            unconditional_nll = self._time_nll_per_sample(
                unconditional_prediction, time, masks
            )

            if eligible.any():
                hard_prediction = self.time_net(
                    noisy_time[eligible],
                    t[eligible],
                    cond=hard_path_emb[eligible],
                )
                hard_nll = self._time_nll_per_sample(
                    hard_prediction, time[eligible], masks[eligible]
                )
                hard_gain = hard_nll - correct_nll[eligible]
                difference = (
                    hard_prediction - correct_prediction[eligible]
                ).square()
                sensitivity = (
                    (difference * masks[eligible].unsqueeze(-1)).sum(dim=(1, 2))
                    / (
                        masks[eligible].sum(dim=1) * difference.size(-1)
                    ).clamp_min(1)
                ).sqrt()
            else:
                hard_gain = correct_nll.new_empty(0)
                sensitivity = correct_nll.new_empty(0)
            self._accumulate_condition_stats(
                validation_t,
                "time",
                correct_nll,
                unconditional_nll,
                hard_gain,
                sensitivity,
            )

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        path, raw_time, masks = batch[0]
        condition_batch = batch[2]
        time = self._transform_time(raw_time, masks)
        hard_time = self._transform_time(condition_batch["hard_time"], masks)
        batch_size = path.size(0)

        path_emb = self.path_embedding(path)
        time_emb = self.time_embedding(time)
        hard_time_emb = self.time_embedding(hard_time)
        hard_path_emb = self.path_embedding(condition_batch["hard_path"])
        path_emb = F.normalize(path_emb, p=2, dim=-1) * math.sqrt(self.emb_dim)
        time_emb = F.normalize(time_emb, p=2, dim=-1) * math.sqrt(self.emb_dim)
        hard_time_emb = F.normalize(
            hard_time_emb, p=2, dim=-1
        ) * math.sqrt(self.emb_dim)
        hard_path_emb = F.normalize(
            hard_path_emb, p=2, dim=-1
        ) * math.sqrt(self.emb_dim)
        path_condition = self._path_condition(time_emb, path_emb, masks)

        t_random = self._sample_t(batch_size, path_emb.device).view(
            -1, *([1] * (path_emb.ndim - 1))
        )

        r_noise = torch.randn_like(path_emb)
        r_t = t_random * path_emb + (1 - t_random) * r_noise
        v_r = (path_emb - r_t) / (1 - t_random).clamp_min(self.t_eps)

        r_pred = self.path_net(r_t, t_random.flatten(), cond=path_condition)
        v_r_pred = (r_pred - r_t) / (1 - t_random).clamp_min(self.t_eps)

        val_path_v_loss = self.masked_mse_loss(v_r_pred, v_r, masks)

        path_logits = self._compute_tied_logits(
            r_pred, self.path_embedding, self.path_temperature
        )
        val_path_ce_loss = self.ce_loss(
            path_logits.view(-1, path_logits.size(-1)), path.view(-1).long()
        )
        val_path_entropy = self._compute_logit_entropy(path_logits, masks)

        t_time_random = self._sample_t(batch_size, time_emb.device).view(
            -1, *([1] * (time_emb.ndim - 1))
        )
        tau_noise = torch.randn_like(time_emb)
        tau_t = t_time_random * time_emb + (1 - t_time_random) * tau_noise
        v_tau = (time_emb - tau_t) / (1 - t_time_random).clamp_min(self.t_eps)

        tau_pred = self.time_net(
            tau_t, t_time_random.flatten(), cond=path_emb
        )
        v_tau_pred = (tau_pred - tau_t) / (
            1 - t_time_random
        ).clamp_min(self.t_eps)

        val_time_v_loss = self.masked_mse_loss(v_tau_pred, v_tau, masks)

        val_time_ce_loss = 0.0
        val_time_entropy = 0.0

        for i, emb_layer in enumerate(self.time_embedding_weights):
            val_time_ce_loss += self._compute_ce_loss(
                tau_pred, time[:, :, i], emb_layer, self.time_temperature
            )
            time_logits_i = self._compute_tied_logits(
                tau_pred, emb_layer, self.time_temperature
            )
            val_time_entropy += self._compute_logit_entropy(time_logits_i, masks)

        val_time_entropy /= len(self.time_embedding_weights)

        # Match training_step exactly: average the four time CEs and apply
        # the same 0.2 scale to the time vector-field loss.
        val_time_ce_loss = val_time_ce_loss / len(self.time_embedding_weights)
        val_time_v_loss = val_time_v_loss * 0.2
        total_val_loss = (
            val_path_v_loss
            + val_time_v_loss
            + self.lambda_ce * (val_path_ce_loss + val_time_ce_loss)
        )

        # Match the one-step anomaly detector: evaluate at 90% noise (t=0.1).
        t_fixed = torch.full((batch_size,), 0.1, device=path_emb.device)
        t_fixed_exp = t_fixed.view(-1, *([1] * (path_emb.ndim - 1)))

        r_t_fixed = t_fixed_exp * path_emb + (1 - t_fixed_exp) * r_noise
        r_pred_fixed = self.path_net(r_t_fixed, t_fixed, cond=path_condition)

        fixed_logits = self._compute_tied_logits(
            r_pred_fixed, self.path_embedding, self.path_temperature
        )

        preds = fixed_logits.argmax(dim=-1)
        correct = (preds == path) * masks
        path_accuracy = correct.sum() / masks.sum().clamp_min(1.0)
        path_entropy_t0_1 = self._compute_logit_entropy(fixed_logits, masks)

        self._log_path_condition_quality(
            path,
            path_emb,
            time_emb,
            hard_time_emb,
            condition_batch["hard_time_eligible"],
            masks,
            r_noise,
            batch_idx,
        )
        self._log_time_condition_quality(
            time,
            time_emb,
            path_emb,
            hard_path_emb,
            condition_batch["hard_path_eligible"],
                masks,
                tau_noise,
                batch_idx,
            )

        self.log_dict(
            {
                "val/loss": total_val_loss,
                "val/path_v_loss": val_path_v_loss,
                "val/path_ce_loss": val_path_ce_loss,
                "val/time_v_loss": val_time_v_loss,
                "val/time_ce_loss": val_time_ce_loss,
                "val/path_entropy": val_path_entropy,
                "val/time_entropy": val_time_entropy,
                "val/path_acc_t0.1": path_accuracy,
                "val/path_entropy_t0.1": path_entropy_t0_1,
                "val/path_temperature": self.path_temperature,
                "val/time_temperature": self.time_temperature,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def training_step(self, batch, batch_idx):
        path, time, masks = batch[0]
        time = self._transform_time(time, masks)
        batch_size = path.size(0)

        path_emb = self.path_embedding(path)
        time_emb = self.time_embedding(time)

        # Match the noise scale while preserving embedding direction.
        path_emb = F.normalize(path_emb, p=2, dim=-1) * math.sqrt(self.emb_dim)
        time_emb = F.normalize(time_emb, p=2, dim=-1) * math.sqrt(self.emb_dim)

        dropped_path_emb = self._apply_classifier_free_dropout(
            path_emb, self.uncond_path_emb
        )
        correct_od = self._path_condition(time_emb, path_emb, masks)
        path_condition = self._apply_classifier_free_dropout(
            correct_od, self.uncond_od_emb
        )
        path_condition = path_condition * masks.unsqueeze(-1).to(
            dtype=path_condition.dtype
        )

        t_path = self._sample_t(batch_size, path_emb.device).view(
            -1, *([1] * (path_emb.ndim - 1))
        )

        r_noise = torch.randn_like(path_emb)
        r_t = t_path * path_emb + (1 - t_path) * r_noise
        v_r = (path_emb - r_t) / (1 - t_path).clamp_min(self.t_eps)

        r_pred = self.path_net(r_t, t_path.flatten(), cond=path_condition)
        v_r_pred = (r_pred - r_t) / (1 - t_path).clamp_min(self.t_eps)

        path_v_loss = self.masked_mse_loss(v_r_pred, v_r, masks)

        path_ce_loss = self._compute_ce_loss(
            r_pred, path, self.path_embedding, self.path_temperature
        )

        t_time = self._sample_t(batch_size, time_emb.device).view(
            -1, *([1] * (time_emb.ndim - 1))
        )

        tau_noise = torch.randn_like(time_emb)
        tau_t = t_time * time_emb + (1 - t_time) * tau_noise
        v_tau = (time_emb - tau_t) / (1 - t_time).clamp_min(self.t_eps)

        tau_pred = self.time_net(tau_t, t_time.flatten(), cond=dropped_path_emb)
        v_tau_pred = (tau_pred - tau_t) / (1 - t_time).clamp_min(self.t_eps)

        time_v_loss = self.masked_mse_loss(v_tau_pred, v_tau, masks)

        time_ce_loss = 0.0

        for i, emb_layer in enumerate(self.time_embedding_weights):
            time_ce_loss += self._compute_ce_loss(
                tau_pred, time[:, :, i], emb_layer, self.time_temperature
            )

        time_ce_loss = time_ce_loss / len(self.time_embedding_weights)
        time_v_loss = time_v_loss * 0.2
        base_loss = (
            path_v_loss
            + time_v_loss
            + self.lambda_ce * (path_ce_loss + time_ce_loss)
        )
        loss = base_loss

        self.log_dict(
            {
                "total_loss": loss,
                "base_loss": base_loss,
                "path_v_loss": path_v_loss,
                "time_v_loss": time_v_loss,
                "path_ce_loss": path_ce_loss * self.lambda_ce,
                "time_ce_loss": time_ce_loss * self.lambda_ce,
                "path_temperature": self.path_temperature,
                "time_temperature": self.time_temperature,
            },
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        return loss

    @torch.no_grad()
    def _compute_fast_anomaly_score(
        self,
        tgt_path,
        tgt_time_trans,
        tgt_path_emb,
        tgt_time_emb,
        masks,
        batch_size,
        fixed_time=0.4,
    ):
        """
        Adds fixed noise at t=fixed_time and calculates the reconstruction NLL.
        """
        t = torch.full((batch_size,), fixed_time, device=tgt_path_emb.device)
        t_exp = t.view(-1, *([1] * (tgt_path_emb.ndim - 1)))

        t_next = torch.full((batch_size,), 1.0, device=tgt_path_emb.device)
        t_next_exp = t_next.view(-1, *([1] * (tgt_path_emb.ndim - 1)))

        r_noise = torch.randn_like(tgt_path_emb)
        tau_noise = torch.randn_like(tgt_time_emb)

        r_t = t_exp * tgt_path_emb + (1 - t_exp) * r_noise
        tau_t = t_exp * tgt_time_emb + (1 - t_exp) * tau_noise

        path_condition = self._path_condition(
            tgt_time_emb, tgt_path_emb, masks
        )
        path_unconditional = self._unconditional_path_condition(
            tgt_path_emb, masks
        )
        path_pred_latents = self._euler_step(
            z=r_t,
            t=t_exp,
            t_next=t_next_exp,
            cond=path_condition,
            uncond_emb=path_unconditional,
            net=self.path_net,
        )
        time_pred_latents = self._euler_step(
            z=tau_t,
            t=t_exp,
            t_next=t_next_exp,
            cond=tgt_path_emb,
            uncond_emb=self.uncond_path_emb,
            net=self.time_net,
        )

        path_score, time_score = self._calculate_nll_scores(
            path_pred_latents, time_pred_latents, tgt_path, tgt_time_trans, masks
        )
        return path_score, time_score

    @torch.no_grad()
    def _calculate_nll_scores(
        self, path_latents, time_latents, tgt_path, tgt_time_trans, masks
    ):
        """
        Shared logic for Weight Tying NLL calculation.
        """
        valid_lengths = masks.sum(dim=1).clamp_min(1.0)

        path_logits = self._compute_tied_logits(
            path_latents, self.path_embedding, self.path_temperature
        )
        path_log_probs = F.log_softmax(path_logits, dim=-1)
        target_path_log_probs = torch.gather(
            path_log_probs, dim=-1, index=tgt_path.unsqueeze(-1).long()
        ).squeeze(-1)
        path_score = -((target_path_log_probs * masks).sum(dim=1) / valid_lengths)

        total_time_log_probs = torch.zeros_like(target_path_log_probs)

        for i, emb_layer in enumerate(self.time_embedding_weights):
            time_logits_i = self._compute_tied_logits(
                time_latents, emb_layer, self.time_temperature
            )
            time_log_probs_i = F.log_softmax(time_logits_i, dim=-1)

            target_feature = tgt_time_trans[:, :, i].long()
            total_time_log_probs += torch.gather(
                time_log_probs_i, dim=-1, index=target_feature.unsqueeze(-1)
            ).squeeze(-1)

        time_score = -(
            ((total_time_log_probs * masks).sum(dim=1) / valid_lengths) / 4.0
        )

        return path_score, time_score

    def test_step(self, batch, batch_idx):
        (tgt_path, tgt_time, masks), labels = batch
        batch_size = labels.size(0)
        tgt_time_trans = self._transform_time(tgt_time, masks)

        tgt_path_emb = self.path_embedding(tgt_path)
        tgt_time_emb = self.time_embedding(tgt_time_trans)

        tgt_path_emb = F.normalize(tgt_path_emb, p=2, dim=-1) * math.sqrt(self.emb_dim)
        tgt_time_emb = F.normalize(tgt_time_emb, p=2, dim=-1) * math.sqrt(self.emb_dim)

        path_score, time_score = self._compute_fast_anomaly_score(
            tgt_path,
            tgt_time_trans,
            tgt_path_emb,
            tgt_time_emb,
            masks,
            batch_size,
            fixed_time=self.test_t,
        )

        anomaly_score = path_score + time_score

        self._test_path_scores.append(path_score.detach().float().cpu())
        self._test_time_scores.append(time_score.detach().float().cpu())
        self._test_labels.append(labels.detach().long().cpu())

        self.auroc.update(anomaly_score, labels)
        self.ap.update(anomaly_score, labels)

        self.log_dict(
            {
                "path_score": path_score.mean(),
                "time_score": time_score.mean(),
                "score": anomaly_score.mean(),
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
            sync_dist=True,
        )

    def on_test_epoch_start(self):
        self._test_path_scores = []
        self._test_labels = []
        self._test_time_scores = []

    def on_test_epoch_end(self):
        # Log metrics directly to avoid torchmetrics typing stub issues on compute().
        self.log("test_auroc", self.auroc, sync_dist=True)
        self.log("test_ap", self.ap, sync_dist=True)

        if not self.score_cache_path:
            return

        if not self._test_path_scores:
            path_scores = torch.tensor([], dtype=torch.float32)
            time_scores = torch.tensor([], dtype=torch.float32)
            labels = torch.tensor([], dtype=torch.long)
        else:
            path_scores = torch.cat(self._test_path_scores)
            time_scores = torch.cat(self._test_time_scores)
            labels = torch.cat(self._test_labels)

        cache_path = self.score_cache_path

        wandb_run_id = (
            getattr(self.logger, "version", None) if self.global_rank == 0 else ""
        )
        if hasattr(self.trainer.strategy, "broadcast"):
            wandb_run_id = self.trainer.strategy.broadcast(wandb_run_id, src=0)

        if wandb_run_id and isinstance(wandb_run_id, str):
            cache_path = os.path.join(
                os.path.dirname(cache_path),
                wandb_run_id,
                os.path.basename(cache_path),
            )

        cache_root, cache_ext = os.path.splitext(cache_path)
        if not cache_ext:
            cache_ext = ".pt"

        cache_dir = os.path.dirname(cache_root)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        world_size = int(getattr(self.trainer, "world_size", 1))
        rank = int(getattr(self, "global_rank", 0))

        if world_size > 1:
            rank_cache_path = f"{cache_root}.rank{rank}{cache_ext}"
            torch.save(
                {
                    "path_scores": path_scores,
                    "time_scores": time_scores,
                    "labels": labels,
                    "rank": rank,
                    "world_size": world_size,
                },
                rank_cache_path,
            )
            self.trainer.strategy.barrier()

            if rank == 0:
                merged_path_scores = []
                merged_time_scores = []
                merged_labels = []
                for r in range(world_size):
                    rank_path = f"{cache_root}.rank{r}{cache_ext}"
                    rank_data = torch.load(rank_path, map_location="cpu")
                    merged_path_scores.append(rank_data["path_scores"])
                    merged_time_scores.append(rank_data["time_scores"])
                    merged_labels.append(rank_data["labels"])

                final_cache_path = f"{cache_root}{cache_ext}"
                torch.save(
                    {
                        "path_scores": torch.cat(merged_path_scores),
                        "time_scores": torch.cat(merged_time_scores),
                        "labels": torch.cat(merged_labels),
                        "world_size": world_size,
                    },
                    final_cache_path,
                )
                logging.info("Saved merged test scores to %s", final_cache_path)
        else:
            final_cache_path = f"{cache_root}{cache_ext}"
            torch.save(
                {
                    "path_scores": path_scores,
                    "time_scores": time_scores,
                    "labels": labels,
                    "world_size": 1,
                },
                final_cache_path,
            )
            logging.info("Saved test scores to %s", final_cache_path)

    def configure_optimizers(self):
        scaled_lr = self.learning_rate * math.sqrt(self.trainer.accumulate_grad_batches)
        optimizer = optim.AdamW(
            self.parameters(),
            lr=scaled_lr,
            fused=True,
        )

        total_steps = self.trainer.estimated_stepping_batches
        if total_steps == float("inf"):
            total_steps = 1000000  # Fallback if max_epochs is not set
        else:
            total_steps = int(total_steps)

        if self.warmup_steps <= 0:
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

        warmup_scheduler = lr_scheduler.LinearLR(
            optimizer, start_factor=1e-8, end_factor=1.0, total_iters=self.warmup_steps
        )
        cosine_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_steps - self.warmup_steps)
        )

        scheduler = lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_steps],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def train_flow(
    args,
    best_model_path: str,
    trainer: Trainer,
    timestamp_converter: TimestampConverter,
    dit_config: dict,
    last_ckpt_path: str,
):
    # Build the complete vocabulary before sizing the model embeddings.
    token_data_module = TokenTrajectoryDataModule(
        location=args.location,
        timestamp_converter=timestamp_converter,
        random_seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        test_anomaly_type=args.test_anomaly_type,
        test_anomaly_types=args.test_anomaly_types,
        test_anomaly_ratio=args.test_anomaly_ratio,
        switch_relax=args.switch_relax,
        shift_time_gap=args.shift_time_gap,
        edge_mapping=None,
        mllm_subset_path=args.mllm_subset_path,
    )
    token_data_module.prepare_edge_mapping()
    num_edges = token_data_module.num_edges

    if args.status == "train":
        logging.info("--- Training Conditional Flow Matching ---")

        flow_model = ConditionalFlowMatching(
            num_edges=num_edges,
            emb_dim=args.emb_dim,
            P_mean=args.P_mean,
            P_std=args.P_std,
            cfg_interval=args.cfg_interval,
            t_eps=args.t_eps,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            cfg_drop_prob=args.cfg_drop_prob,
            cfg_warmup_epochs=args.cfg_warmup_epochs,
            guidance_scale=args.guidance_scale,
            dit_config=dit_config,
            score_cache_path=args.score_cache_path,
            test_t=args.test_t,
            lambda_ce=args.lambda_ce,
            condition_val_t=args.condition_val_t,
            condition_val_ts=args.condition_val_ts,
            condition_val_max_batches=args.condition_val_max_batches,
            t_sampling=args.t_sampling,
        )

        token_data_module.setup(stage="fit")

        # Resume from the latest training state if available.
        ckpt_path = (
            last_ckpt_path
            if os.path.exists(last_ckpt_path)
            else (best_model_path if os.path.exists(best_model_path) else None)
        )

        trainer.fit(
            flow_model,
            train_dataloaders=token_data_module.train_dataloader(),
            val_dataloaders=token_data_module.val_dataloader(),
            ckpt_path=ckpt_path,
        )

    elif args.status == "test":
        logging.info("--- Testing Conditional Flow Matching ---")

        if best_model_path is None or not os.path.exists(best_model_path):
            raise FileNotFoundError(
                f"Flow model checkpoint not found at {best_model_path}. Please run 'train_flow' stage first."
            )

        logging.info("Loading Flow model from %s...", best_model_path)
        flow_model = ConditionalFlowMatching.load_from_checkpoint(
            checkpoint_path=best_model_path,
            map_location="cpu",
            weights_only=False,
            guidance_scale=args.guidance_scale,
            score_cache_path=args.score_cache_path,
            test_t=args.test_t,
        )
        test_stage = "mllm_subset" if args.mllm_subset_path else "test"
        token_data_module.setup(stage=test_stage)
        trainer.test(flow_model, dataloaders=token_data_module.test_dataloader())


class MetricsHistoryCallback(L.Callback):
    def __init__(self):
        self.history = []

    def on_validation_epoch_end(self, trainer, pl_module):
        m = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in trainer.callback_metrics.items()
        }
        m["epoch"] = trainer.current_epoch
        self.history.append(m)


def main(args):
    """Train or evaluate CompatTrajFlow."""
    logging.getLogger("lightning.pytorch").setLevel(logging.INFO)
    L.seed_everything(args.seed, workers=True)

    timestamp_converter = TimestampConverter(args.location)

    architecture_tag = "odcfg_path_x2time"
    name = args.run_name or (
        f"{args.location}_weight_tying_{architecture_tag}_{args.t_sampling}"
    )

    logger = False
    if args.wandb_mode != "disabled":
        logger = WandbLogger(
            project=args.wandb_project,
            name=name,
            config=vars(args),
            save_code=True,
            offline=args.wandb_mode == "offline",
        )

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        every_n_epochs=1,
        filename=(name),
        save_on_train_epoch_end=True,
    )

    last_checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename=f"{name}_last",
        monitor=None,
        save_top_k=1,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
    )

    best_model_path = args.checkpoint_path or os.path.join(
        args.checkpoint_dir, f"{name}.ckpt"
    )
    last_ckpt_path = os.path.join(args.checkpoint_dir, f"{name}_last.ckpt")

    ema_callback = WeightAveraging(multi_avg_fn=get_ema_multi_avg_fn(args.ema_decay))
    rich_progress = RichProgressBar(leave=True)
    rich_summary = RichModelSummary(max_depth=-1)
    history_cb = MetricsHistoryCallback()

    callbacks = [
        ema_callback,
        rich_progress,
        rich_summary,
        history_cb,
    ]
    if args.status == "train":
        # Select checkpoints only by clean reconstruction validation loss.
        # Hard-condition diagnostics are logged but never used for model selection.
        callbacks.extend(
            [
                checkpoint_callback,
                last_checkpoint_callback,
            ]
        )

    trainer = Trainer(
        max_epochs=args.epochs,
        logger=logger,
        accelerator=args.accelerator,
        devices=args.gpus,
        strategy=args.strategy,
        enable_progress_bar=True,
        log_every_n_steps=10,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        limit_val_batches=args.limit_val_batches,
        callbacks=callbacks,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        profiler=args.profiler,
    )
    dit_config = dict(
        depth=args.depth,
        emb_dim=args.emb_dim,
        hidden_size=args.hidden_dim,
        num_heads=args.nheads,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    train_flow(
        args,
        best_model_path,
        trainer,
        timestamp_converter,
        dit_config,
        last_ckpt_path,
    )

    if args.metrics_out:
        metrics = {}
        for k, v in trainer.callback_metrics.items():
            metrics[k] = v.item() if isinstance(v, torch.Tensor) else v

        output_data = {"final": metrics, "history": history_cb.history}

        os.makedirs(os.path.dirname(os.path.abspath(args.metrics_out)), exist_ok=True)
        with open(args.metrics_out, "w") as f:
            json.dump(output_data, f, indent=4)
        logging.info(f"Saved metrics to {args.metrics_out}")


def sys_settings():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_DISABLE_P2P"] = "1"
    os.environ["NCCL_P2P_DISABLE"] = "1"

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Stable run/checkpoint name",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory for training checkpoints",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Explicit checkpoint to load in test mode",
    )
    parser.add_argument(
        "--wandb_mode",
        choices=["disabled", "offline", "online"],
        default="disabled",
        help="Weights & Biases logging mode",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="compattrajflow",
        help="Weights & Biases project name when logging is enabled",
    )
    parser.add_argument(
        "--location",
        type=str,
        default="porto",
        choices=["xian", "porto"],
        help="Location for the dataset",
    )
    parser.add_argument(
        "--status",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Run mode",
    )
    parser.add_argument(
        "--test_anomaly_type",
        type=str,
        default="detour",
        choices=["detour", "switch", "time_shift"],
    )
    parser.add_argument(
        "--test_anomaly_ratio",
        type=float,
        default=0.3,
        help="Ratio of anomalies in the test set",
    )
    parser.add_argument(
        "--switch_relax",
        type=int,
        default=3,
        help="Relaxation for switch anomaly extra random walk",
    )
    parser.add_argument(
        "--shift_time_gap",
        type=int,
        default=30,
        help="Time gap for time_shift anomaly (in seconds)",
    )
    parser.add_argument(
        "--lambda_ce",
        type=float,
        default=0.25,
        help="Weight for CE loss (0.0 for MSE only)",
    )
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument(
        "--accelerator", type=str, default="gpu", help="Accelerator type"
    )
    parser.add_argument(
        "--strategy", type=str, default="auto", help="Training strategy"
    )
    parser.add_argument(
        "--precision", type=str, default="bf16-mixed", help="Precision for training"
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing in DiT blocks to reduce peak GPU memory during training",
    )
    parser.add_argument(
        "--num_workers", type=int, default=16, help="Number of workers for data loading"
    )
    parser.add_argument(
        "--check_val_every_n_epoch",
        type=int,
        default=1,
        help="Frequency of validation checks in epochs",
    )
    parser.add_argument(
        "--limit_val_batches",
        type=float,
        default=1.0,
        help="Fraction of validation batches to run per validation check",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs to train"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Per-GPU batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=10000,
        help="Number of linear warmup steps for learning rate",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.9999,
        help="Decay rate for Exponential Moving Average (EMA)",
    )
    parser.add_argument(
        "--emb_dim", type=int, default=128, help="Embedding dimension for the model"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=256, help="Hidden dimension for DiT layers"
    )
    parser.add_argument("--depth", type=int, default=4, help="Number of DiT layers")
    parser.add_argument(
        "--nheads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument(
        "--t_sampling",
        type=str,
        choices=["uniform", "logit_normal"],
        default="logit_normal",
        help="Training/validation t distribution.",
    )
    parser.add_argument(
        "--P_mean",
        type=float,
        default=-0.8,
        help="Mean of the logit-normal t distribution (ignored for uniform sampling)",
    )
    parser.add_argument(
        "--P_std",
        type=float,
        default=0.8,
        help="Standard deviation of the logit-normal t distribution (ignored for uniform sampling)",
    )
    parser.add_argument(
        "--cfg_interval",
        type=float,
        nargs=2,
        default=(0.1, 1.0),
        metavar=("LOW", "HIGH"),
        help="Interval of t for applying classifier-free guidance",
    )
    parser.add_argument(
        "--t_eps",
        type=float,
        default=5e-2,
        help="Epsilon value for the preventing division by zero in time variable",
    )
    parser.add_argument(
        "--cfg_drop_prob",
        type=float,
        default=0.1,
        help="Drop probability for label dropout during training",
    )
    parser.add_argument(
        "--cfg_warmup_epochs",
        type=int,
        default=5,
        help="Number of epochs to train without CFG dropout for stability warmup",
    )
    parser.add_argument(
        "--test_t",
        type=float,
        default=0.1,
        help="Fixed target-mixture t used by anomaly detection testing",
    )
    parser.add_argument(
        "--condition_val_t",
        type=float,
        default=0.1,
        help="Fixed target-mixture t used for bidirectional condition validation",
    )
    parser.add_argument(
        "--condition_val_ts",
        type=float,
        nargs="+",
        default=None,
        help="Optional fixed t values for condition diagnostics.",
    )
    parser.add_argument(
        "--condition_val_max_batches",
        type=int,
        default=32,
        help="Maximum validation batches used for condition diagnostics (0 means all)",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="Guidance scale for classifier-free guidance",
    )
    parser.add_argument(
        "--score_cache_path",
        type=str,
        default="result/test_scores_labels.pt",
        help="Path to save test scores and labels for fast metric recalculation",
    )
    parser.add_argument(
        "--profiler",
        type=str,
        default=None,
        choices=[None, "simple", "advanced", "pytorch"],
        help="Profiler to use for identifying bottlenecks.",
    )
    parser.add_argument(
        "--metrics_out",
        type=str,
        default=None,
        help="Path to dump metrics.json",
    )
    parser.add_argument(
        "--mllm_subset_path",
        type=str,
        default=None,
        help="Path to directory containing MLLM-selected subset index JSON files.",
    )
    parser.add_argument(
        "--test_anomaly_types",
        type=str,
        nargs="+",
        default=["detour", "switch", "time_shift"],
        help="Test anomaly types (default: all three)",
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )

    sys_settings()  # Set system settings for the script
    logging.info("System settings configured.")
    parsed_args = parser.parse_args()
    if parsed_args.wandb_mode != "disabled":
        wandb.finish()
    main(parsed_args)
