"""Entry point for baseline training and evaluation."""

import gc
import logging
import os
import pickle
import sys
from argparse import Namespace

import numpy as np
import pandas as pd
import torch
from torch import Generator

from args import init_args
from preprocessing.anomaly import AnomalyGenerator
from preprocessing.preprocessing import Preprocessing
from utils.edge_remapper import (
    apply_edge_mapping,
    get_or_build_edge_mapping,
    get_or_build_grid_mapping,
)
from utils.road_network import RoadNetwork
from utils.timestamp_converter import TimestampConverter
from utils.tools import (
    get_test_runner_dataloader_pairs,
    get_train_runner_dataloader_pairs,
    set_device,
)

logger = logging.getLogger(__name__)


def main(args: Namespace):
    """Run preprocessing, anomaly generation, and the selected baselines."""
    gpu_rng = Generator(args.device).manual_seed(args.random_seed)
    rng = np.random.default_rng(args.random_seed)

    road_network = RoadNetwork(args.location)
    preprocessing = Preprocessing(
        args.location,
        args.trajectory_len_threshold,
        road_network,
        tuple(args.data_split_ratio),
        gpu_rng,
        rng,
    )

    train_trajs, val_trajs, test_trajs = preprocessing()

    timestamp_converter = TimestampConverter(args.location)
    args.num_times = timestamp_converter.num_time_tokens
    logger.info("Number of time tokens: %d", args.num_times)

    anomaly_generator = AnomalyGenerator(
        location=args.location,
        anomaly_ratio=args.anomaly_ratio,
        test_trajectories=test_trajs,
        rng=rng,
        random_seed=args.random_seed,
        switch_relax=args.switch_relax,
        shift_time_gap=args.shift_time_gap,
    )

    anomaly_trajectories_dict = {}
    for anomaly_type in args.anomaly_types:
        for anomaly_proportion in args.anomaly_proportions:
            logger.info(
                "Generating anomaly type: %s with proportion: %.2f",
                anomaly_type,
                anomaly_proportion,
            )
            anomaly_trajectories = anomaly_generator(
                anomaly_type, anomaly_proportion, 16, 2000
            )
            for trajectory in anomaly_trajectories:
                trajectory.set_observation_ratio(args.observation_ratio)
            anomaly_trajectories_dict[f"{anomaly_type}_{anomaly_proportion}"] = (
                anomaly_trajectories
            )

    # Build the vocabulary after anomaly generation so grid boundaries include
    # every trajectory that will be tokenized in this run.
    all_trajs_for_mapping = train_trajs + val_trajs + test_trajs
    for anomalies in anomaly_trajectories_dict.values():
        all_trajs_for_mapping.extend(anomalies)

    if args.use_grid_tokens:
        edge_mapping, num_grid_tokens, grid_shape = get_or_build_grid_mapping(
            args.location,
            args.grid_height_km,
            args.grid_width_km,
            all_trajs_for_mapping,
            return_num_tokens=True,
        )
        args.num_edges = num_grid_tokens
        args.grid_num_rows, args.grid_num_cols = grid_shape
    else:
        edge_mapping = get_or_build_edge_mapping(args.location)
        args.num_edges = len(set(edge_mapping.values()))

    del all_trajs_for_mapping

    logger.info("Number of mapped edge/grid tokens: %d", args.num_edges)

    logger.info("Applying edge remapping to trajectories...")
    logger.info("Mapping train trajectories...")
    apply_edge_mapping(train_trajs, edge_mapping)
    logger.info("Mapping validation trajectories...")
    apply_edge_mapping(val_trajs, edge_mapping)
    logger.info("Mapping test trajectories...")
    apply_edge_mapping(test_trajs, edge_mapping)
    for anomaly_type, anomalies in anomaly_trajectories_dict.items():
        logger.info("Mapping %s anomaly trajectories...", anomaly_type)
        apply_edge_mapping(anomalies, edge_mapping)

    # Grid and edge vocabularies use separate files to prevent accidental reuse.
    if args.use_grid_tokens:
        mapping_path = (
            f"data/{args.location}/processed/"
            f"grid_mapping_{args.grid_height_km}_{args.grid_width_km}_expanded.pkl"
        )
    else:
        mapping_path = f"data/{args.location}/processed/edge_mapping.pkl"
    with open(mapping_path, "wb") as f:
        pickle.dump(edge_mapping, f)
    logger.info("Saved expanded mapping to %s", mapping_path)

    if not args.use_grid_tokens:
        args.num_edges = len(set(edge_mapping.values()))
    args.edge_mapping = edge_mapping
    logger.info(
        "Final number of mapped edge/grid tokens including anomalies: %d",
        args.num_edges,
    )

    if not args.train:
        # Mapping is complete, so test-only runs can release train/validation data.
        logger.info(
            "Test-only mode: freeing train (%d) and val (%d) trajectories from memory",
            len(train_trajs),
            len(val_trajs),
        )
        del train_trajs
        del val_trajs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for trajectory in test_trajs:
            trajectory.set_observation_ratio(args.observation_ratio)

        for anomalies in anomaly_trajectories_dict.values():
            for trajectory in anomalies:
                trajectory.set_observation_ratio(args.observation_ratio)

    if args.train:
        train_runner_dataloaders_pairs = get_train_runner_dataloader_pairs(
            train_trajs,
            val_trajs,
            test_trajs,
            anomaly_trajectories_dict,
            timestamp_converter,
            gpu_rng,
            args,
        )
        col_dict = {}
        for (
            runner,
            train_dataloader,
            val_dataloader,
            test_dataloader_dict,
        ) in train_runner_dataloaders_pairs:
            train_runner = runner(args, gpu_rng)
            result = train_runner(
                "train",
                train_dataloader,
                val_dataloader,
                test_dataloader_dict,
            )
            if result:
                col_dict[train_runner.name] = result
        for key, value in col_dict.items():
            result = pd.DataFrame(value)
            logger.info("\n%s", result.to_string(index=False))
            os.makedirs(f"result/{args.location}", exist_ok=True)
            result.to_csv(
                f"result/{args.location}/{key}.csv",
                index=False,
                float_format="%.4f",
            )
        del train_runner_dataloaders_pairs

    else:
        test_runner_dataloaders_pairs = get_test_runner_dataloader_pairs(
            test_trajs,
            anomaly_trajectories_dict,
            timestamp_converter,
            gpu_rng,
            args,
        )
        col_dict = {}
        for runner, test_dataloader_dict in test_runner_dataloaders_pairs:
            test_runner = runner(args, gpu_rng)
            test_result = test_runner(
                "test",
                None,
                None,
                test_dataloader_dict,
            )
            if test_result:
                col_dict[test_runner.name] = test_result
        for key, value in col_dict.items():
            result = pd.DataFrame(value)
            logger.info("\n%s", result.to_string(index=False))
            os.makedirs(f"result/{args.location}", exist_ok=True)
            result.to_csv(
                f"result/{args.location}/{key}.csv",
                index=False,
                float_format="%.4f",
            )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    all_args = init_args()
    run_device = set_device(args=all_args)
    all_args.device = run_device
    main(all_args)
