#!/usr/bin/env python3
"""Build paper data splits and synthetic validation/test anomalies."""

from __future__ import annotations

import argparse
import logging

import numpy as np
import torch

from preprocessing.anomaly import AnomalyGenerator
from preprocessing.preprocessing import Preprocessing
from utils.road_network import RoadNetwork


PAPER_ANOMALIES = ("detour", "switch", "time_shift")
PAPER_PROPORTIONS = (0.1, 0.3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess Porto/Xi'an and generate the paper anomaly sets."
    )
    parser.add_argument("--location", required=True, choices=("porto", "xian"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--min_length", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=300)
    parser.add_argument("--anomaly_ratio", type=float, default=0.05)
    parser.add_argument("--proportions", type=float, nargs="+", default=PAPER_PROPORTIONS)
    parser.add_argument("--switch_relax", type=int, default=3)
    parser.add_argument("--time_gap", type=int, default=30)
    parser.add_argument("--num_processes", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--skip_anomalies", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    np_rng = np.random.default_rng(args.seed)
    torch_rng = torch.Generator().manual_seed(args.seed)
    preprocessing = Preprocessing(
        args.location,
        (args.min_length, args.max_length),
        RoadNetwork(args.location),
        (0.8, 0.1, 0.1),
        torch_rng,
        np_rng,
    )
    train, validation, test = preprocessing()
    logging.info(
        "Prepared %s: train=%d validation=%d test=%d",
        args.location,
        len(train),
        len(validation),
        len(test),
    )
    if args.skip_anomalies:
        return

    for split, trajectories in (("val", validation), ("test", test)):
        generator = AnomalyGenerator(
            location=args.location,
            anomaly_ratio=args.anomaly_ratio,
            test_trajectories=trajectories,
            rng=np_rng,
            random_seed=args.seed,
            switch_relax=args.switch_relax,
            shift_time_gap=args.time_gap,
            dataset_type=split,
        )
        for anomaly_type in PAPER_ANOMALIES:
            for proportion in args.proportions:
                generated = generator(
                    anomaly_type,
                    proportion,
                    args.num_processes,
                    args.batch_size,
                )
                logging.info(
                    "Generated %s/%s ratio=%.1f: %d trajectories",
                    split,
                    anomaly_type,
                    proportion,
                    len(generated),
                )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main(parse_args())
