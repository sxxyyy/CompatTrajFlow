#!/usr/bin/env python
"""Evaluate WT four-parameter z-score fusion on Traj-MLLM's 500-item subsets."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import pickle
from pathlib import Path

import pandas as pd
import torch

import validation_grid_search as validation_grid
from evaluate_fourparam_zsum import (
    ANOMALIES,
    METRICS,
    RATIOS,
    fill_metrics,
    load_selection,
    make_loader,
    score_loader_selected,
    standardize,
    test_anomaly_path,
)
from utils.edge_remapper import apply_edge_mapping
from utils.timestamp_converter import TimestampConverter


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def load_trajectories(path: Path):
    with path.open("rb") as handle:
        loaded = pickle.load(handle)
    return loaded if isinstance(loaded, list) else loaded[0]


def tokenize_subset(data_module, source: Path, indices: list[int], description: str, label: int):
    trajectories = load_trajectories(source)
    if not indices or min(indices) < 0 or max(indices) >= len(trajectories):
        raise IndexError(f"subset indices are invalid for {source}")
    selected = [trajectories[index] for index in indices]
    apply_edge_mapping(selected, data_module.edge_mapping)
    return data_module._load_and_tokenize_trajectories(selected, description, label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["porto", "xian"])
    parser.add_argument("--model_module", default="flow_od_wt")
    parser.add_argument("--checkpoint_path", type=Path, required=True)
    parser.add_argument(
        "--selection_json", type=Path,
        default=Path("result/fourparam_zsum/frozen_fourparam_selections.json"),
    )
    parser.add_argument("--official_test_metadata", type=Path, required=True)
    parser.add_argument("--inference_seed", type=int, required=True)
    parser.add_argument("--subset_seed", type=int, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    selection, frozen = load_selection(args.selection_json, args.dataset, "wt")
    path_key = (float(selection["path_t"]), float(selection["path_w"]))
    time_key = (float(selection["time_t"]), float(selection["time_w"]))
    official = json.loads(args.official_test_metadata.read_text())
    if (
        official["evaluation_split"] != "test"
        or official["seeds"] != [args.inference_seed]
    ):
        raise ValueError(
            "official metadata must match the user-provided inference seed"
        )
    official_keys = (
        official["path_key"]["t_r"], official["path_key"]["w"],
        official["time_key"]["t_r"], official["time_key"]["w"],
    )
    if official_keys != (*path_key, *time_key):
        raise ValueError("official test metadata and frozen selection disagree")
    calibration = official["calibration_statistics"][str(args.inference_seed)]

    module = importlib.import_module(args.model_module)
    validation_grid.collate_fn = module.collate_fn
    device = torch.device(args.device)
    model = module.ConditionalFlowMatching.load_from_checkpoint(
        str(args.checkpoint_path), map_location="cpu", strict=False
    ).eval().to(device)
    data_module = module.TokenTrajectoryDataModule(
        location=args.dataset,
        timestamp_converter=TimestampConverter(args.dataset),
        random_seed=args.subset_seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        test_anomaly_ratio=0.3,
        switch_relax=3,
        shift_time_gap=30,
    )
    data_module.prepare_edge_mapping()

    subset_root = (
        Path("data")
        / args.dataset
        / "traj_mllm_work"
        / f"seed_{args.subset_seed}"
        / "subsets"
    )
    manifests = {}
    normal_indices = None
    for kind in ANOMALIES:
        for ratio in RATIOS:
            path = subset_root / f"{kind}_{ratio}_selected_indices.json"
            manifest = json.loads(path.read_text())
            if (
                manifest["random_seed"] != args.subset_seed
                or manifest["max_trajectories"] != 500
            ):
                raise ValueError(f"unexpected subset protocol in {path}")
            if len(manifest["selected_anomaly_indices"]) != 24:
                raise ValueError(f"expected 24 anomalies in {path}")
            if len(manifest["selected_normal_indices"]) != 476:
                raise ValueError(f"expected 476 normal samples in {path}")
            current_normal = manifest["selected_normal_indices"]
            if normal_indices is None:
                normal_indices = current_normal
            elif current_normal != normal_indices:
                raise ValueError("normal subset differs across perturbations")
            manifests[(kind, ratio)] = (path, manifest)

    normal_dataset = tokenize_subset(
        data_module,
        Path("data") / args.dataset / "processed" / "test_trajectories.pkl",
        normal_indices,
        "Loading Traj-MLLM normal test subset",
        0,
    )
    normal_raw = score_loader_selected(
        model, make_loader(normal_dataset, args.batch_size, args.num_workers),
        path_key, time_key, (args.inference_seed,), device, 700_000_049,
    )[args.inference_seed]
    normal = standardize(normal_raw, calibration)

    anomaly_order = [(kind, ratio) for kind in ANOMALIES for ratio in RATIOS]
    rows = []
    manifest_metadata = {}
    for kind, ratio in anomaly_order:
        manifest_path, manifest = manifests[(kind, ratio)]
        anomaly_dataset = tokenize_subset(
            data_module,
            test_anomaly_path(args.dataset, kind, ratio),
            manifest["selected_anomaly_indices"],
            f"Loading Traj-MLLM {kind} {ratio:g} subset",
            1,
        )
        offset = (anomaly_order.index((kind, ratio)) + 8) * 100_000_007
        anomaly_raw = score_loader_selected(
            model, make_loader(anomaly_dataset, args.batch_size, args.num_workers),
            path_key, time_key, (args.inference_seed,), device, offset,
        )[args.inference_seed]
        anomaly = standardize(anomaly_raw, calibration)
        row = {
            "dataset": args.dataset,
            "head": "wt",
            "inference_seed": args.inference_seed,
            "subset_seed": args.subset_seed,
            "anomaly_type": kind,
            "ratio": ratio,
            "n_normal": len(normal_indices),
            "n_anomaly": len(manifest["selected_anomaly_indices"]),
            "path_t": path_key[0], "path_w": path_key[1],
            "time_t": time_key[0], "time_w": time_key[1],
        }
        fill_metrics(row, normal, anomaly)
        rows.append(row)
        manifest_metadata[f"{kind}_{ratio}"] = {
            "path": str(manifest_path.resolve()),
            "sha256": sha256(manifest_path),
        }
        print(
            f"{kind} {ratio:g}: Z AP={row['z_ap']:.6f}, "
            f"AUROC={row['z_auroc']:.6f}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "subset_500_metrics.csv", index=False)
    metadata = {
        "evaluation_split": "test_subset",
        "subset_name": f"Traj-MLLM seed-{args.subset_seed} subset",
        "dataset": args.dataset,
        "head": "wt",
        "checkpoint": str(args.checkpoint_path.resolve()),
        "selection_artifact": str(args.selection_json.resolve()),
        "selection_artifact_selection_seeds": frozen["selection_seeds"],
        "path_key": {"t_r": path_key[0], "w": path_key[1]},
        "time_key": {"t_r": time_key[0], "w": time_key[1]},
        "inference_seed": args.inference_seed,
        "subset_seed": args.subset_seed,
        "n_normal": 476,
        "n_anomaly": 24,
        "fusion": "zscore_sum",
        "calibration": (
            "reused user-seeded normal-validation statistics from official "
            "test metadata"
        ),
        "calibration_statistics": calibration,
        "official_test_metadata": str(args.official_test_metadata.resolve()),
        "manifests": manifest_metadata,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
