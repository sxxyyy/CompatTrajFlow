#!/usr/bin/env python
"""Evaluate frozen modality-specific parameters with validation z-score fusion.

The evaluator never searches hyperparameters.  It reads a frozen selection
artifact, scores X and T at their own keys, calibrates both branches on normal
validation trajectories, and defines Z exactly as zX + zT.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import pickle
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from torchmetrics.functional.classification import binary_average_precision, binary_auroc
from tqdm import tqdm

import validation_grid_search as validation_grid
from utils.edge_remapper import apply_edge_mapping
from utils.timestamp_converter import TimestampConverter


ANOMALIES = ("detour", "switch", "time_shift")
RATIOS = (0.1, 0.3)
SCORES = ("x", "t", "z")
METRICS = tuple(f"{score}_{metric}" for score in SCORES for metric in ("auroc", "ap"))


def load_selection(path: Path, dataset: str, head: str) -> tuple[dict, dict]:
    frozen = json.loads(path.read_text())
    if frozen.get("selection_split") != "validation":
        raise ValueError("selection artifact is not validation-only")
    if frozen.get("fusion") != "zscore_sum":
        raise ValueError("selection artifact does not declare zscore_sum")
    matches = [
        row for row in frozen["selections"]
        if row["dataset"] == dataset and row["head"] == head
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one frozen selection for {dataset}/{head}")
    return matches[0], frozen


def expand_unconditional(unconditional: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
    if unconditional.ndim == condition.ndim:
        return unconditional
    return unconditional.view(*([1] * (condition.ndim - 1)), -1).expand_as(condition)


@torch.inference_mode()
def score_loader_selected(
    model,
    loader,
    path_key: tuple[float, float],
    time_key: tuple[float, float],
    seeds: tuple[int, ...],
    device: torch.device,
    dataset_seed_offset: int,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Score only X's and T's independently frozen parameter pairs."""
    parts = {seed: {"x": [], "t": []} for seed in seeds}
    for batch_index, batch in enumerate(tqdm(loader, desc="Scoring", leave=False)):
        path, raw_time, masks = batch[0]
        path = path.to(device, non_blocking=True)
        raw_time = raw_time.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        time = model._transform_time(raw_time, masks)

        path_emb = F.normalize(model.path_embedding(path), p=2, dim=-1) * math.sqrt(model.emb_dim)
        time_emb = F.normalize(model.time_embedding(time), p=2, dim=-1) * math.sqrt(model.emb_dim)
        path_condition = model._path_condition(time_emb, path_emb, masks)
        path_unconditional = model._unconditional_path_condition(path_emb, masks)
        time_unconditional = expand_unconditional(model.uncond_path_emb, path_emb)
        time_unconditional = time_unconditional * (path_emb.abs().sum(dim=-1, keepdim=True) > 0)

        for seed in seeds:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed) + dataset_seed_offset + batch_index * 1_000_003)
            path_noise = torch.randn(
                path_emb.shape, dtype=path_emb.dtype, device=device, generator=generator
            )
            time_noise = torch.randn(
                time_emb.shape, dtype=time_emb.dtype, device=device, generator=generator
            )

            path_t, path_w = path_key
            t_x = torch.full((path.size(0),), path_t, device=device, dtype=path_emb.dtype)
            tx3 = t_x.view(-1, 1, 1)
            path_z = tx3 * path_emb + (1.0 - tx3) * path_noise
            path_cond = model.path_net(path_z, t_x, path_condition)
            path_uncond = model.path_net(path_z, t_x, path_unconditional)
            path_v_cond = (path_cond - path_z) / (1.0 - tx3).clamp_min(model.t_eps)
            path_v_uncond = (path_uncond - path_z) / (1.0 - tx3).clamp_min(model.t_eps)
            low, high = model.cfg_interval
            effective_path_w = path_w if low <= path_t < high else 1.0
            path_v = path_v_uncond + effective_path_w * (path_v_cond - path_v_uncond)
            path_prediction = path_z + path_v * (1.0 - tx3)
            x_score = model._path_nll_per_sample(path_prediction, path, masks)

            time_t, time_w = time_key
            t_t = torch.full((path.size(0),), time_t, device=device, dtype=time_emb.dtype)
            tt3 = t_t.view(-1, 1, 1)
            time_z = tt3 * time_emb + (1.0 - tt3) * time_noise
            time_cond = model.time_net(time_z, t_t, path_emb)
            time_uncond = model.time_net(time_z, t_t, time_unconditional)
            time_v_cond = (time_cond - time_z) / (1.0 - tt3).clamp_min(model.t_eps)
            time_v_uncond = (time_uncond - time_z) / (1.0 - tt3).clamp_min(model.t_eps)
            effective_time_w = time_w if low <= time_t < high else 1.0
            time_v = time_v_uncond + effective_time_w * (time_v_cond - time_v_uncond)
            time_prediction = time_z + time_v * (1.0 - tt3)
            t_score = model._time_nll_per_sample(time_prediction, time, masks)

            parts[seed]["x"].append(x_score.float().cpu())
            parts[seed]["t"].append(t_score.float().cpu())

    if not parts[seeds[0]]["x"]:
        raise RuntimeError("loader yielded no batches")
    return {
        seed: (torch.cat(parts[seed]["x"]), torch.cat(parts[seed]["t"]))
        for seed in seeds
    }


def metric_pair(normal: torch.Tensor, anomaly: torch.Tensor) -> tuple[float, float]:
    values = torch.cat((normal, anomaly)).float()
    labels = torch.cat(
        (
            torch.zeros(normal.numel(), dtype=torch.long),
            torch.ones(anomaly.numel(), dtype=torch.long),
        )
    )
    return float(binary_auroc(values, labels)), float(binary_average_precision(values, labels))


def calibration_stats(pair: tuple[torch.Tensor, torch.Tensor]) -> dict[str, float]:
    x, t = pair
    return {
        "x_mean": float(x.mean()),
        "x_std": float(x.std(unbiased=False).clamp_min(1e-8)),
        "t_mean": float(t.mean()),
        "t_std": float(t.std(unbiased=False).clamp_min(1e-8)),
    }


def standardize(pair: tuple[torch.Tensor, torch.Tensor], stats: dict[str, float]) -> dict[str, torch.Tensor]:
    x, t = pair
    z_x = (x - stats["x_mean"]) / max(stats["x_std"], 1e-8)
    z_t = (t - stats["t_mean"]) / max(stats["t_std"], 1e-8)
    result = {"x": z_x, "t": z_t, "z": z_x + z_t}
    if not torch.equal(result["z"], result["x"] + result["t"]):
        raise AssertionError("Z is not exactly zX + zT")
    return result


def fill_metrics(row: dict, normal: dict[str, torch.Tensor], anomaly: dict[str, torch.Tensor]) -> None:
    for score in SCORES:
        auroc, ap = metric_pair(normal[score], anomaly[score])
        row[f"{score}_auroc"] = auroc
        row[f"{score}_ap"] = ap


def make_loader(dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=validation_grid.collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )


def validation_loaders(args, data_module):
    loader_args = SimpleNamespace(
        location=args.dataset,
        anomaly_ratio=0.3,
        switch_relax=3,
        shift_time_gap=30,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    # This helper only constructs normal/anomaly validation paths.
    return validation_grid.load_validation_loaders(loader_args, data_module)


def test_anomaly_path(dataset: str, kind: str, ratio: float) -> Path:
    root = Path("data") / dataset / "anomaly"
    if kind == "switch":
        return root / f"switch_{ratio}_relax_3_test.pkl"
    if kind == "time_shift":
        return root / f"time_shift_{ratio}_gap_30_test.pkl"
    return root / f"{kind}_{ratio}_test.pkl"


def test_loaders(args, data_module):
    data_module.prepare_edge_mapping()

    def tokenize(source: Path, description: str, label: int):
        with source.open("rb") as handle:
            loaded = pickle.load(handle)
        trajectories = loaded if isinstance(loaded, list) else loaded[0]
        missing = sorted(
            {
                edge
                for trajectory in trajectories
                for edge in trajectory.path
                if edge not in data_module.edge_mapping
            }
        )
        if missing:
            preview = ", ".join(str(edge) for edge in missing[:10])
            raise ValueError(
                f"{source} contains {len(missing)} raw edge ID(s) absent from "
                f"the frozen mapping ({preview}). Use the processed/anomaly "
                "snapshot that matches the checkpoint; dynamically extending "
                "the mapping would index beyond its embedding vocabulary."
            )
        apply_edge_mapping(trajectories, data_module.edge_mapping)
        return data_module._load_and_tokenize_trajectories(trajectories, description, label)

    val_normal = tokenize(
        Path("data") / args.dataset / "processed" / "val_trajectories.pkl",
        "Loading normal validation trajectories for calibration",
        0,
    )
    test_normal = tokenize(
        Path("data") / args.dataset / "processed" / "test_trajectories.pkl",
        "Loading normal test trajectories",
        0,
    )
    loaders = {
        "calibration_val_normal": make_loader(val_normal, args.batch_size, args.num_workers),
        "test_normal": make_loader(test_normal, args.batch_size, args.num_workers),
    }
    for kind in ANOMALIES:
        for ratio in RATIOS:
            dataset = tokenize(
                test_anomaly_path(args.dataset, kind, ratio),
                f"Loading {kind} {ratio:g} test anomalies",
                1,
            )
            loaders[(kind, ratio)] = make_loader(dataset, args.batch_size, args.num_workers)
    return loaders


def stability_frame(all_scores: dict, args, path_key, time_key) -> pd.DataFrame:
    rows = []
    seeds = sorted(all_scores)
    for anomaly in ANOMALIES:
        for score in SCORES:
            vectors = {
                seed: torch.cat((all_scores[seed]["normal"][score], all_scores[seed][anomaly][score])).numpy()
                for seed in seeds
            }
            correlations = [
                float(spearmanr(vectors[left], vectors[right]).statistic)
                for left, right in combinations(seeds, 2)
            ]
            stacked = torch.stack(
                [
                    torch.cat((all_scores[seed]["normal"][score], all_scores[seed][anomaly][score]))
                    for seed in seeds
                ]
            )
            sample_std = stacked.std(dim=0, unbiased=True)
            rows.append(
                {
                    "dataset": args.dataset,
                    "head": args.head,
                    "anomaly_type": anomaly,
                    "score": score.upper(),
                    "path_t": path_key[0], "path_w": path_key[1],
                    "time_t": time_key[0], "time_w": time_key[1],
                    "pairwise_spearman_mean": sum(correlations) / len(correlations),
                    "pairwise_spearman_std": float(torch.tensor(correlations).std(unbiased=True)),
                    "pairwise_spearman_min": min(correlations),
                    "mean_sample_score_std": float(sample_std.mean()),
                    "median_sample_score_std": float(sample_std.median()),
                }
            )
    return pd.DataFrame(rows)


def evaluate_validation(model, loaders, args, path_key, time_key, device):
    raw_by_dataset = {}
    for dataset_index, (name, loader) in enumerate(loaders.items()):
        print(f"Scoring validation dataset: {name}", flush=True)
        raw_by_dataset[name] = score_loader_selected(
            model, loader, path_key, time_key, tuple(args.seeds), device,
            dataset_index * 100_000_007,
        )

    rows = []
    all_scores = {}
    calibration = {}
    for seed in args.seeds:
        raw = {name: values[seed] for name, values in raw_by_dataset.items()}
        stats = calibration_stats(raw["normal"])
        calibration[str(seed)] = stats
        standardized = {name: standardize(pair, stats) for name, pair in raw.items()}
        all_scores[seed] = standardized
        per_type = []
        for anomaly in ANOMALIES:
            row = {
                "dataset": args.dataset, "head": args.head, "seed": seed,
                "anomaly_type": anomaly, "aggregation": "per_type",
                "path_t": path_key[0], "path_w": path_key[1],
                "time_t": time_key[0], "time_w": time_key[1],
            }
            fill_metrics(row, standardized["normal"], standardized[anomaly])
            rows.append(row)
            per_type.append(row)
        macro = {
            "dataset": args.dataset, "head": args.head, "seed": seed,
            "anomaly_type": "macro", "aggregation": "macro",
            "path_t": path_key[0], "path_w": path_key[1],
            "time_t": time_key[0], "time_w": time_key[1],
        }
        for metric in METRICS:
            macro[metric] = sum(row[metric] for row in per_type) / len(per_type)
        rows.append(macro)
        print(f"seed {seed} macro Z AUROC={macro['z_auroc']:.6f}, AP={macro['z_ap']:.6f}")

    frame = pd.DataFrame(rows)
    summary = frame.groupby(
        ["dataset", "head", "anomaly_type", "aggregation", "path_t", "path_w", "time_t", "time_w"]
    )[list(METRICS)].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return frame, summary.reset_index(), stability_frame(all_scores, args, path_key, time_key), calibration


def evaluate_test(model, loaders, args, path_key, time_key, device):
    if len(args.seeds) != 1:
        raise ValueError("test evaluation requires exactly one user-provided seed")
    test_seed = args.seeds[0]
    raw = {}
    anomaly_order = [(kind, ratio) for kind in ANOMALIES for ratio in RATIOS]
    for name, loader in loaders.items():
        if name == "calibration_val_normal":
            offset = 0
        elif name == "test_normal":
            offset = 700_000_049
        else:
            offset = (anomaly_order.index(name) + 8) * 100_000_007
        print(f"Scoring: {name}", flush=True)
        raw[name] = score_loader_selected(
            model, loader, path_key, time_key, (test_seed,), device, offset
        )[test_seed]

    stats = calibration_stats(raw["calibration_val_normal"])
    normal = standardize(raw["test_normal"], stats)
    standardized = {key: standardize(pair, stats) for key, pair in raw.items() if isinstance(key, tuple)}
    rows = []
    for kind in ANOMALIES:
        for ratio in RATIOS:
            row = {
                "dataset": args.dataset, "head": args.head, "seed": test_seed,
                "anomaly_type": kind, "ratio": ratio, "aggregation": "per_type",
                "path_t": path_key[0], "path_w": path_key[1],
                "time_t": time_key[0], "time_w": time_key[1],
            }
            fill_metrics(row, normal, standardized[(kind, ratio)])
            rows.append(row)
    for ratio in RATIOS:
        per_type = [row for row in rows if row["ratio"] == ratio and row["aggregation"] == "per_type"]
        macro = {
            "dataset": args.dataset, "head": args.head, "seed": test_seed,
            "anomaly_type": "macro", "ratio": ratio, "aggregation": "macro",
            "path_t": path_key[0], "path_w": path_key[1],
            "time_t": time_key[0], "time_w": time_key[1],
        }
        for metric in METRICS:
            macro[metric] = sum(row[metric] for row in per_type) / len(per_type)
        rows.append(macro)

        pooled_anomaly = {
            score: torch.cat([standardized[(kind, ratio)][score] for kind in ANOMALIES])
            for score in SCORES
        }
        pooled = {
            "dataset": args.dataset, "head": args.head, "seed": test_seed,
            "anomaly_type": "pooled", "ratio": ratio, "aggregation": "pooled_diagnostic",
            "path_t": path_key[0], "path_w": path_key[1],
            "time_t": time_key[0], "time_w": time_key[1],
        }
        fill_metrics(pooled, normal, pooled_anomaly)
        rows.append(pooled)
        print(f"ratio {ratio:g} macro Z AUROC={macro['z_auroc']:.6f}, AP={macro['z_ap']:.6f}")
    return pd.DataFrame(rows), stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=["validation", "test"])
    parser.add_argument("--dataset", required=True, choices=["porto", "xian"])
    parser.add_argument("--head", required=True, choices=["wt"])
    parser.add_argument("--model_module", required=True)
    parser.add_argument("--checkpoint_path", type=Path, required=True)
    parser.add_argument(
        "--selection_json", type=Path,
        default=Path("result/fourparam_zsum/frozen_fourparam_selections.json"),
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        required=True,
        help="User-provided reconstruction-noise seeds.",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.split == "validation" and len(args.seeds) < 2:
        raise ValueError("validation evaluation requires at least two user-provided seeds")
    if args.split == "test" and len(args.seeds) != 1:
        raise ValueError("test evaluation requires exactly one user-provided seed")

    selection, frozen = load_selection(args.selection_json, args.dataset, args.head)
    path_key = (float(selection["path_t"]), float(selection["path_w"]))
    time_key = (float(selection["time_t"]), float(selection["time_w"]))
    print(f"Frozen four-tuple: X={path_key}, T={time_key}", flush=True)

    module = importlib.import_module(args.model_module)
    validation_grid.collate_fn = module.collate_fn
    if not args.checkpoint_path.exists():
        raise FileNotFoundError(args.checkpoint_path)
    device = torch.device(args.device)
    model = module.ConditionalFlowMatching.load_from_checkpoint(
        str(args.checkpoint_path), map_location="cpu", strict=False
    ).eval().to(device)
    data_module = module.TokenTrajectoryDataModule(
        location=args.dataset,
        timestamp_converter=TimestampConverter(args.dataset),
        random_seed=args.seeds[0],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        test_anomaly_ratio=0.3,
        switch_relax=3,
        shift_time_gap=30,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.split == "validation":
        loaders = validation_loaders(args, data_module)
        frame, summary, stability, calibration = evaluate_validation(
            model, loaders, args, path_key, time_key, device
        )
        frame.to_csv(args.output_dir / "per_seed_validation_metrics.csv", index=False)
        summary.to_csv(args.output_dir / "validation_metrics_mean_std.csv", index=False)
        stability.to_csv(args.output_dir / "validation_score_stability.csv", index=False)
    else:
        loaders = test_loaders(args, data_module)
        frame, test_calibration = evaluate_test(
            model, loaders, args, path_key, time_key, device
        )
        frame.to_csv(args.output_dir / "test_metrics.csv", index=False)
        calibration = {str(args.seeds[0]): test_calibration}

    metadata = {
        "evaluation_split": args.split,
        "dataset": args.dataset,
        "head": args.head,
        "checkpoint": str(args.checkpoint_path.resolve()),
        "selection_artifact": str(args.selection_json.resolve()),
        "selection_artifact_snapshot": {
            "selection_split": frozen["selection_split"],
            "selection_ratio": frozen["selection_ratio"],
            "selection_seeds": frozen["selection_seeds"],
            "selection_metric": frozen["selection_metric"],
            "ap_used_for_selection": frozen["ap_used_for_selection"],
        },
        "path_key": {"t_r": path_key[0], "w": path_key[1]},
        "time_key": {"t_r": time_key[0], "w": time_key[1]},
        "seeds": args.seeds,
        "fusion": "zscore_sum",
        "fusion_definition": "Z = zX + zT, computed sample by sample",
        "calibration": "per-seed normal-validation mean/std; std clamped at 1e-8",
        "calibration_statistics": calibration,
        "anomaly_types": list(ANOMALIES),
        "ratios": [0.3] if args.split == "validation" else list(RATIOS),
        "macro_definition": "equal-weight mean of per-anomaly-type metrics",
        "pooled_metrics": "not computed" if args.split == "validation" else "diagnostic only",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
