"""Validation-only grid search for CompatTrajFlow branch scores.

The path and time branches are searched independently, then all cached branch
scores are combined with an uncalibrated raw sum.  Model selection uses only
the macro AUROC over Detour, Switch, and Time Shift validation anomalies at a
single perturbation ratio (0.3 by default).  Test data are never loaded.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import pickle
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchmetrics.functional.classification import binary_auroc
from tqdm import tqdm

from utils.edge_remapper import apply_edge_mapping
from utils.timestamp_converter import TimestampConverter


ANOMALY_TYPES = ("detour", "switch", "time_shift")
collate_fn = None


def _selected_anomaly_types(args) -> tuple[str, ...]:
    """Resolve anomaly types from --anomaly_types argument."""
    if args.anomaly_types:
        for t in args.anomaly_types:
            if t not in ANOMALY_TYPES:
                raise ValueError(f"Unknown anomaly type: {t}, must be one of {ANOMALY_TYPES}")
        return tuple(args.anomaly_types)
    return ANOMALY_TYPES


def anomaly_path(
    location: str,
    anomaly_type: str,
    ratio: float,
    switch_relax: int,
    shift_time_gap: int,
) -> Path:
    root = Path("data") / location / "anomaly"
    if anomaly_type == "switch":
        name = f"switch_{ratio}_relax_{switch_relax}_val.pkl"
    elif anomaly_type == "time_shift":
        name = f"time_shift_{ratio}_gap_{shift_time_gap}_val.pkl"
    else:
        name = f"{anomaly_type}_{ratio}_val.pkl"
    return root / name


def make_loader(dataset, batch_size: int, num_workers: int, sampler=None) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )


def load_validation_loaders(args, data_module, anomaly_types=None, rank=0, world_size=1):
    """Load clean validation and each specified validation anomaly set."""
    if anomaly_types is None:
        anomaly_types = ANOMALY_TYPES
    data_module.prepare_edge_mapping()
    data_module.setup(stage="validate")

    def _sampler(dataset):
        if world_size > 1:
            return DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
        return None

    val_ds = data_module.val_condition_dataset
    class _StrippedDataset(torch.utils.data.Dataset):
        def __init__(self, ds):
            self.ds = ds
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, idx):
            item = self.ds[idx]
            return (item[0], item[1])

    normal_dataset = _StrippedDataset(val_ds)
    loaders = {"normal": make_loader(normal_dataset, args.batch_size, args.num_workers, _sampler(normal_dataset))}

    for anomaly_type in anomaly_types:
        source = anomaly_path(
            args.location,
            anomaly_type,
            args.anomaly_ratio,
            args.switch_relax,
            args.shift_time_gap,
        )
        if not source.exists():
            raise FileNotFoundError(f"validation anomaly file not found: {source}")
        with source.open("rb") as handle:
            trajectories, _ = pickle.load(handle)
        apply_edge_mapping(trajectories, data_module.edge_mapping)
        dataset = data_module._load_and_tokenize_trajectories(
            trajectories,
            f"Loading all {anomaly_type} validation anomalies",
            1,
        )
        loaders[anomaly_type] = make_loader(
            dataset, args.batch_size, args.num_workers, _sampler(dataset)
        )
    return loaders


def expand_unconditional(unconditional, condition):
    if unconditional.ndim == condition.ndim:
        return unconditional
    return unconditional.view(
        *([1] * (condition.ndim - 1)), -1
    ).expand_as(condition)


def _all_gather_1d(tensor: torch.Tensor, device, world_size: int):
    """Gather differently sized 1-D tensors from all DDP ranks."""
    if world_size <= 1:
        return tensor.detach().cpu()
    tensor = tensor.detach().to(device)
    local_size = torch.tensor([tensor.numel()], dtype=torch.long, device=device)
    sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(sizes, local_size)
    max_size = max(s.item() for s in sizes)
    padded = torch.zeros(max_size, dtype=tensor.dtype, device=device)
    padded[: tensor.numel()] = tensor
    gathered = [torch.empty_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded)
    return torch.cat([p[: s.item()].cpu() for p, s in zip(gathered, sizes)])


@torch.inference_mode()
def score_loader(
    model,
    loader,
    times,
    scales,
    seeds,
    device,
    dataset_seed_offset: int,
    max_batches: int | None,
    rank: int = 0,
    world_size: int = 1,
    gather_device=None,
):
    """Return seed-averaged per-sample NLL for all branch parameter pairs."""
    path_parts = {(t, s): [] for t in times for s in scales}
    time_parts = {(t, s): [] for t in times for s in scales}
    if gather_device is None:
        gather_device = device

    total = len(loader) if max_batches is None else min(len(loader), max_batches)
    for batch_index, batch in enumerate(
        tqdm(loader, total=total, desc="Scoring", leave=False, disable=rank != 0)
    ):
        if max_batches is not None and batch_index >= max_batches:
            break
        # Clean validation additionally returns a hard-condition dictionary;
        # anomaly loaders return only targets and labels. Both expose the
        # target tuple at batch position zero.
        path, raw_time, masks = batch[0]
        path = path.to(device, non_blocking=True)
        raw_time = raw_time.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        time = model._transform_time(raw_time, masks)

        path_emb = F.normalize(
            model.path_embedding(path), p=2, dim=-1
        ) * math.sqrt(model.emb_dim)
        time_emb = F.normalize(
            model.time_embedding(time), p=2, dim=-1
        ) * math.sqrt(model.emb_dim)
        path_condition = model._path_condition(time_emb, path_emb, masks)
        path_unconditional = model._unconditional_path_condition(path_emb, masks)
        time_unconditional = expand_unconditional(model.uncond_path_emb, path_emb)
        valid_time_condition = path_emb.abs().sum(dim=-1, keepdim=True) > 0
        time_unconditional = time_unconditional * valid_time_condition

        batch_path = {
            (t, s): torch.zeros(path.size(0), device=device)
            for t in times
            for s in scales
        }
        batch_time = {
            (t, s): torch.zeros(path.size(0), device=device)
            for t in times
            for s in scales
        }

        for seed in seeds:
            generator = torch.Generator(device=device)
            generator.manual_seed(
                int(seed) + dataset_seed_offset + batch_index * 1_000_003
            )
            path_noise = torch.randn(
                path_emb.shape,
                dtype=path_emb.dtype,
                device=device,
                generator=generator,
            )
            time_noise = torch.randn(
                time_emb.shape,
                dtype=time_emb.dtype,
                device=device,
                generator=generator,
            )

            for time_value in times:
                t = torch.full(
                    (path.size(0),),
                    time_value,
                    device=device,
                    dtype=path_emb.dtype,
                )
                t_expanded = t.view(-1, 1, 1)
                path_z = t_expanded * path_emb + (1.0 - t_expanded) * path_noise
                time_z = t_expanded * time_emb + (1.0 - t_expanded) * time_noise

                path_cond = model.path_net(path_z, t, path_condition)
                path_uncond = model.path_net(path_z, t, path_unconditional)
                time_cond = model.time_net(time_z, t, path_emb)
                time_uncond = model.time_net(time_z, t, time_unconditional)

                path_v_cond = (path_cond - path_z) / (
                    1.0 - t_expanded
                ).clamp_min(model.t_eps)
                path_v_uncond = (path_uncond - path_z) / (
                    1.0 - t_expanded
                ).clamp_min(model.t_eps)
                time_v_cond = (time_cond - time_z) / (
                    1.0 - t_expanded
                ).clamp_min(model.t_eps)
                time_v_uncond = (time_uncond - time_z) / (
                    1.0 - t_expanded
                ).clamp_min(model.t_eps)

                low, high = model.cfg_interval
                cfg_active = low <= time_value < high
                for scale in scales:
                    effective_scale = scale if cfg_active else 1.0
                    path_v = path_v_uncond + effective_scale * (
                        path_v_cond - path_v_uncond
                    )
                    time_v = time_v_uncond + effective_scale * (
                        time_v_cond - time_v_uncond
                    )
                    path_prediction = path_z + path_v * (1.0 - t_expanded)
                    time_prediction = time_z + time_v * (1.0 - t_expanded)
                    batch_path[(time_value, scale)] += model._path_nll_per_sample(
                        path_prediction, path, masks
                    )
                    batch_time[(time_value, scale)] += model._time_nll_per_sample(
                        time_prediction, time, masks
                    )

        divisor = float(len(seeds))
        for key in batch_path:
            path_parts[key].append((batch_path[key] / divisor).float().cpu())
            time_parts[key].append((batch_time[key] / divisor).float().cpu())

    if not next(iter(path_parts.values())):
        raise RuntimeError("validation loader yielded no batches")
    
    # All-gather across DDP ranks
    if world_size > 1:
        gathered_path, gathered_time = {}, {}
        for key in path_parts:
            gathered_path[key] = _all_gather_1d(torch.cat(path_parts[key]), gather_device, world_size)
            gathered_time[key] = _all_gather_1d(torch.cat(time_parts[key]), gather_device, world_size)
        return gathered_path, gathered_time
    
    return (
        {key: torch.cat(parts) for key, parts in path_parts.items()},
        {key: torch.cat(parts) for key, parts in time_parts.items()},
    )


def auc(normal_scores: torch.Tensor, anomaly_scores: torch.Tensor) -> float:
    labels = torch.cat(
        (
            torch.zeros(normal_scores.numel(), dtype=torch.long),
            torch.ones(anomaly_scores.numel(), dtype=torch.long),
        )
    )
    return float(binary_auroc(torch.cat((normal_scores, anomaly_scores)), labels))


def branch_table(scores, branch: str):
    rows = []
    for (time_value, scale), normal_scores in scores["normal"].items():
        per_type = {
            anomaly_type: auc(normal_scores, scores[anomaly_type][(time_value, scale)])
            for anomaly_type in ANOMALY_TYPES
        }
        rows.append(
            {
                "branch": branch,
                "t": time_value,
                "guidance_scale": scale,
                **{f"{name}_auroc": value for name, value in per_type.items()},
                "macro_auroc": sum(per_type.values()) / len(per_type),
            } 
        )
    return rows


def joint_table(path_scores, time_scores):
    rows = []
    for path_key, normal_path in path_scores["normal"].items():
        for time_key, normal_time in time_scores["normal"].items():
            normal_joint = normal_path + normal_time
            per_type = {}
            for anomaly_type in ANOMALY_TYPES:
                anomaly_joint = (
                    path_scores[anomaly_type][path_key]
                    + time_scores[anomaly_type][time_key]
                )
                per_type[anomaly_type] = auc(normal_joint, anomaly_joint)
            rows.append(
                {
                    "path_t": path_key[0],
                    "path_guidance_scale": path_key[1],
                    "time_t": time_key[0],
                    "time_guidance_scale": time_key[1],
                    **{f"{name}_auroc": value for name, value in per_type.items()},
                    "macro_auroc": sum(per_type.values()) / len(per_type),
                }
            )
    return rows


def best_row(frame: pd.DataFrame) -> dict:
    # Stable tie breaking favors smaller t and guidance values.
    parameter_columns = [
        name
        for name in (
            "t",
            "guidance_scale",
            "path_t",
            "path_guidance_scale",
            "time_t",
            "time_guidance_scale",
        )
        if name in frame.columns
    ]
    ordered = frame.sort_values(
        ["macro_auroc", *parameter_columns],
        ascending=[False, *([True] * len(parameter_columns))],
    )
    return ordered.iloc[0].to_dict()


def main():
    global collate_fn
    parser = argparse.ArgumentParser(
        description="Grid-search separate path/time t and CFG scales on val AUROC"
    )
    parser.add_argument("--location", default="xian", choices=["xian", "porto"])
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument(
        "--model_module",
        default="flow_od_wt",
        help="Python module defining ConditionalFlowMatching, TokenTrajectoryDataModule, and collate_fn",
    )
    parser.add_argument(
        "--times", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5]
    )
    parser.add_argument(
        "--guidance_scales", nargs="+", type=float, default=[1, 3, 5, 7, 9]
    )
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--anomaly_ratio", type=float, default=0.3)
    parser.add_argument("--switch_relax", type=int, default=3)
    parser.add_argument("--shift_time_gap", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--output_dir", default="result/validation_grid_search")
    parser.add_argument(
        "--anomaly_types", nargs="+", default=None,
        choices=["detour", "switch", "time_shift"],
        help="Anomaly types to score (default: all three). Use to split across GPUs."
    )
    args = parser.parse_args()
    
    # ---- DDP init ----
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    
    anomaly_types = _selected_anomaly_types(args)

    model_module = importlib.import_module(args.model_module)
    ConditionalFlowMatching = model_module.ConditionalFlowMatching
    TokenTrajectoryDataModule = model_module.TokenTrajectoryDataModule
    collate_fn = model_module.collate_fn

    if not args.times or not args.guidance_scales or not args.seeds:
        raise ValueError("times, guidance_scales, and seeds must be non-empty")
    if any(not 0.0 < value < 1.0 for value in args.times):
        raise ValueError("every t must be strictly between zero and one")

    checkpoint = Path(args.checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    output_dir = Path(args.output_dir)

    # Load model on CPU BEFORE DDP init (avoids NCCL sync issues during load)
    print(f"Loading checkpoint: {checkpoint}")
    model = ConditionalFlowMatching.load_from_checkpoint(
        str(checkpoint), map_location="cpu", strict=False,
    ).eval()

    if distributed:
        device_index = local_rank
        torch.cuda.set_device(device_index)
        device = torch.device("cuda", device_index)
        dist.init_process_group(backend="nccl")
        if rank == 0:
            print(f"[DDP] {world_size} GPUs, rank {rank}")
        model = model.to(device)
        model = DDP(model, device_ids=[device_index], output_device=device_index)
        _model = model.module
    else:
        device = torch.device(args.device)
        model = model.to(device)
        _model = model

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    data_module = TokenTrajectoryDataModule(
        location=args.location,
        timestamp_converter=TimestampConverter(args.location),
        random_seed=args.seeds[0],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        test_anomaly_ratio=args.anomaly_ratio,
        switch_relax=args.switch_relax,
        shift_time_gap=args.shift_time_gap,
    )
    loaders = load_validation_loaders(args, data_module, anomaly_types=anomaly_types, rank=rank, world_size=world_size)

    path_scores = {}
    time_scores = {}
    for dataset_index, (name, loader) in enumerate(loaders.items()):
        if rank == 0:
            print(f"\nScoring validation dataset: {name}")
        path_scores[name], time_scores[name] = score_loader(
            _model,
            loader,
            tuple(args.times),
            tuple(args.guidance_scales),
            tuple(args.seeds),
            device,
            dataset_seed_offset=dataset_index * 100_000_007,
            max_batches=args.max_batches,
            rank=rank, world_size=world_size, gather_device=device,
        )

    if distributed:
        dist.barrier()

    if rank != 0:
        dist.destroy_process_group()
        return

    # Save per-anomaly-type cached scores for merge
    for at in anomaly_types:
        subset = {"normal": path_scores["normal"], at: path_scores[at]}
        torch.save(subset, output_dir / f"cached_path_scores_{at}.pt")
        subset = {"normal": time_scores["normal"], at: time_scores[at]}
        torch.save(subset, output_dir / f"cached_time_scores_{at}.pt")

    branch_frame = pd.DataFrame(
        branch_table(path_scores, "path") + branch_table(time_scores, "time")
    )
    joint_frame = pd.DataFrame(joint_table(path_scores, time_scores))
    branch_frame.to_csv(output_dir / "branch_grid.csv", index=False)
    joint_frame.to_csv(output_dir / "joint_grid.csv", index=False)
    torch.save(
        {"path": path_scores, "time": time_scores, "seeds": args.seeds},
        output_dir / "cached_scores.pt",
    )

    best = {
        "selection_split": "validation",
        "selection_anomaly_ratio": args.anomaly_ratio,
        "selection_metric": "macro AUROC over detour/switch/time_shift",
        "fusion": "raw path NLL + raw time NLL",
        "noise_aggregation": "mean per-sample score over user-provided seeds",
        "checkpoint": str(checkpoint.resolve()),
        "times": args.times,
        "guidance_scales": args.guidance_scales,
        "seeds": args.seeds,
        "best_path": best_row(branch_frame[branch_frame.branch == "path"]),
        "best_time": best_row(branch_frame[branch_frame.branch == "time"]),
        "best_joint": best_row(joint_frame),
    }
    with (output_dir / "best_params.json").open("w") as handle:
        json.dump(best, handle, indent=2)

    print("\nBest path parameters:", best["best_path"])
    print("Best time parameters:", best["best_time"])
    print("Best raw-sum joint parameters:", best["best_joint"])
    print(f"Results written to {output_dir}")

    if distributed:
        dist.destroy_process_group()
if __name__ == "__main__":
    main()
