#!/usr/bin/env python
"""Freeze modality-specific hyperparameters from validation score caches.

Each modality is selected independently by macro AUROC over the three
validation anomaly types.  AP is reported only as a diagnostic.  No joint or
raw-score fusion is computed in this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
from torchmetrics.functional.classification import binary_average_precision, binary_auroc


ANOMALIES = ("detour", "switch", "time_shift")
CONFIGS = {
    "porto": Path("result/porto_wt_wide_val_grid"),
    "xian": Path("result/xian_wt_wide_val_grid"),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(normal: torch.Tensor, anomaly: torch.Tensor) -> tuple[float, float]:
    scores = torch.cat((normal, anomaly)).float()
    labels = torch.cat(
        (
            torch.zeros(normal.numel(), dtype=torch.long),
            torch.ones(anomaly.numel(), dtype=torch.long),
        )
    )
    return (
        float(binary_auroc(scores, labels)),
        float(binary_average_precision(scores, labels)),
    )


def branch_rows(scores: dict, branch: str) -> list[dict]:
    rows = []
    for t_value, scale in sorted(scores["normal"]):
        key = (t_value, scale)
        per_type = {
            name: metrics(scores["normal"][key], scores[name][key])
            for name in ANOMALIES
        }
        rows.append(
            {
                "branch": branch,
                "modality": "X" if branch == "path" else "T",
                "t_r": float(t_value),
                "w": float(scale),
                **{f"{name}_auroc": value[0] for name, value in per_type.items()},
                **{f"{name}_ap": value[1] for name, value in per_type.items()},
                "macro_auroc": sum(value[0] for value in per_type.values()) / len(ANOMALIES),
                "macro_ap": sum(value[1] for value in per_type.values()) / len(ANOMALIES),
            }
        )
    return rows


def best_row(frame: pd.DataFrame) -> dict:
    # Stable tie break: smaller reconstruction time, then smaller CFG scale.
    ordered = frame.sort_values(
        ["macro_auroc", "t_r", "w"], ascending=[False, True, True], kind="mergesort"
    )
    return ordered.iloc[0].to_dict()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=Path("result/fourparam_zsum"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selections = []
    grid_rows = []
    sources = {}
    selection_seeds = {}
    for dataset, val_dir in CONFIGS.items():
        head = "wt"
        cache = val_dir / "cached_scores.pt"
        if not cache.exists():
            raise FileNotFoundError(cache)
        cached = torch.load(cache, map_location="cpu", weights_only=False)
        cache_seeds = cached.get("seeds")
        if not isinstance(cache_seeds, list) or not cache_seeds:
            raise ValueError(f"{cache} does not record user-provided seeds")
        selection_seeds[dataset] = [int(seed) for seed in cache_seeds]
        selected = {}
        for branch in ("path", "time"):
            frame = pd.DataFrame(branch_rows(cached[branch], branch))
            frame.insert(0, "head", head)
            frame.insert(0, "dataset", dataset)
            grid_rows.extend(frame.to_dict("records"))
            selected[branch] = best_row(frame)

        actual = (
            float(selected["path"]["t_r"]),
            float(selected["path"]["w"]),
            float(selected["time"]["t_r"]),
            float(selected["time"]["w"]),
        )
        row = {
            "dataset": dataset,
            "head": head,
            "path_t": actual[0],
            "path_w": actual[1],
            "time_t": actual[2],
            "time_w": actual[3],
            "path_macro_auroc": selected["path"]["macro_auroc"],
            "path_macro_ap_at_selection": selected["path"]["macro_ap"],
            "time_macro_auroc": selected["time"]["macro_auroc"],
            "time_macro_ap_at_selection": selected["time"]["macro_ap"],
        }
        selections.append(row)
        sources[f"{dataset}_{head}"] = {
            "cache": str(cache.resolve()),
            "cache_sha256": file_sha256(cache),
        }
        print(
            f"{dataset}/{head}: X=({actual[0]:g},{actual[1]:g}), "
            f"T=({actual[2]:g},{actual[3]:g})"
        )

    selection_frame = pd.DataFrame(selections)
    selection_frame.to_csv(args.output_dir / "frozen_fourparam_selections.csv", index=False)
    pd.DataFrame(grid_rows).to_csv(args.output_dir / "validation_modality_grids.csv", index=False)
    frozen = {
        "selection_split": "validation",
        "selection_ratio": 0.3,
        "selection_seeds": selection_seeds,
        "selection_metric": "equal-weight macro AUROC over detour, switch, and time_shift",
        "tie_break": "smaller t_r, then smaller w",
        "ap_used_for_selection": False,
        "fusion": "zscore_sum",
        "selections": selections,
        "sources": sources,
    }
    (args.output_dir / "frozen_fourparam_selections.json").write_text(
        json.dumps(frozen, indent=2) + "\n"
    )
    print(f"Frozen selections written to {args.output_dir}")


if __name__ == "__main__":
    main()
