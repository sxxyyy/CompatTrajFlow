import argparse
import ast
import os
import subprocess

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import pandas as pd

MODEL_CSV_NAMES = {
    "deep_tea": "DeepTea",
    "mst_oatd": "MST-OATD",
    "causal_tad": "CausalTAD",
    "gmvsae": "GMVSAE",
    "vsae": "VSAE",
    "fotraj": "FoTraj",
}


def main(
    seeds,
    models=None,
    anomaly_types=None,
    location="porto",
    batch_size=64,
    switch_relax=3,
    shift_time_gap=30,
    grid_height_km=1.0,
    grid_width_km=1.0,
    anomaly_proportions=None,
):
    if models is None:
        models = ["gmvsae"]
    if anomaly_types is None:
        anomaly_types = ["detour", "switch", "time_shift"]
    if not seeds:
        raise ValueError("at least one user-provided seed is required")
    num_runs = len(seeds)

    for model in models:
        print(f"\n{'=' * 50}\nStarting experiments for model: {model}\n{'=' * 50}")
        if model not in MODEL_CSV_NAMES:
            print(f"Warning: Unknown CSV mapping for model '{model}'. Skipping.")
            continue

        csv_name = MODEL_CSV_NAMES[model]

        for i, seed in enumerate(seeds):
            print(f"Starting run {i + 1}/{num_runs} for {model}...")
            cmd = (
                [
                    "python",
                    "main.py",
                    "--location",
                    location,
                    "-t",
                    "--force_train",
                    "-m",
                    model,
                    "-b",
                    str(batch_size),
                    "--switch_relax",
                    str(switch_relax),
                    "--shift_time_gap",
                    str(shift_time_gap),
                    "--anomaly_types",
                ]
                + anomaly_types
                + ["--random_seed", str(seed)]
            )
            if anomaly_proportions is not None:
                cmd += [
                    "--anomaly_proportions",
                ] + [str(p) for p in anomaly_proportions]
            if model != "causal_tad":
                cmd += [
                    "--use_grid_tokens",
                    "--grid_height_km",
                    str(grid_height_km),
                    "--grid_width_km",
                    str(grid_width_km),
                ]

            subprocess.run(cmd, check=True)

            src_csv = f"result/{location}/{csv_name}.csv"
            tgt_csv = f"result/{location}/{csv_name}_run_{i}.csv"
            if os.path.exists(src_csv):
                os.rename(src_csv, tgt_csv)
                print(f"Results for run {i + 1} saved to {tgt_csv}")
            else:
                print(f"Warning: {src_csv} not found after run {i + 1}.")

        print(f"\nAll runs completed for {model}. Calculating averages...")

        dfs = []
        for i in range(num_runs):
            tgt_csv = f"result/{location}/{csv_name}_run_{i}.csv"
            if os.path.exists(tgt_csv):
                dfs.append(pd.read_csv(tgt_csv))

        if not dfs:
            print(f"No result CSVs found for {model}. Skipping average calculation.")
            continue

        avg_results = {}
        for col in dfs[0].columns:
            metrics = {}
            for df in dfs:
                # Metric cells contain serialized dictionaries.
                val = ast.literal_eval(df[col].iloc[0])
                for k, v in val.items():
                    metrics[k] = metrics.get(k, 0.0) + (v / len(dfs))
            avg_results[col] = [str(metrics)]

        avg_df = pd.DataFrame(avg_results)
        avg_out_path = f"result/{location}/{csv_name}_average.csv"
        avg_df.to_csv(avg_out_path, index=False)

        print(f"\nAverage results for {model} saved to {avg_out_path}:")
        print(avg_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run multiple anomaly detection models"
    )
    parser.add_argument(
        "-m",
        "--models",
        nargs="+",
        default=["deep_tea", "causal_tad", "gmvsae", "vsae", "mst_oatd"],
        help="List of models to run (e.g. gmvsae causal_tad)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        required=True,
        help="Explicit seed for each repeated run",
    )
    parser.add_argument(
        "-a",
        "--anomaly_types",
        nargs="+",
        default=["detour", "switch", "time_shift"],
        help="Anomaly types",
    )
    parser.add_argument(
        "--location",
        type=str,
        default="porto",
        help="Dataset location (e.g. porto, xian)",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training",
    )
    parser.add_argument(
        "--switch_relax",
        default=3,
        type=int,
        help="Relaxation for switch anomaly extra random walk, default is 0",
    )
    parser.add_argument(
        "--shift_time_gap",
        default=30,
        type=int,
        help="Time gap for time shift anomaly in seconds, default is 3 seconds",
    )
    parser.add_argument(
        "--grid_height_km",
        default=0.1,
        type=float,
        help="Height of the grid in km for grid-token models",
    )
    parser.add_argument(
        "--grid_width_km",
        default=0.1,
        type=float,
        help="Width of the grid in km for grid-token models",
    )
    parser.add_argument(
        "--anomaly_proportions",
        nargs="+",
        type=float,
        default=[0.1, 0.3],
        help="Anomaly proportions (e.g. 0.1 0.5). Passed through to main.py.",
    )

    args = parser.parse_args()
    main(
        models=args.models,
        seeds=args.seeds,
        anomaly_types=args.anomaly_types,
        location=args.location,
        batch_size=args.batch_size,
        switch_relax=args.switch_relax,
        shift_time_gap=args.shift_time_gap,
        grid_height_km=args.grid_height_km,
        grid_width_km=args.grid_width_km,
        anomaly_proportions=args.anomaly_proportions,
    )
