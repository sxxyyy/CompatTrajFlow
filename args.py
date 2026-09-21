"""Command-line arguments for the baseline pipeline."""

import argparse
from argparse import Namespace


def init_args() -> Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog="trajectory-baselines")

    parser.add_argument_group("General settings")
    parser.add_argument("--random_seed", required=True, type=int)
    parser.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="Device to use and program will auto select the available device if not set",
    )
    parser.add_argument(
        "--location", default="porto", type=str, choices=["porto", "xian"]
    )
    parser.add_argument(
        "--use_grid_tokens",
        action="store_true",
        help="whether to use grid ID instead of edge ID for training",
    )
    parser.add_argument(
        "--grid_height_km", default=0.1, type=float, help="Height of the grid in km"
    )
    parser.add_argument(
        "--grid_width_km", default=0.1, type=float, help="Width of the grid in km"
    )
    parser.add_argument(
        "--trajectory_len_threshold",
        default=[10, 300],
        type=int,
        nargs=2,
        help="threshold for trajectory length, trajectories with length outside the threshold will be filtered out",
    )
    parser.add_argument(
        "--force_train",
        action="store_true",
        help="Force training even if checkpoint exists",
    )
    parser.add_argument(
        "-t",
        "--train",
        action="store_true",
        help="Run training, if not set, only run testing",
    )
    parser.add_argument(
        "--data_split_ratio",
        nargs=3,
        default=[0.8, 0.1, 0.1],
        type=float,
        help="Train, val, test split ratio",
    )
    parser.add_argument(
        "--num_workers",
        default=16,
        type=int,
        help="Number of workers for data loading",
    )
    parser.add_argument("--epoch", default=50, type=int, help="Number of epochs")
    parser.add_argument("-b", "--batch_size", default=16, type=int, help="Batch size")
    parser.add_argument(
        "--test_batch_size", default=16, type=int, help="Test batch size"
    )
    parser.add_argument(
        "--batch_log_interval", default=100, type=int, help="Log every n batches"
    )
    parser.add_argument(
        "--test_log_interval",
        default=100,
        type=int,
        help="Log every n batches during testing",
    )

    parser.add_argument(
        "-a", "--use_amp", action="store_true", help="Use automatic mixed precision"
    )

    parser.add_argument_group("Early stopping settings")
    parser.add_argument(
        "--patience", default=1, type=int, help="Early stopping patience"
    )
    parser.add_argument(
        "--relative_delta",
        default=0.10,
        type=float,
        help="Early stopping delta",
    )

    parser.add_argument_group("Anomaly settings")
    parser.add_argument(
        "-o",
        "--observation_ratio",
        default=1.0,
        type=float,
        help="Ratio of observations for each trajectory",
    )
    parser.add_argument(
        "--anomaly_types",
        nargs="+",
        default=["detour", "switch", "grid_detour", "time_shift"],
        type=str,
        choices=[
            "detour",
            "switch",
            "detour_without_time",
            "grid_detour",
            "time_shift",
        ],
        help="Type of anomaly to generate",
    )
    parser.add_argument(
        "--switch_relax",
        default=0,
        type=int,
        help="Relaxation for switch anomaly extra random walk, default is 0 * 2 more edges than the original switch point",
    )
    parser.add_argument(
        "--shift_time_gap",
        default=3,
        type=int,
        help="Time gap for time shift anomaly in seconds, default is 3 seconds",
    )
    parser.add_argument(
        "--anomaly_ratio",
        default=0.05,
        type=float,
        help="Ratio of anomalies in the dataset",
    )
    parser.add_argument(
        "--anomaly_proportions",
        nargs="+",
        default=[0.1, 0.3],
        type=float,
        help="Proportions of a trajectory to be anomalous",
    )

    parser.add_argument_group("Test Models")
    parser.add_argument(
        "-m",
        "--models",
        nargs="+",
        default=[
            "mst_oatd",
            "causal_tad",
            "gmvsae",
            "vsae",
        ],
        choices=[
            "deep_tea",
            "mst_oatd",
            "causal_tad",
            "gmvsae",
            "vsae",
            "fotraj",
            "traj_mllm",
        ],
        type=str,
        help="Choose one or more models to run in the program",
    )

    parser.add_argument_group("General hyperparameters")
    parser.add_argument("-e", "--embedding_size", default=128, type=int)
    parser.add_argument("--hidden_size", default=256, type=int)

    parser.add_argument_group("MST_OATD hyperparameters")
    parser.add_argument("--mst_num_clusters", default=10, type=float)
    parser.add_argument("--s_learning_rate", default=2e-4, type=float)
    parser.add_argument("--t_learning_rate", default=8e-5, type=float)
    parser.add_argument("--s1_size", default=2, type=int)
    parser.add_argument("--s2_size", default=4, type=int)

    parser.add_argument_group("CausalTAD hyperparameters")
    parser.add_argument("--causal_tad_learning_rate", default=1e-3, type=float)
    parser.add_argument("--causal_tad_weight_decay", default=1e-4, type=float)

    parser.add_argument_group("DeepTea hyperparameters")
    parser.add_argument(
        "--deeptea_speed_map_time_interval",
        default=20,
        type=int,
        help="Time interval for each speed map in DeepTea, in minutes",
    )
    parser.add_argument("--in_channels", default=1, type=int)
    parser.add_argument("--kernel_size", default=5, type=int)
    parser.add_argument("--deeptea_num_clusters", default=10, type=int)
    parser.add_argument("--deeptea_learning_rate", default=1e-4, type=float)

    parser.add_argument_group("GMVSAE hyperparameters")
    parser.add_argument("--gmvsae_num_clusters", default=10, type=int)
    parser.add_argument("--gmvsae_learning_rate", default=1e-4, type=float)

    parser.add_argument_group("VSAE hyperparameters")
    parser.add_argument("--vsae_learning_rate", default=1e-4, type=float)

    parser.add_argument_group("FOTraj hyperparameters")
    parser.add_argument("--fotraj_learning_rate", default=2e-4, type=float)
    parser.add_argument(
        "--fotraj_dropout", default=0.05, type=float, help="Dropout rate in the model"
    )
    parser.add_argument(
        "--fotraj_llm_path", default="meta-llama/Llama-3.1-8B-Instruct", type=str
    )
    parser.add_argument(
        "--fotraj_drop_patch_prob", type=float, default=0.2, help="Drop patch prob"
    )

    parser.add_argument_group("TrajMLLM hyperparameters")
    parser.add_argument(
        "--mllm_model_name",
        default="o4-mini",
        type=str,
        help="MLLM model name (default: o4-mini)",
    )
    parser.add_argument(
        "--mllm_base_url",
        default="https://api.openai.com/v1",
        type=str,
        help="MLLM API base URL",
    )
    parser.add_argument(
        "--mllm_work_dir",
        default=None,
        type=str,
        help="Working directory for intermediate files (JSON, PNG, cache). "
        "Defaults to data/<location>/traj_mllm_work/",
    )
    parser.add_argument(
        "--mllm_render_workers",
        default=256,
        type=int,
        help="Concurrent Puppeteer pages for Road-Network PNG rendering (default: 8).",
    )
    parser.add_argument(
        "--mllm_poi_workers",
        default=16,
        type=int,
        help="Concurrent Puppeteer pages for POI PNG rendering (default: 4, max: 16). "
        "POI uses OpenStreetMap tiles which are heavier; keep this lower than road.",
    )
    parser.add_argument(
        "--mllm_max_trajectories",
        default=500,
        type=int,
        help="Maximum number of test trajectories to process with the MLLM "
        "runner.  Set to -1 for unlimited (warning: 120K+ trajectories will "
        "take days).  Default: 500.",
    )
    return parser.parse_args()
