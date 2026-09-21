import logging
import os
from argparse import Namespace
from collections.abc import Callable
from os.path import exists

import torch
from runner.causal_tad_runner import CausalTADRunner
from runner.deep_tea_runner import DeepTeaRunner
from runner.fotraj_runner import FOTrajRunner
from runner.gmvsae_runner import GMVSAERunner
from runner.mst_oatd_runner import MSTOATDRunner
from runner.traj_mllm_runner import TrajMLLMRunner
from runner.vsae_runner import VSAERunner
from torch import Generator
from torch.utils.data import DataLoader

from utils.collate_fns import (
    causal_tad_collate_fn,
    deep_tea_collate_fn,
    fotraj_collate_fn,
    gmvsae_collate_fn,
    mst_oatd_collate_fn,
    vsae_collate_fn,
)
from utils.timestamp_converter import TimestampConverter
from utils.trajectories_dataset import (
    GraphTrajectoryDataset,
    TokenTrajectoryDataset,
    TrajectoryDataset,
)
from utils.trajectory import TokenTrajectory, Trajectory

logger = logging.getLogger(__name__)


def create_dir_if_not_exists(
    directory: str, custom_msg_create: str, custom_msg_exists: str
):
    if not exists(directory):
        os.makedirs(directory)
        logger.info("Created directory: %s", directory)
        logger.info(custom_msg_create)
    else:
        logger.info("Directory already exists: %s", directory)
        logger.info(custom_msg_exists)


def set_device(args: Namespace):
    """
    Sets the default torch device based on the provided arguments and system capabilities.
    This function determines the appropriate device (CPU, CUDA, or MPS) to use for PyTorch operations.
    It prioritizes the device specified in `args.device`, then checks for MPS (Apple Silicon), then CUDA (NVIDIA GPU),
    and falls back to CPU if none are available. If a CUDA device is selected, it resets the CUDA memory cache.
    The selected device is set as the default for all subsequent torch operations.
    Args:
        args (Namespace): An argument namespace expected to have a `device` attribute specifying the desired device
                          (e.g., "cpu", "cuda", "cuda:0", "mps"), or None to auto-select.
    Returns:
        torch.device: The torch device object that has been set as default.
    Logs:
        Logs the selected device using the logging module.
    """

    if args.device is not None:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        # For macOS with M-series GPU support
        device = torch.device("mps")
    elif torch.cuda.is_available():
        # For NVIDIA GPUs
        device = torch.device("cuda")
    else:
        # Fallback to CPU
        device = torch.device("cpu")
    torch.set_default_device(device)
    logger.info("Set default device: %s", device)
    return device


def get_collate_fn(model_name: str) -> Callable:
    """Get different collate function based on model

    Args:
        model_name (str): model name from the args -m option

    Raises:
        ValueError: mismatch model name

    Returns:
        Callable: collate_fn in Dataloader
    """
    logger.info("Get collate_fn for model: %s", model_name)
    match model_name:
        case "deep_tea":
            return deep_tea_collate_fn
        case "mst_oatd":
            return mst_oatd_collate_fn
        case "causal_tad":
            return causal_tad_collate_fn
        case "gmvsae":
            return gmvsae_collate_fn
        case "vsae":
            return vsae_collate_fn
        case "fotraj":
            return fotraj_collate_fn
        case "traj_mllm":
            # MLLM runner uses raw Trajectory objects; the default collate
            # simply collects them into a list.
            return _identity_collate
        case _:
            raise ValueError(f"No matched collate_fn for model name : {model_name}!")


def get_runner(model_name: str):
    """
    Returns the appropriate runner class based on the provided model name.
    Args:
        model_name (str): The name of the model (args -m option) for which to get the runner class.
    Returns:
        type: The runner class corresponding to the given model name.
    Raises:
        ValueError: If the model name does not match any known runner.
    """
    logger.info("Get runner for model: %s", model_name)
    match model_name:
        case "deep_tea":
            return DeepTeaRunner
        case "mst_oatd":
            return MSTOATDRunner
        case "causal_tad":
            return CausalTADRunner
        case "gmvsae":
            return GMVSAERunner
        case "vsae":
            return VSAERunner
        case "fotraj":
            return FOTrajRunner
        case "traj_mllm":
            return TrajMLLMRunner
        case _:
            raise ValueError(f"No matched trainer for model name : {model_name}!")


def _identity_collate(batch):
    """Collate function that returns the batch as-is (list of tuples).

    Used by the TrajMLLM runner which needs raw ``Trajectory`` objects.
    """
    return batch


def get_train_dataset_by_model(
    args: Namespace,
    model_name: str,
    train: list[Trajectory],
    test: list[Trajectory],
    val: list[Trajectory],
    anomaly_trajectories_dict: dict[str, list[Trajectory]],
    timestamp_converter: TimestampConverter,
) -> (
    tuple[TokenTrajectoryDataset, None, dict[str, TokenTrajectoryDataset]]
    | tuple[
        GraphTrajectoryDataset,
        GraphTrajectoryDataset,
        dict[str, GraphTrajectoryDataset],
    ]
):
    """
    Returns the appropriate dataset based on the provided model name.
    Args:
        model_name (str): The name of the model (args -m option) for which to get the dataset.
        train (list[Trajectory]): List of training trajectories.
        test (list[Trajectory]): List of testing trajectories.
        val (list[Trajectory]): List of validation trajectories.
        anomaly_trajectories (list[Trajectory]): List of anomaly trajectories.
        timestamp_converter (TimestampConverter): Converter for timestamps to tokens.
    Returns:
        tuple[TrajectoryDataset, TrajectoryDataset]: A tuple containing:
            - Training dataset as TrajectoryDataset.
            - Testing dataset as TrajectoryDataset.
    Raises:
        ValueError: If the model name does not match any known dataset type.
    """
    logger.info("Get dataset for model: %s", model_name)
    match model_name:
        case "deep_tea":
            logger.info("Tokenizing training trajectories for %s", model_name)
            tokenized_train = []
            for i, t in enumerate(train):
                time = timestamp_converter(t)
                tokenized_train.append(
                    TokenTrajectory(path=t.path, time=time, speed=t.speed)
                )
                if i % 10000 == 0:  # Log every 10000 trajectories
                    logger.info(
                        "Tokenized trajectory %d/%d",
                        i,
                        len(train),
                    )
            logger.info("Tokenizing test trajectories for %s", model_name)
            tokenized_test = []
            for t in test:
                time = timestamp_converter(t)
                tokenized_test.append(
                    TokenTrajectory(path=t.path, time=time, speed=t.speed)
                )
            anomaly_dataset_dict = {}
            for anomaly_type, anomaly_trajectories in anomaly_trajectories_dict.items():
                logger.info(
                    "Tokenizing %s anomaly trajectories for %s",
                    anomaly_type,
                    model_name,
                )
                tokenized_anomalies = []
                for t in anomaly_trajectories:
                    time = timestamp_converter(t)
                    tokenized_anomalies.append(
                        TokenTrajectory(path=t.path, time=time, speed=t.speed)
                    )
                anomaly_dataset_dict[anomaly_type] = TokenTrajectoryDataset(
                    tokenized_anomalies + tokenized_test,
                    [1] * len(tokenized_anomalies) + [0] * len(tokenized_test),
                )
            return (
                TokenTrajectoryDataset(tokenized_train, [0] * len(tokenized_train)),
                None,
                anomaly_dataset_dict,
            )

        case "mst_oatd":
            logger.info("Tokenizing training trajectories for %s", model_name)
            tokenized_train = []
            for i, t in enumerate(train):
                time = timestamp_converter.convert_to_time(t)
                tau = timestamp_converter.convert_to_tau(t.timestamps)
                tokenized_train.append(TokenTrajectory(path=t.path, time=time, tau=tau))
                if i % 10000 == 0:  # Log every 10000 trajectories
                    logger.info(
                        "Tokenized trajectory %d/%d",
                        i,
                        len(train),
                    )
            logger.info("Tokenizing test trajectories for %s", model_name)
            tokenized_test = []
            for t in test:
                time = timestamp_converter.convert_to_time(t)
                tau = timestamp_converter.convert_to_tau(t.timestamps)
                tokenized_test.append(TokenTrajectory(path=t.path, time=time, tau=tau))
            anomaly_dataset_dict = {}
            for anomaly_type, anomaly_trajectories in anomaly_trajectories_dict.items():
                logger.info(
                    "Tokenizing %s anomaly trajectories for %s",
                    anomaly_type,
                    model_name,
                )
                tokenized_anomalies = []
                for t in anomaly_trajectories:
                    time = timestamp_converter.convert_to_time(t)
                    tau = timestamp_converter.convert_to_tau(t.timestamps)
                    tokenized_anomalies.append(
                        TokenTrajectory(path=t.path, time=time, tau=tau)
                    )
                anomaly_dataset_dict[anomaly_type] = TokenTrajectoryDataset(
                    tokenized_anomalies + tokenized_test,
                    [1] * len(tokenized_anomalies) + [0] * len(tokenized_test),
                )
            return (
                TokenTrajectoryDataset(tokenized_train, [0] * len(tokenized_train)),
                None,
                anomaly_dataset_dict,
            )

        case "gmvsae" | "vsae" | "causal_tad":
            logger.info("Tokenizing training trajectories for %s", model_name)
            tokenized_train = []
            for i, t in enumerate(train):
                tokenized_train.append(TokenTrajectory(path=t.path))
                if i % 10000 == 0:  # Log every 10000 trajectories
                    logger.info(
                        "Tokenized trajectory %d with path length %d",
                        i,
                        len(train),
                    )
            logger.info("Tokenizing test trajectories for %s", model_name)
            tokenized_test = []
            for t in test:
                tokenized_test.append(TokenTrajectory(path=t.path))
            anomaly_dataset_dict = {}
            for anomaly_type, anomaly_trajectories in anomaly_trajectories_dict.items():
                logger.info(
                    "Tokenizing %s anomaly trajectories for %s",
                    anomaly_type,
                    model_name,
                )
                tokenized_anomalies = []
                for t in anomaly_trajectories:
                    tokenized_anomalies.append(TokenTrajectory(path=t.path))
                anomaly_dataset_dict[anomaly_type] = TokenTrajectoryDataset(
                    tokenized_anomalies + tokenized_test,
                    [1] * len(tokenized_anomalies) + [0] * len(tokenized_test),
                )
            return (
                TokenTrajectoryDataset(tokenized_train, [0] * len(tokenized_train)),
                None,
                anomaly_dataset_dict,
            )
        case "fotraj":
            logger.info("Tokenizing training trajectories for %s", model_name)
            tokenized_train = []
            for i, t in enumerate(train):
                time_tuple = timestamp_converter.get_time_tuple(t)
                tokenized_train.append(
                    TokenTrajectory(path=t.path, time_tuple=time_tuple)
                )
                if i % 10000 == 0:  # Log every 10000 trajectories
                    logger.info(
                        "Tokenized trajectory %d/%d",
                        i,
                        len(train),
                    )
            del train
            logger.info("Tokenizing validation trajectories for %s", model_name)
            tokenized_val = []
            for t in val:
                time_tuple = timestamp_converter.get_time_tuple(t)
                tokenized_val.append(
                    TokenTrajectory(path=t.path, time_tuple=time_tuple)
                )
            del val
            logger.info("Tokenizing test trajectories for %s", model_name)
            tokenized_test = []
            for t in test:
                time_tuple = timestamp_converter.get_time_tuple(t)
                tokenized_test.append(
                    TokenTrajectory(path=t.path, time_tuple=time_tuple)
                )
            anomaly_dataset_dict = {}
            for anomaly_type, anomaly_trajectories in anomaly_trajectories_dict.items():
                logger.info(
                    "Tokenizing %s anomaly trajectories for %s",
                    anomaly_type,
                    model_name,
                )
                tokenized_anomalies = []
                for t in anomaly_trajectories:
                    time_tuple = timestamp_converter.get_time_tuple(t)
                    tokenized_anomalies.append(
                        TokenTrajectory(path=t.path, time_tuple=time_tuple)
                    )
                anomaly_dataset_dict[anomaly_type] = GraphTrajectoryDataset(
                    args,
                    tokenized_anomalies + tokenized_test,
                    [1] * len(tokenized_anomalies) + [0] * len(tokenized_test),
                )
            del test
            del anomaly_trajectories_dict
            return (
                GraphTrajectoryDataset(
                    args, tokenized_train, [0] * len(tokenized_train)
                ),
                GraphTrajectoryDataset(args, tokenized_val, [0] * len(tokenized_val)),
                anomaly_dataset_dict,
            )
        case "traj_mllm":
            # MLLM runner: raw Trajectory objects (no tokenization).
            logger.info(
                "Building raw TrajectoryDataset for %s (no tokenization)", model_name
            )
            anomaly_dataset_dict = {}
            for anomaly_type, anomaly_trajectories in anomaly_trajectories_dict.items():
                anomaly_dataset_dict[anomaly_type] = TrajectoryDataset(
                    anomaly_trajectories + test,
                    [1] * len(anomaly_trajectories) + [0] * len(test),
                )
            return (
                TrajectoryDataset(train, [0] * len(train)),
                None,
                anomaly_dataset_dict,
            )
        case _:
            raise ValueError(f"No matched dataset for model name : {model_name}!")


def get_test_dataset_by_model(
    args: Namespace,
    model_name: str,
    test: list[Trajectory],
    anomaly_trajectories_dict: dict[str, list[Trajectory]],
    timestamp_converter: TimestampConverter,
) -> dict[str, TokenTrajectoryDataset] | dict[str, GraphTrajectoryDataset]:
    """
    Returns the appropriate dataset based on the provided model name.
    Args:
        model_name (str): The name of the model (args -m option) for which to get the dataset.
        train (list[Trajectory]): List of training trajectories.
        test (list[Trajectory]): List of testing trajectories.
        anomaly_trajectories (list[Trajectory]): List of anomaly trajectories.
        timestamp_converter (TimestampConverter): Converter for timestamps to tokens.
    Returns:
        tuple[TrajectoryDataset, TrajectoryDataset]: A tuple containing:
            - Training dataset as TrajectoryDataset.
            - Testing dataset as TrajectoryDataset.
    Raises:
        ValueError: If the model name does not match any known dataset type.
    """
    logger.info("Get dataset for model: %s", model_name)
    match model_name:
        case "deep_tea":
            logger.info("Tokenizing test trajectories for %s", model_name)
            tokenized_test = []
            for t in test:
                time = timestamp_converter(t)
                tokenized_test.append(
                    TokenTrajectory(path=t.path, time=time, speed=t.speed)
                )
            anomaly_dataset_dict = {}
            for anomaly_type, anomaly_trajectories in anomaly_trajectories_dict.items():
                logger.info(
                    "Tokenizing %s anomaly trajectories for %s",
                    anomaly_type,
                    model_name,
                )
                tokenized_anomalies = []
                for t in anomaly_trajectories:
                    time = timestamp_converter(t)
                    tokenized_anomalies.append(
                        TokenTrajectory(path=t.path, time=time, speed=t.speed)
                    )
                anomaly_dataset_dict[anomaly_type] = TokenTrajectoryDataset(
                    tokenized_anomalies + tokenized_test,
                    [1] * len(tokenized_anomalies) + [0] * len(tokenized_test),
                )
            return anomaly_dataset_dict

        case "mst_oatd":
            logger.info("Tokenizing test trajectories for %s", model_name)
            tokenized_test = []
            for t in test:
                time = timestamp_converter.convert_to_time(t)
                tau = timestamp_converter.convert_to_tau(t.timestamps)
                tokenized_test.append(TokenTrajectory(path=t.path, time=time, tau=tau))
            anomaly_dataset_dict = {}
            for anomaly_type, anomaly_trajectories in anomaly_trajectories_dict.items():
                logger.info(
                    "Tokenizing %s anomaly trajectories for %s",
                    anomaly_type,
                    model_name,
                )
                tokenized_anomalies = []
                for t in anomaly_trajectories:
                    time = timestamp_converter.convert_to_time(t)
                    tau = timestamp_converter.convert_to_tau(t.timestamps)
                    tokenized_anomalies.append(
                        TokenTrajectory(path=t.path, time=time, tau=tau)
                    )
                anomaly_dataset_dict[anomaly_type] = TokenTrajectoryDataset(
                    tokenized_anomalies + tokenized_test,
                    [1] * len(tokenized_anomalies) + [0] * len(tokenized_test),
                )
            return anomaly_dataset_dict

        case "gmvsae" | "vsae" | "causal_tad":
            logger.info("Tokenizing test trajectories for %s", model_name)
            tokenized_test = []
            for t in test:
                tokenized_test.append(TokenTrajectory(path=t.path))
            anomaly_dataset_dict = {}
            for anomaly_type, anomaly_trajectories in anomaly_trajectories_dict.items():
                logger.info(
                    "Tokenizing %s anomaly trajectories for %s",
                    anomaly_type,
                    model_name,
                )
                tokenized_anomalies = []
                for t in anomaly_trajectories:
                    tokenized_anomalies.append(TokenTrajectory(path=t.path))
                anomaly_dataset_dict[anomaly_type] = TokenTrajectoryDataset(
                    tokenized_anomalies + tokenized_test,
                    [1] * len(tokenized_anomalies) + [0] * len(tokenized_test),
                )
            return anomaly_dataset_dict
        case "fotraj":
            logger.info("Tokenizing test trajectories for %s", model_name)
            tokenized_test = []
            for t in test:
                time_tuple = timestamp_converter.get_time_tuple(t)
                tokenized_test.append(
                    TokenTrajectory(path=t.path, time_tuple=time_tuple)
                )
            anomaly_dataset_dict = {}
            for anomaly_type, anomaly_trajectories in anomaly_trajectories_dict.items():
                logger.info(
                    "Tokenizing %s anomaly trajectories for %s",
                    anomaly_type,
                    model_name,
                )
                tokenized_anomalies = []
                for t in anomaly_trajectories:
                    time_tuple = timestamp_converter.get_time_tuple(t)
                    tokenized_anomalies.append(
                        TokenTrajectory(path=t.path, time_tuple=time_tuple)
                    )
                anomaly_dataset_dict[anomaly_type] = GraphTrajectoryDataset(
                    args,
                    tokenized_anomalies + tokenized_test,
                    [1] * len(tokenized_anomalies) + [0] * len(tokenized_test),
                )
            return anomaly_dataset_dict
        case "traj_mllm":
            # MLLM runner needs raw Trajectory objects (not tokenized) so it
            # can derive GPS coordinates from edge IDs.
            logger.info(
                "Building raw TrajectoryDataset for %s (no tokenization)", model_name
            )
            anomaly_dataset_dict = {}
            for anomaly_type, anomaly_trajectories in anomaly_trajectories_dict.items():
                anomaly_dataset_dict[anomaly_type] = TrajectoryDataset(
                    anomaly_trajectories + test,
                    [1] * len(anomaly_trajectories) + [0] * len(test),
                )
            return anomaly_dataset_dict
        case _:
            raise ValueError(f"No matched dataset for model name : {model_name}!")


def get_train_dataloader(
    dataset: TokenTrajectoryDataset | GraphTrajectoryDataset | TrajectoryDataset,
    batch_size: int,
    num_workers: int,
    collate_fn: Callable,
    generator: Generator,
) -> DataLoader:
    logger.info("Creating DataLoader for training dataset")
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        generator=generator,
        pin_memory=True,
        drop_last=True,
    )


def get_val_dataloader(
    dataset: TokenTrajectoryDataset | GraphTrajectoryDataset | TrajectoryDataset,
    batch_size: int,
    num_workers: int,
    collate_fn: Callable,
    generator: Generator,
) -> DataLoader:
    logger.info("Creating DataLoader for validation dataset")
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        generator=generator,
        pin_memory=True,
        drop_last=False,
    )


def get_test_dataloader_dict(
    anomaly_dataset_dict: (
        dict[str, TokenTrajectoryDataset]
        | dict[str, GraphTrajectoryDataset]
        | dict[str, TrajectoryDataset]
    ),
    batch_size: int,
    num_workers: int,
    collate_fn: Callable,
    generator: Generator,
):
    """
    Get DataLoaders for each anomaly dataset.
    Yields:
        tuple: A tuple containing the anomaly type and its corresponding DataLoader.
    """
    logger.info("Creating DataLoaders for anomaly datasets")
    anomaly_dataloaders_dict = {}
    for anomaly_type, anomaly_dataset in anomaly_dataset_dict.items():
        logger.info("Creating DataLoader for anomaly dataset: %s", anomaly_type)
        anomaly_dataloaders_dict[anomaly_type] = DataLoader(
            dataset=anomaly_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            generator=generator,
            pin_memory=True,
            drop_last=True,
        )
    return anomaly_dataloaders_dict


def get_train_runner_dataloader_pairs(
    train: list[Trajectory],
    val: list[Trajectory],
    test: list[Trajectory],
    anomaly_trajectories_dict: dict[str, list[Trajectory]],
    timestamp_converter: TimestampConverter,
    generator: Generator,
    args: Namespace,
):
    for model_name in args.models:
        runner = get_runner(model_name)
        collate_fn = get_collate_fn(model_name)
        train_dataset, val_dataset, anomaly_dataset_dict = get_train_dataset_by_model(
            args=args,
            model_name=model_name,
            train=train,
            val=val,
            test=test,
            anomaly_trajectories_dict=anomaly_trajectories_dict,
            timestamp_converter=timestamp_converter,
        )
        train_dataloader = get_train_dataloader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            generator=generator,
        )
        test_dataloader_dict = get_test_dataloader_dict(
            anomaly_dataset_dict=anomaly_dataset_dict,
            batch_size=args.test_batch_size,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            generator=generator,
        )

        val_dataloader = None
        if val_dataset is not None:
            val_dataloader = get_val_dataloader(
                dataset=val_dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                collate_fn=collate_fn,
                generator=generator,
            )

        yield runner, train_dataloader, val_dataloader, test_dataloader_dict


def get_test_runner_dataloader_pairs(
    test: list[Trajectory],
    anomaly_trajectories_dict: dict[str, list[Trajectory]],
    timestamp_converter: TimestampConverter,
    generator: Generator,
    args: Namespace,
):
    for model_name in args.models:
        runner = get_runner(model_name)
        collate_fn = get_collate_fn(model_name)
        anomaly_dataset_dict = get_test_dataset_by_model(
            args=args,
            model_name=model_name,
            test=test,
            anomaly_trajectories_dict=anomaly_trajectories_dict,
            timestamp_converter=timestamp_converter,
        )
        test_dataloader_dict = get_test_dataloader_dict(
            anomaly_dataset_dict=anomaly_dataset_dict,
            batch_size=args.test_batch_size,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            generator=generator,
        )
        yield runner, test_dataloader_dict
