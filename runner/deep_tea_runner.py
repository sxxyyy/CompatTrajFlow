import ast
import gc
import logging
import os
from argparse import Namespace
from os.path import exists

import geopandas as gpd
import numpy as np
import torch
from geopy.distance import geodesic
from numpy import ndarray
from torch import Generator, GradScaler, Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader

from model.deep_tea.model import DeepTea
from runner.abstract_runner import AbstractRunner

logger = logging.getLogger(__name__)


class DeepTeaRunner(AbstractRunner):
    """Runner for DeepTea model."""

    def __init__(self, args: Namespace, generator: Generator) -> None:
        """
        Initializes DeepTeaRunner, model, optimizer, and speed map parameters.

        Args:
            args (Namespace): Arguments for the runner.
            generator (Generator): Torch random generator.
        """
        logger.info("Initializing DeepTeaRunner")
        super().__init__(args, generator)
        self.name = "DeepTea"
        self.speed_map_time_interval = args.deeptea_speed_map_time_interval
        self.use_grid_tokens = getattr(args, "use_grid_tokens", False)
        self.edge_mapping = getattr(args, "edge_mapping", {})
        self.gdf = self.get_geodataframe()
        self.min_lon, self.min_lat, self.max_lon, self.max_lat = self.gdf.total_bounds
        if self.use_grid_tokens:
            self.map_size = (args.grid_num_rows, args.grid_num_cols)
            logger.info(
                "Map size initialized from grid mapper: %s rows x %s columns",
                self.map_size[0],
                self.map_size[1],
            )
        else:
            if args.location == "xian":
                self.grid_size = 1.0  # in kilometers
            elif args.location == "porto":
                self.grid_size = 0.1  # in kilometers
            lat_distance = geodesic(
                (self.min_lat, self.min_lon), (self.max_lat, self.min_lon)
            ).kilometers
            lon_distance = geodesic(
                (self.min_lat, self.min_lon), (self.min_lat, self.max_lon)
            ).kilometers
            self.map_size = (
                int(lat_distance / self.grid_size),
                int(lon_distance / self.grid_size),
            )
            logger.info(
                "Map size initialized from geodesic bounds: %s rows x %s columns",
                self.map_size[0],
                self.map_size[1],
            )

        self.model = DeepTea(self.map_size, args)
        self.optimizer = AdamW(self.model.parameters(), lr=args.deeptea_learning_rate)
        self.scaler = GradScaler(args.device.type, enabled=args.use_amp)

    def get_geodataframe(self) -> gpd.GeoDataFrame:
        """
        Loads GeoDataFrame for the specified location.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame containing edges for the specified location.

        Raises:
            ValueError: If the location is not supported.
        """
        if self.location == "xian":
            gdf = gpd.read_file(f"data/{self.location}/raw/edges.shp")
        elif self.location == "porto":
            gdf = gpd.read_file(f"data/{self.location}/raw/edges.shp")
        else:
            raise ValueError("Unsupported location for geodataframe retrieval.")
        return gdf

    def get_grid_location(
        self, lat: float, lon: float, map_size: tuple
    ) -> tuple[int, int]:
        """
        Converts latitude and longitude to grid coordinates.

        Args:
            lat (float): Latitude value.
            lon (float): Longitude value.
            map_size (tuple): Size of the map as (height, width).

        Returns:
            tuple[int, int]: Grid location as (row, column).

        Raises:
            AssertionError: If latitude or longitude is out of bounds.
        """
        assert self.max_lat >= lat >= self.min_lat, (
            f"Latitude {lat} is out of bounds ({self.min_lat}, {self.max_lat})"
        )
        assert self.max_lon >= lon >= self.min_lon, (
            f"Longitude {lon} is out of bounds ({self.min_lon}, {self.max_lon})"
        )
        row = int((lat - self.min_lat) / (self.max_lat - self.min_lat) * map_size[0])
        col = int((lon - self.min_lon) / (self.max_lon - self.min_lon) * map_size[1])
        row = min(max(row, 0), map_size[0] - 1)
        col = min(max(col, 0), map_size[1] - 1)
        return row, col

    def init_empty_map(self) -> ndarray:
        """
        Creates an empty map array of zeros.

        Returns:
            ndarray: An empty map of the specified size filled with zeros.
        """
        return np.zeros(self.map_size, dtype=int)

    def get_edge_location_dict(self) -> dict:
        """
        Creates a dictionary mapping edge or grid IDs to speed-map locations.

        Returns:
            dict: A dictionary where keys are token IDs and values are tuples of
            grid coordinates (row, column).
        """
        if self.use_grid_tokens:
            rows, cols = self.map_size
            return {grid_id: divmod(grid_id, cols) for grid_id in range(rows * cols)}

        edge_location_dict = {}
        for _, row in self.gdf.iterrows():
            lon = row["geometry"].centroid.x
            lat = row["geometry"].centroid.y
            edge_location_dict[row["fid"]] = self.get_grid_location(
                lat, lon, self.map_size
            )
        return edge_location_dict

    def create_edge_speed_list_dict(
        self, dataloader: DataLoader
    ) -> dict[int, list[float]]:
        """
        Creates a dictionary mapping edge IDs to lists of speed records.
        Args:
            dataloader (DataLoader): DataLoader for the dataset.
        Returns:
            dict: A dictionary where keys are edge IDs and values are lists of speed records.
        """
        edge_speed_records_dict = {}
        for t, _ in dataloader.dataset:
            for edge_fid, edge_speed in zip(t.path, t.speed):
                if edge_fid not in edge_speed_records_dict:
                    edge_speed_records_dict[edge_fid] = []
                edge_speed_records_dict[edge_fid].append(edge_speed)
        return edge_speed_records_dict

    def create_edge_speed_limits_dict(self) -> dict[int, int]:
        """
        Creates a dictionary mapping edge/grid token IDs to maximum speed limits.
        Returns:
            dict: A dictionary where keys are token IDs and values are speed limits.
        """
        edge_speed_limits = {}
        for edge, speed_limits in self.gdf[["fid", "maxspeed"]].values:
            if isinstance(speed_limits, str):
                speed_limits = ast.literal_eval(speed_limits)
                if isinstance(speed_limits, list):
                    speed_limits = [int(i) for i in speed_limits]
                    speed_limits = max(speed_limits)
                if self.use_grid_tokens:
                    if edge not in self.edge_mapping:
                        continue
                    token_id = self.edge_mapping[edge]
                else:
                    token_id = edge
                edge_speed_limits[token_id] = max(
                    speed_limits, edge_speed_limits.get(token_id, 0)
                )
        return edge_speed_limits

    def generate_speed_map(self, dataloader: DataLoader):
        """
        Generates a speed map from the dataloader.

        Args:
            dataloader (DataLoader): DataLoader for the dataset.

        Returns:
            ndarray: Speed map array.
        """
        green = 3
        yellow = 2
        red = 1
        dataset = dataloader.dataset
        edge_location_dict = self.get_edge_location_dict()
        interval_in_seconds = self.speed_map_time_interval * 60

        time_trajectories_dict = {
            time: [] for time in range(int(24 * 60 * 60 / interval_in_seconds))
        }
        for t, _ in dataset:
            appear_list = []
            for edge_fid, time, edge_speed in zip(t.path, t.time, t.speed):
                if time >= 24 * 60 * 60:
                    time -= 24 * 60 * 60  # Convert to workday
                time //= (
                    interval_in_seconds  # Calculate which bucket the time falls into
                )
                if time not in appear_list:
                    appear_list.append(time)
            for time in appear_list:
                time_trajectories_dict[time].append(t)

        edge_speed_list_dict = self.create_edge_speed_list_dict(dataloader)

        edge_speed_limits_dict = self.create_edge_speed_limits_dict()

        edge_speed_thresholds = {}

        for edge_fid, speed_list in edge_speed_list_dict.items():
            if len(speed_list) < 2:
                if edge_fid not in edge_speed_limits_dict:
                    continue

                max_speed = edge_speed_limits_dict.get(edge_fid)
                if max_speed:
                    separator = max_speed / 3
                    edge_speed_thresholds[edge_fid] = {
                        "red": separator,
                        "yellow": separator * 2,
                        "green": max_speed,
                    }
            else:
                speed_list.sort()
                max_speed, min_speed = speed_list[-1], speed_list[0]
                sep = (max_speed - min_speed) / 3
                edge_speed_thresholds[edge_fid] = {
                    "red": sep,
                    "yellow": sep * 2,
                    "green": max_speed,
                }

        map_series = []
        for _, trajectories in time_trajectories_dict.items():
            speed_map = self.init_empty_map()
            for trajectory in trajectories:
                for edge_fid, edge_speed in zip(trajectory.path, trajectory.speed):
                    if edge_fid in edge_speed_thresholds:
                        edge_location = edge_location_dict[edge_fid]
                        if edge_speed < edge_speed_thresholds[edge_fid]["red"]:
                            speed_map[edge_location] = red
                        elif edge_speed < edge_speed_thresholds[edge_fid]["yellow"]:
                            speed_map[edge_location] = yellow
                        elif edge_speed < edge_speed_thresholds[edge_fid]["green"]:
                            speed_map[edge_location] = green
            map_series.append(speed_map)
        speed_map_array = np.array(map_series)
        logger.info("Speed map created with shape: %s", speed_map_array.shape)
        uniques, counts = np.unique(speed_map_array, return_counts=True)
        logger.info("Speed map unique values: %s", dict(zip(uniques, counts)))
        return speed_map_array

    def get_speed_map(self, dataloader: DataLoader) -> Tensor:
        """
        Retrieves or generates a speed map tensor for a given status and dataloader.

        If a speed map corresponding to the specified status already exists, it is loaded from disk.
        Otherwise, a new speed map is generated using the provided dataloader, saved to disk, and returned.

        Args:
            status (str): The status identifier used to check for an existing speed map or to save a new one.
            dataloader (DataLoader): The dataloader used to generate a new speed map if one does not exist.

        Returns:
            Tensor: The speed map tensor associated with the given status.
        """
        speed_map = self.generate_speed_map(dataloader)
        speed_map = torch.tensor(speed_map, dtype=torch.float32)
        speed_map = speed_map.unsqueeze(0)  # Add batch dimensions
        speed_map = speed_map.unsqueeze(2)  # Add channel dimension
        logger.info("Speed map shape: %s with batch and channel dims", speed_map.shape)
        return speed_map

    def train(self, dataloader: DataLoader):
        """
        Trains DeepTea model using the dataloader and speed map.

        Args:
            dataloader (DataLoader): DataLoader for training data.

        Returns:
            float: Average loss for this epoch.
        """
        epoch_loss = 0.0
        speed_map = self.get_speed_map(dataloader)
        for batch_index, mini_batch in enumerate(dataloader):
            mini_batch = map(lambda x: x.to(self.device, non_blocking=True), mini_batch)
            self.optimizer.zero_grad()
            with torch.autocast(
                device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp
            ):
                loss: Tensor = self.model(speed_map, tuple(mini_batch))
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            epoch_loss += loss.item()
            self.global_step += 1
            self.total_train_loss += loss.item()
            global_avg = self.total_train_loss / self.global_step
            self.log_batch_loss(
                batch_index=batch_index,
                num_batch=len(dataloader),
                loss=global_avg,
            )
        epoch_avg = epoch_loss / len(dataloader)
        return epoch_avg

    def test(self, dataloader: DataLoader) -> tuple[ndarray, ndarray]:
        """
        Evaluates DeepTea model and returns true labels and predictions.

        Args:
            dataloader (DataLoader): DataLoader for testing data.

        Returns:
            tuple[ndarray, ndarray]: Tuple containing true labels and predicted scores.
        """
        y_true = []
        y_pred = []
        speed_map = self.get_speed_map(dataloader)
        with torch.no_grad():
            for i, mini_batch in enumerate(dataloader):
                mini_batch = map(
                    lambda x: x.to(self.device, non_blocking=True), mini_batch
                )
                labels, pred = self.model.compute_anomaly_scores(
                    speed_map, tuple(mini_batch)
                )
                y_true.append(labels)
                y_pred.append(pred)
                self.log_test_progress(batch_index=i, num_batch=len(dataloader))
        y_true = torch.cat(y_true).cpu().numpy()
        y_pred = torch.cat(y_pred).cpu().numpy()
        return y_true, y_pred

    def create_checkpoint_dir(self):
        """
        Creates the checkpoint directory and saves model state.
        """
        checkpoint_path = f"{self.checkpoint_path}/{self.location}/deep_tea/"
        os.makedirs(checkpoint_path, exist_ok=True)
        logger.info("Creating checkpoint directory for DeepTea at %s", checkpoint_path)

    def save_checkpoint(self):
        """
        Saves the model checkpoint to disk.
        """
        checkpoint_path = f"{self.checkpoint_path}/{self.location}/deep_tea/model.pt"
        torch.save(self.model.state_dict(), checkpoint_path)
        logger.info("Model checkpoint saved: %s", checkpoint_path)

    def load_checkpoint(self):
        """
        Loads the model checkpoint from disk.
        """
        checkpoint_path = f"{self.checkpoint_path}/{self.location}/deep_tea/model.pt"
        self.model.load_state_dict(torch.load(checkpoint_path, self.device))
        logger.info("Model checkpoint loaded: %s", checkpoint_path)

    def is_checkpoint_exists(self) -> bool:
        """
        Checks if the model checkpoint exists.

        Returns:
            bool: True if the checkpoint exists, False otherwise.
        """
        checkpoint_path = f"{self.checkpoint_path}/{self.location}/deep_tea/model.pt"
        return exists(checkpoint_path)

    def free_vram(self):
        del self.model
        del self.optimizer
        if self.use_amp:
            del self.scaler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("VRAM freed for next training or testing")

    def __str__(self) -> str:
        return self.__class__.__name__ + "_" + super().__str__()
