"Preprocessing module"

import logging
import os
import pickle
from datetime import datetime
from os.path import exists

from numpy.random import Generator as RNG
from torch import Generator

from formatter.abstract_formatter import AbstractFormatter
from formatter.porto_formatter import PortoFormatter
from formatter.xian_formatter import XianFormatter
from utils.fmm_docker import FmmDocker
from utils.road_network import RoadNetwork
from utils.trajectory import Trajectory

logger = logging.getLogger(__name__)


class Preprocessing:
    """Preprocessing class"""

    def __init__(
        self,
        location: str,
        trajectory_len_threshold: tuple[int, int],
        road_network: RoadNetwork,
        data_split_ratio: tuple[float, float, float],
        generator: Generator,
        rng: RNG,
    ):
        """
        Initializes the preprocessing object for a specific location, setting up file paths, road network, timestamp converter, and data formatter.
        Args:
            location (str): The dataset location (`"porto"` or `"xian"`).
            road_network (RoadNetwork): An instance representing the road network for the location.
        Raises:
            ValueError: If the specified location is not supported.
        Side Effects:
            - Logs the start of preprocessing.
            - Sets up the appropriate data formatter based on the location.
        """

        logger.info("Start preprocessing...")
        self.processed_file_dir = f"data/{location}/processed"
        self.raw_data_path = f"data/{location}/raw"

        self.road_network = road_network
        self.formatter = self.get_formatter(location, trajectory_len_threshold)
        assert sum(data_split_ratio) == 1, (
            "Data split ratio must sum to 1, got: "
            f"{data_split_ratio} with sum {sum(data_split_ratio)}"
        )
        self.threshold = trajectory_len_threshold
        self.data_split_ratio = data_split_ratio
        self.generator = generator
        self.rng = rng
        self.location = location
        self._adjacency_dict: dict[int, list[int]] | None = None

    def get_formatter(
        self, location: str, threshold: tuple[int, int]
    ) -> AbstractFormatter:
        """
        Get the formatter instance based on the specified location.
        Args:
            location (str): The dataset location (`"porto"` or `"xian"`).
            threshold (tuple[int, int]): The minimum and maximum length thresholds for trajectories.
        Returns:
            AbstractFormatter: An instance of the formatter for the specified location.
        Raises:
            ValueError: If the specified location is not supported.
        """
        match location:
            case "xian":
                return XianFormatter(self.raw_data_path, threshold)
            case "porto":
                return PortoFormatter(self.raw_data_path, threshold)
            case _:
                raise ValueError(f"Unsupported location: {location}")

    def __call__(self) -> tuple[list[Trajectory], list[Trajectory], list[Trajectory]]:

        if (
            exists(f"{self.processed_file_dir}/train_trajectories.pkl")
            and exists(f"{self.processed_file_dir}/val_trajectories.pkl")
            and exists(f"{self.processed_file_dir}/test_trajectories.pkl")
        ):
            logger.info("Load train, val and test trajectories from processed files")
            train_trajectories = self.load_trajectories_from_file(
                "train_trajectories.pkl"
            )
            val_trajectories = self.load_trajectories_from_file("val_trajectories.pkl")
            test_trajectories = self.load_trajectories_from_file(
                "test_trajectories.pkl"
            )
        elif exists(f"{self.processed_file_dir}/all_trajectories.pkl"):
            all_trajectories = self.load_trajectories_from_file("all_trajectories.pkl")
            logger.info("Load trajectories from processed all_trajectories.pkl file")
            logger.info("Split trajectories into train, val and test sets")
            train_trajectories, val_trajectories, test_trajectories = (
                self.split_by_time(all_trajectories)
            )
            logger.info("Save train, val and test trajectories to processed files")
            self.save_trajectories_to_file(train_trajectories, "train_trajectories.pkl")
            self.save_trajectories_to_file(val_trajectories, "val_trajectories.pkl")
            self.save_trajectories_to_file(test_trajectories, "test_trajectories.pkl")
        else:
            logger.info("Generate road network...")
            self.road_network()
            if not exists(f"{self.processed_file_dir}/mr.txt"):
                logger.info("Unzip and format data...")
                self.formatter()
                self.create_processed_file_dir()
                FmmDocker()(
                    self.processed_file_dir,
                    self.road_network.edges_file,
                    self.formatter.filtered_points_file,
                )
            trajectories = self.form_trajectories()
            self.save_trajectories_to_file(trajectories, "all_trajectories.pkl")
            train_trajectories, val_trajectories, test_trajectories = (
                self.split_by_time(trajectories)
            )
            self.save_trajectories_to_file(train_trajectories, "train_trajectories.pkl")
            self.save_trajectories_to_file(val_trajectories, "val_trajectories.pkl")
            self.save_trajectories_to_file(test_trajectories, "test_trajectories.pkl")
            logger.info("Save train, val and test trajectories to processed files")

        return train_trajectories, val_trajectories, test_trajectories

    def split_by_time(self, trajectories: list[Trajectory]):
        start_date = float("inf")
        end_date = -1
        for t in trajectories:
            time = t.get_start_time()
            start_date = min(start_date, time)
            end_date = max(end_date, time)
        logger.info("Dataset Start date: %s, End date: %s", start_date, end_date)
        gap = end_date - start_date
        train_end_date = start_date + gap * self.data_split_ratio[0]
        val_end_date = train_end_date + gap * self.data_split_ratio[1]
        train_trajectories = []
        val_trajectories = []
        test_trajectories = []
        for t in trajectories:
            time = t.get_start_time()
            if time < train_end_date:
                train_trajectories.append(t)
            elif time < val_end_date:
                val_trajectories.append(t)
            else:
                test_trajectories.append(t)
        logger.info(
            "Split trajectories into train: %d, val: %d, test: %d",
            len(train_trajectories),
            len(val_trajectories),
            len(test_trajectories),
        )
        return train_trajectories, val_trajectories, test_trajectories

    def save_trajectories_to_file(
        self,
        trajectories: list[Trajectory],
        file_name: str = "all_trajectories.pkl",
    ):
        """
        Save the processed trajectories to a file in the processed file directory.
        Args:
            trajectories (list[Trajectory]): The list of trajectories to save.
            file_name (str): The name of the file to save the trajectories to. Defaults to "all_trajectories.pkl".
        """
        path = f"{self.processed_file_dir}/{file_name}"
        logger.info("Save trajectories to %s", path)
        with open(path, mode="wb") as f:
            pickle.dump(trajectories, f)

    def create_processed_file_dir(self):
        """create processed file dir if not exist."""
        if exists(self.processed_file_dir):
            logger.info("Processed data's file path exist")
        else:
            logger.info("Processed data's file path doesn't exist")
            logger.info(
                "Create processed data's file path at %s", self.processed_file_dir
            )
            os.makedirs(self.processed_file_dir)

    def load_trajectories_from_file(self, file_name: str) -> list[Trajectory]:
        """Read trajectories from processed .pkl file

        Returns:
            list[Trajectory]: List of loaded trajectories
        """
        path = f"{self.processed_file_dir}/{file_name}"
        logger.info("Load trajectories from %s", path)
        with open(path, mode="rb") as f:
            trajectories = pickle.load(f)
            logger.info("Loaded %d trajectories from %s", len(trajectories), path)
            f.close()
        return trajectories

    def get_tid_timestamps_dict(self) -> dict[int, list[int]]:
        """
        Reads a filtered points file and constructs a dictionary mapping each TID (time series ID)
        to a list of its associated timestamps.
        Returns:
            dict[int, list[int]]: A dictionary where each key is a TID (int) and the value is a list of
            timestamps (int) corresponding to that TID.
        """

        tid_timeseries_dict = {}
        with open(self.formatter.filtered_points_file, mode="r", encoding="UTF-8") as f:
            f.readline()
            while line := f.readline():
                string_list = line.split(";")
                tid, timestamp = int(string_list[0]), int(float(string_list[-1]))
                if tid in tid_timeseries_dict:
                    tid_timeseries_dict[tid].append(timestamp)
                else:
                    tid_timeseries_dict[tid] = [timestamp]
            f.close()
        return tid_timeseries_dict

    def get_tid_properties(
        self,
    ):
        with open(f"{self.processed_file_dir}/mr.txt", mode="r", encoding="UTF-8") as f:
            f.readline()
            while line := f.readline():
                tid_str, opath_str, cpath_str, speed_str = line.strip().split(";")
                tid = int(tid_str)
                if not opath_str or not cpath_str:
                    continue
                opath = [int(x) for x in opath_str.split(",")]
                cpath = [int(x) for x in cpath_str.split(",")]
                speed = [float(x) for x in speed_str.split(",")]
                speed.append(speed[-1])
                yield tid, opath, cpath, speed

    def load_adjacency_dict(self) -> dict[int, list[int]]:
        """Load the road-network adjacency dictionary used to validate trajectories.

        The adjacency dictionary is generated by `RoadNetwork` and stored in
        `data/<location>/raw/adj_dict.pkl`. Each key is an edge id and each value is
        the list of edge ids that are adjacent to that edge in the road network.
        """

        if self._adjacency_dict is not None:
            return self._adjacency_dict

        path = self.road_network.cause_tad_required_adj_dict
        logger.info("Load road-network adjacency dictionary from %s", path)
        with open(path, mode="rb") as f:
            self._adjacency_dict = pickle.load(f)
        return self._adjacency_dict

    def is_adjacent_trajectory(self, path: list[int]) -> bool:
        """Check whether every consecutive pair of edges is adjacent in the graph.

        A trajectory is considered valid only if each edge can be followed by the next
        edge according to the road-network adjacency dictionary. This is a stricter
        filter than the FMM output and helps reject disconnected path fragments.
        """

        adjacency_dict = self.load_adjacency_dict()
        for current_edge, next_edge in zip(path[:-2], path[1:-1]):
            if next_edge not in adjacency_dict.get(current_edge, []):
                return False
        return True

    def form_trajectories(self) -> list[Trajectory]:
        """
        Constructs a list of Trajectory objects by processing raw path and timestamp data.
        This method retrieves trajectory ID and their opath and cpath,
        as well as their corresponding timestamps. For each trajectory, it performs upsampling
        and downsampling operations to adjust the path and timestamps, then creates a Trajectory
        object with the processed data. Progress is logged every 1000 trajectories.
        Returns:
            list[Trajectory]: A list of processed Trajectory objects.
        """

        logger.info("Build trajectories with time")
        tid_timestamps_dict = self.get_tid_timestamps_dict()
        trajectories = []
        total_trajectories = 0
        rejected_non_adjacent = 0
        for i, (tid, opath, cpath, speed) in enumerate(self.get_tid_properties()):
            timestamps = tid_timestamps_dict[tid]
            upsample_path, upsample_time, upsample_speed = self.upsample(
                opath, cpath, speed, timestamps
            )
            path, timestamps, speed = self.downsample(
                upsample_path, upsample_time, upsample_speed
            )

            if len(path) < self.threshold[0] or len(path) > self.threshold[1]:
                continue

            if not self.is_adjacent_trajectory(path):
                rejected_non_adjacent += 1
                continue

            correct_time = True
            for pre_time, next_time in zip(timestamps[:-1], timestamps[1:]):
                next_time = datetime.fromtimestamp(next_time)
                pre_time = datetime.fromtimestamp(pre_time)
                if next_time < pre_time:
                    correct_time = False
                    logger.warning(
                        "Error in trajectory %d: timestamp not increasing",
                        total_trajectories,
                    )
            if not correct_time:
                continue
            trajectories.append(Trajectory(path, timestamps, speed))
            if i % 1000 == 0:
                logger.info("Build trajectories: %d", i)
            total_trajectories += 1
        logger.info("Trajectories average length: %.2f", sum(len(t) for t in trajectories) / len(trajectories))
        logger.info("Rejected %d non-adjacent trajectories", rejected_non_adjacent)
        logger.info("Total trajectories built: %d", total_trajectories)

        return trajectories

    def make_inject_time(
        self, start_timestamp: int, end_timestamp: int, inject_path_len: int
    ) -> list[int]:
        """
        Generates a list of timestamps at which to inject events, evenly spaced between the given start and end timestamps.
        Args:
            start_timestamp (int): The starting timestamp (inclusive).
            end_timestamp (int): The ending timestamp (inclusive).
            inject_path_len (int): The number of injection points to generate.
        Returns:
            list[int]: A list of timestamps at which to inject events, evenly spaced between start_timestamp and end_timestamp.
        Example:
            >>> make_inject_time(0, 100, 4)
            [20, 40, 60, 80]
        """
        interval = int((end_timestamp - start_timestamp) / (inject_path_len + 1))
        inject_time = [
            start_timestamp + (i + 1) * interval for i in range(inject_path_len)
        ]

        return inject_time

    def upsample(
        self,
        opath: list[int],
        cpath: list[int],
        speed: list[float],
        timestamps: list[int],
    ) -> tuple[list[int], list[int], list[float]]:
        """
        Upsample a sequence of edge indices (`opath`) and their corresponding timestamps.
        This method aligns the opath with a cpath by inserting missing path indices from
        `cpath` into the upsampled path. For each inserted path index, it generates
        interpolated timestamps using `make_inject_time` and assume the speed remains same
        as the start index as it is calculated by average speed between two gps. The result is a new
        path, a timestamp sequence and a average speed sequence.
        Args:
            opath (list[int]): The opath indices to upsample to.
            cpath (list[int]): The cpath indices.
            speed (list[float]): The speed values corresponding to each index in `opath`.
            timestamps (list[int]): The timestamps corresponding to each index in `opath`.
        Returns:
            tuple[list[int], list[int], list[float]]:
                - upsample_path: The upsampled path sequence contains all traveled edges.
                - upsample_time: The corresponding timestamps, including interpolated values for inserted indices.
                - upsample_speed: The corresponding speed values for the upsampled path.
        """
        upsample_path = []
        upsample_speed = []
        upsample_time = []
        o_index = 0
        c_index = 0
        while c_index < len(cpath) and o_index < len(opath):
            if opath[o_index] == cpath[c_index]:
                upsample_path.append(opath[o_index])
                upsample_speed.append(speed[o_index])
                upsample_time.append(timestamps[o_index])
                o_index += 1
            else:
                c_index += 1
                if opath[o_index] == cpath[c_index]:
                    continue
                inject_path = []
                inject_speed = []
                while opath[o_index] != cpath[c_index]:
                    inject_path.append(cpath[c_index])
                    inject_speed.append(speed[o_index])
                    c_index += 1
                inject_time = self.make_inject_time(
                    timestamps[o_index - 1], timestamps[o_index], len(inject_path)
                )
                upsample_path.extend(inject_path)
                upsample_speed.extend(inject_speed)
                upsample_time.extend(inject_time)

        return upsample_path, upsample_time, upsample_speed

    def downsample(
        self,
        upsample_path: list[int],
        upsample_time: list[int],
        upsample_speed: list[float],
    ) -> tuple[list[int], list[int], list[float]]:
        """
        Downsample the given path, time, and speed sequences by removing consecutive duplicate edges.
        Args:
            upsample_path (list[int]): The list of edge identifiers representing the upsampled path.
            upsample_time (list[int]): The list of timestamps corresponding to each edge in the upsampled path.
            upsample_speed (list[float]): The list of speeds corresponding to each edge in the upsampled path.
        Returns:
            tuple[list[int], list[int], list[float]]:
                - downsampled_path: The downsampled list of edge identifiers with consecutive duplicates removed.
                - downsampled_time: The corresponding timestamps for the downsampled path.
                - downsampled_speed: The corresponding speeds for the downsampled path.
        """
        downsampled_path = [upsample_path[0]]
        downsampled_time = [upsample_time[0]]
        downsampled_speed = [upsample_speed[0]]
        for edge, time, speed in zip(
            upsample_path[1:-1], upsample_time[1:-1], upsample_speed[1:-1]
        ):
            if edge != downsampled_path[-1]:
                downsampled_path.append(edge)
                downsampled_time.append(time)
                downsampled_speed.append(speed)
        downsampled_path.append(upsample_path[-1])
        downsampled_time.append(upsample_time[-1])
        downsampled_speed.append(upsample_speed[-1])
        return downsampled_path, downsampled_time, downsampled_speed
