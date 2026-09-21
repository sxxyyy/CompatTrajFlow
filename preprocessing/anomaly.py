"""Anomaly generator module"""

import ast
import logging
import os
import pickle
import subprocess
from datetime import datetime
from multiprocessing import Pool
from os import mkdir
from os.path import exists
from zoneinfo import ZoneInfo

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
from networkx import MultiDiGraph

from utils.trajectory import Trajectory

logger = logging.getLogger(__name__)


def concatenate_times(
    start: Trajectory, diff: Trajectory, end: Trajectory | None = None
) -> Trajectory:
    """
    Concatenates the timestamps of two Trajectory objects, adjusting the timestamps
    of the second trajectory to follow the first one, and optionally appending
    the timestamps of a third trajectory.
    Args:
        start (Trajectory): The first trajectory object, ``[t_1, t_2 ...t_i]``.
        diff (Trajectory): The second trajectory object to be concatenated, ``[t_i, t_{i+1} ...t_{j}]``.
        end (Trajectory, optional): An optional third trajectory object to be appended ``[t_j, t_{j+1} ... t_{n}]``.
    Returns:
        Trajectory: A new Trajectory object with concatenated paths, speed and adjusted timestamps of length n.
    """
    start_timestamps = start.timestamps
    diff_timestamps = diff.timestamps
    gaps = []
    for i in range(1, len(diff_timestamps)):
        pre_time = diff_timestamps[i - 1]
        current_time = diff_timestamps[i]
        gap = current_time - pre_time
        gaps.append(gap)
    if end is not None:
        end_timestamps = end.timestamps
        for i in range(1, len(end_timestamps)):
            pre_time = end_timestamps[i - 1]
            current_time = end_timestamps[i]
            gap = current_time - pre_time
            gaps.append(gap)
    for gap in gaps:
        last_time = start_timestamps[-1]
        new_time = last_time + gap
        start_timestamps.append(new_time)
    if end is not None:
        return Trajectory(
            start.path[:-1] + diff.path[:-1] + end.path,
            start_timestamps,
            start.speed[:-1] + diff.speed[:-1] + end.speed,
        )

    return Trajectory(
        start.path[:-1] + diff.path,
        start_timestamps,
        start.speed[:-1] + diff.speed,
    )


def get_subset_slices_with_sd(
    trajectories: list[Trajectory], sd_pair: tuple[int, int]
) -> list[tuple[int, list[slice]]]:
    """
    Get slices of trajectories that match the source and destination node IDs.
    Args:
        trajectories (list[Trajectory]): List of Trajectory objects.
        sd_pair (tuple[int, int]): A tuple containing the source and destination node IDs.
    Returns:
        list[tuple[int, list[slice]]]: List of tuples where each tuple contains the index of the trajectory
            and a list of slices representing segments from source to destination.
    """
    fid_slices_pairs = []
    for i, t in enumerate(trajectories):
        if slices := t.get_sd_slice(*sd_pair):
            fid_slices_pairs.append((i, slices))
    return fid_slices_pairs


def get_start_hour(trajectory: Trajectory, location: str) -> int:
    """Get the start hour of a trajectory based on its timestamps."""
    match location:
        case "porto":
            timezone = "Europe/Lisbon"
        case "xian":
            timezone = "Asia/Shanghai"
        case _:
            raise ValueError("Unsupported location for start hour calculation.")
    start_time = datetime.fromtimestamp(trajectory.timestamps[0], tz=ZoneInfo(timezone))
    return start_time.hour


def trajectory_similarity_score(t1: Trajectory, t2: Trajectory, location: str) -> float:
    """Calculate the similarity score between two trajectories based on their start times and edge paths."""
    start_time_1 = get_start_hour(t1, location)
    start_time_2 = get_start_hour(t2, location)
    gap = abs(start_time_1 - start_time_2)
    gap = min(gap, 24 - gap)

    # Time similarity: map from [0, 1] to [0.01, 1.0] to prevent zeroing
    raw_time_similarity = 1.0 - (gap / 12.0)
    time_similarity = 0.01 + 0.99 * raw_time_similarity

    t1_set = set(t1.path)
    t2_set = set(t2.path)

    # Edge similarity: map Jaccard Index from [0, 1] to [0.01, 1.0]
    raw_edge_similarity = len(t1_set & t2_set) / len(t1_set | t2_set)
    edge_similarity = 0.01 + 0.99 * raw_edge_similarity

    return time_similarity * edge_similarity  # smaller is more different


def detour_worker(
    batch_indices: list[int],
    trajectories: list[Trajectory],
    ratio: float,
    location: str,
    random_seed: int,
):
    result = subprocess.check_output(
        ["taskset", "-p", "0xfffff", str(os.getpid())],
        stderr=subprocess.STDOUT,
    )
    first_line, second_line = result.decode().splitlines()
    logger.info(
        "%s",
        first_line.strip(),
    )
    logger.info(
        "%s",
        second_line.strip(),
    )
    results = []
    original_indices = []
    scores = []
    rng = np.random.default_rng(random_seed)
    for idx in batch_indices:
        t = trajectories[idx]
        try:
            start_index = rng.integers(1, len(t) - 2 - np.ceil(len(t) * ratio))
        except ValueError as e:
            logger.error(
                "Error generating start index for trajectory %d: %s",
                idx,
                str(e),
            )
            continue
        head = t[: start_index + 1]
        end_index = int(start_index + np.floor(len(t) * ratio))
        tail = t[end_index:]
        trajectory_need_replace = t[start_index : end_index + 1]
        s, d = trajectory_need_replace.get_sd_pair()
        tid_slices_pairs = get_subset_slices_with_sd(trajectories, (s, d))
        if tid_slices_pairs:
            min_similarity_score = float("inf")
            diff_trajectory = None
            for tid, slices in tid_slices_pairs:
                for trajectory_slice in slices:
                    sub_trajectory = trajectories[tid][trajectory_slice]
                    similarity_score = trajectory_similarity_score(
                        trajectory_need_replace,
                        sub_trajectory,
                        location,
                    )
                    if similarity_score < min_similarity_score:
                        min_similarity_score = similarity_score
                        diff_trajectory = sub_trajectory
            if diff_trajectory is not None:
                scores.append(min_similarity_score)
                results.append(concatenate_times(head, diff_trajectory, tail))
                original_indices.append(idx)
        if idx % 100 == 0 or idx == len(trajectories) - 1:
            logger.info(
                "[Process] Processed trajectory %d/%d",
                idx + 1,
                len(trajectories),
            )
    return results, scores, original_indices


class AnomalyGenerator:
    """Abnormal trajectory generator class"""

    def __init__(
        self,
        location: str,
        anomaly_ratio: float,
        test_trajectories: list[Trajectory],
        rng: np.random.Generator,
        random_seed: int,
        shift_time_gap: int,
        switch_relax: int,
        dataset_type: str = "test",
    ):
        logger.info("Start anomaly generate...")
        self.location = location
        self.anomaly_dir = f"data/{location}/anomaly/"
        self.mkdir_if_not_exists(self.anomaly_dir)
        self.trajectories = test_trajectories
        self.anomaly_ratio = anomaly_ratio
        self.graph_file = f"data/{self.location}/raw/graph.graphml"
        self.edges_file = f"data/{self.location}/raw/edges.shp"
        self.rng = rng
        self.random_seed = random_seed
        self.shift_time_gap = shift_time_gap
        self.switch_relax = switch_relax
        self.dataset_type = dataset_type

    def __call__(
        self, anomaly: str, proportion: float, num_processes: int, batch_size: int
    ) -> list[Trajectory]:
        ds = self.dataset_type
        match anomaly:
            case "detour":
                file_dir = self.anomaly_dir + f"detour_{proportion}_{ds}.pkl"
                if exists(file_dir):
                    anomaly_trajectories = self.load_anomaly(file_dir)
                else:
                    anomaly_trajectories = self.create_detour_anomaly(
                        proportion, num_processes, batch_size
                    )
                    self.save_anomaly(*anomaly_trajectories, file_dir, anomaly)
            case "switch":
                file_dir = (
                    self.anomaly_dir
                    + f"switch_{proportion}_relax_{self.switch_relax}_{ds}.pkl"
                )
                if exists(file_dir):
                    anomaly_trajectories = self.load_anomaly(file_dir)
                else:
                    anomaly_trajectories = self.create_switch_anomaly(
                        proportion, self.switch_relax
                    )
                    self.save_anomaly(*anomaly_trajectories, file_dir, anomaly)
            case "time_shift":
                file_dir = (
                    self.anomaly_dir
                    + f"time_shift_{proportion}_gap_{self.shift_time_gap}_{ds}.pkl"
                )
                if exists(file_dir):
                    anomaly_trajectories = self.load_anomaly(file_dir)
                else:
                    anomaly_trajectories = self.create_time_shift_anomaly(
                        proportion, self.shift_time_gap
                    )
                    self.save_anomaly(*anomaly_trajectories, file_dir, anomaly)
            case _:
                raise ValueError(f"Unknown anomaly type: {anomaly}")
        logger.info(
            "Anomaly generation completed for type '%s' with ratio %.2f",
            anomaly,
            proportion,
        )
        return anomaly_trajectories[0]

    def random_walk(self, g: MultiDiGraph, start_node: int, depth: int):
        possible_nodes = []
        for n, length in nx.single_source_shortest_path_length(
            g, start_node, depth
        ).items():
            if length == depth:
                possible_nodes.append(n)
        return possible_nodes

    def _process_anomalies(
        self,
        worker_func,
        num_processes: int,
        batch_size: int,
        ratio: float,
    ):
        """Generic method to process anomalies using multi-processing and batching."""
        logger.info(
            "Creating anomalies (multi-process=%d, batch_size=%d)...",
            num_processes,
            batch_size,
        )
        if num_processes == 1:
            logger.info("Using single process for anomaly generation.")
            anomaly_trajectories, score_list, original_indices = worker_func(
                list(range(len(self.trajectories))),
                self.trajectories,
                ratio,
                self.location,
                self.random_seed,
            )
        else:
            anomaly_trajectories = []
            score_list = []
            original_indices = []
            number_of_trajectories = len(self.trajectories)
            batches = []
            for batch_index, batch_start in enumerate(
                range(0, number_of_trajectories, batch_size)
            ):
                batch_end = min(batch_start + batch_size, number_of_trajectories)
                batch_indices = list(range(batch_start, batch_end))
                # Prepare only picklable arguments for the worker
                batches.append(
                    (
                        batch_indices,
                        self.trajectories,
                        ratio,
                        self.location,
                        self.random_seed + batch_index,
                    )
                )

            with Pool(processes=num_processes) as pool:
                results = pool.starmap(worker_func, batches)

            for res, sc, origin_idx in results:
                anomaly_trajectories.extend(res)
                score_list.extend(sc)
                original_indices.extend(origin_idx)

        score_list = np.array(score_list, dtype=np.float32)

        num_top = int(len(score_list) * self.anomaly_ratio)
        top_indices = np.argsort(score_list)[:num_top]
        minimum_scores = score_list[top_indices]
        anomaly_trajectories = [anomaly_trajectories[i] for i in top_indices]
        original_indices = [original_indices[i] for i in top_indices]

        logger.info(
            "Anomalies created with %d trajectories (multi-process, batch_size=%d)",
            len(anomaly_trajectories),
            batch_size,
        )
        logger.info("Average similarity score: %.4f", np.mean(minimum_scores))
        logger.info(
            "Standard deviation of similarity scores: %.4f", np.std(minimum_scores)
        )
        return anomaly_trajectories, original_indices

    def create_switch_anomaly(
        self, proportion: float, switch_relax: int = 0
    ) -> tuple[list[Trajectory], list[int]]:
        g = ox.load_graphml(self.graph_file)
        edge_gdf = gpd.read_file(self.edges_file)
        sd_dict = self.get_sd_trajectory_indices_dict(self.trajectories)
        num_anomalies = int(len(self.trajectories) * self.anomaly_ratio)
        switch_anomalies = []
        original_indices = []
        i = 0
        while len(switch_anomalies) < num_anomalies:
            idx = self.rng.choice(len(self.trajectories), 1, replace=False)[0]
            trajectory = self.trajectories[idx]
            # Pick the most dissimilar trajectory sharing the same SD pair via Jaccard
            selected_traj_id = self.select_most_dissimilar_by_jaccard(
                sd_dict, trajectory, idx
            )
            if selected_traj_id == -1:
                continue
            switch_traj = self.trajectories[selected_traj_id]
            anomaly_trajectory = self.switch_trajectory(
                g, edge_gdf, trajectory, switch_traj, proportion, switch_relax
            )
            if anomaly_trajectory is not None:
                switch_anomalies.append(anomaly_trajectory)
                original_indices.append(idx)
            if i % 100 == 0:
                logger.info(
                    "Generated %d/%d switch anomalies for proportion %.2f",
                    i + 1,
                    num_anomalies,
                    proportion,
                )
            i += 1
        logger.info(
            "Generated %d/%d switch anomalies for proportion %.2f",
            len(switch_anomalies),
            num_anomalies,
            proportion,
        )
        return switch_anomalies, original_indices

    def switch_trajectory(
        self,
        g: MultiDiGraph,
        edge_gdf: gpd.GeoDataFrame,
        t_1: Trajectory,
        t_2: Trajectory,
        proportion: float,
        switch_relax: int = 0,
    ) -> Trajectory | None:
        t_1_switch_index = int(len(t_1) * proportion)
        t_2_switch_index = int(len(t_2) * proportion)
        t_1_last_edge = t_1.path[t_1_switch_index - 1]
        t_2_first_edge = t_2.path[t_2_switch_index]
        if t_1_last_edge == t_2_first_edge:
            return concatenate_times(
                t_1[:t_1_switch_index], t_2[t_2_switch_index:]
            )  # minor case
        row = edge_gdf[edge_gdf["fid"] == t_1_last_edge]
        v_1 = row.iloc[0]["v"]  # end node of t_1_last_edge
        row = edge_gdf[edge_gdf["fid"] == t_2_first_edge]
        u_2 = row.iloc[0]["u"]  # start node of t_2_first_edge

        v_1_nodes = [v_1]
        for _ in range(switch_relax):
            successors = list(g.successors(v_1_nodes[-1]))
            if not successors:
                break
            v_1_nodes.append(int(self.rng.choice(successors)))

        u_2_nodes = [u_2]
        for _ in range(switch_relax):
            predecessors = list(g.predecessors(u_2_nodes[-1]))
            if not predecessors:
                break
            u_2_nodes.append(int(self.rng.choice(predecessors)))
        u_2_nodes.reverse()

        v_1_new = v_1_nodes[-1]
        u_2_new = u_2_nodes[0]

        if nx.has_path(g, v_1_new, u_2_new):
            middle_path_nodes = nx.shortest_path(
                g, source=v_1_new, target=u_2_new, weight="length"
            )
            path_nodes = v_1_nodes[:-1] + middle_path_nodes + u_2_nodes[1:]
        else:
            return None
        paths = [t_1_last_edge]
        times = [int(datetime.now().replace(microsecond=0).timestamp())]
        temp_speed = self.get_speed(
            edge_gdf[edge_gdf["fid"] == t_1_last_edge].iloc[0]["maxspeed"]
        )
        speeds = [temp_speed]
        for u, v in zip(path_nodes[:-1], path_nodes[1:]):
            edge_data = g.get_edge_data(u, v)
            best_key = min(
                edge_data, key=lambda k, ed=edge_data: ed[k].get("length", float("inf"))
            )
            edge = edge_gdf[
                (edge_gdf["u"] == u)
                & (edge_gdf["v"] == v)
                & (edge_gdf["key"] == best_key)
            ]
            speed = self.get_speed(edge.iloc[0]["maxspeed"])
            length = edge.iloc[0]["length"]
            travel_time = length / (speed * 1000 / 3600)  # convert speed to m/s
            times.append(int(times[-1] + travel_time))
            speeds.append(speed)
            paths.append(edge.iloc[0]["fid"])
        paths.append(t_2_first_edge)
        length = edge_gdf[edge_gdf["fid"] == t_2_first_edge].iloc[0]["length"]
        speed = self.get_speed(
            edge_gdf[edge_gdf["fid"] == t_2_first_edge].iloc[0]["maxspeed"]
        )
        travel_time = length / (speed * 1000 / 3600)  # convert speed to m/s
        times.append(int(times[-1] + travel_time))
        speeds.append(speed)
        current_trajectory = Trajectory(paths, times, speeds)
        anomaly_trajectory = concatenate_times(
            t_1[:t_1_switch_index], current_trajectory, t_2[t_2_switch_index:]
        )
        return anomaly_trajectory

    def create_detour_anomaly(
        self, proportion: float, num_processes: int, batch_size: int
    ) -> tuple[list[Trajectory], list[int]]:
        """Create detour anomaly by replacing a segment of each trajectory with the most different segment from a subset of trajectories, using multi-processing and batching.
        Args:
            proportion (float): The ratio of the trajectory length to replace with a different segment.
            num_processes (int): The number of processes to use for processing.
            batch_size (int): The number of trajectories per batch.
        Returns:
            list[Trajectory]: List of trajectories with detour anomalies.
        """

        return self._process_anomalies(
            detour_worker, num_processes, batch_size, proportion
        )

    def create_time_shift_anomaly(
        self,
        proportion: float,
        time_shift: int = 3,
    ) -> tuple[list[Trajectory], list[int]]:
        num_anomalies = int(len(self.trajectories) * self.anomaly_ratio)
        anomaly_indices = self.rng.choice(
            len(self.trajectories), num_anomalies, replace=False
        )
        time_shift_anomalies = []
        original_indices = []
        for i, idx in enumerate(anomaly_indices):
            trajectory = self.trajectories[idx]
            try:
                delay_start_index = self.rng.integers(
                    1, len(trajectory) - 2 - np.ceil(len(trajectory) * proportion)
                )
            except ValueError as e:
                logger.error(
                    "Error generating start index for trajectory %d: %s",
                    idx,
                    str(e),
                )
                continue
            delay_end_index = delay_start_index + int(len(trajectory) * proportion)
            shifted_timestamps: list[int] = trajectory.timestamps[:]
            total_delay = 0
            for j in range(delay_start_index, delay_end_index + 1):
                total_delay += time_shift
                shifted_timestamps[j] += total_delay
            for j in range(delay_end_index + 1, len(trajectory)):
                shifted_timestamps[j] += total_delay
            anomaly_trajectory = Trajectory(
                trajectory.path,
                shifted_timestamps,
                trajectory.speed,
            )
            time_shift_anomalies.append(anomaly_trajectory)
            original_indices.append(idx)
            if len(time_shift_anomalies) % 100 == 0:
                logger.info(
                    "Generated %d/%d time shift anomalies for proportion %.2f",
                    len(time_shift_anomalies),
                    num_anomalies,
                    proportion,
                )
        return time_shift_anomalies, original_indices

    def load_anomaly(self, file_path: str) -> tuple[list[Trajectory], list[int]]:
        """Load anomaly dictionary from a file."""
        with open(file_path, "rb") as f:
            trajectories, original_indices = pickle.load(f)
        logger.info(
            "Anomaly loaded from %s with %d trajectories", file_path, len(trajectories)
        )
        return trajectories, original_indices

    def save_anomaly(
        self,
        trajectories: list[Trajectory],
        original_indices: list[int],
        file_path: str,
        anomaly_type: str,
    ):
        """Save anomaly dictionary to a file."""

        with open(file_path, "wb") as f:
            pickle.dump((trajectories, original_indices), f)
        logger.info(
            "Anomaly of type '%s' saved to %s with %d trajectories",
            anomaly_type,
            file_path,
            len(trajectories),
        )

    def mkdir_if_not_exists(self, path: str):
        """Create directory if it does not exist."""
        if not exists(path):
            mkdir(path)
            logger.info("Created directory: %s", path)
        else:
            logger.info("Directory already exists: %s", path)

    def get_sd_trajectory_indices_dict(
        self, trajectories: list[Trajectory]
    ) -> dict[tuple[int, int], list[int]]:
        """_summary_

        Args:
            trajectories (list[Trajectory]): _description_

        Returns:
            dict[tuple[int, int], list[int]]: _description_
        """
        sd_dict = {}
        for i, trajectory in enumerate(trajectories):
            s, d = trajectory.get_sd_pair()
            if (s, d) in sd_dict:
                sd_dict[(s, d)].append(i)
            else:
                sd_dict[(s, d)] = [i]
        return sd_dict

    def get_speed(self, speed) -> float:
        if self.location == "porto":
            default_speed = 50.0  # default speed if not available
        elif self.location == "xian":
            default_speed = 60.0  # default speed if not available
        else:
            raise ValueError(f"Unknown location: {self.location}")

        float_speed = None
        if speed is None:
            float_speed = default_speed
        if isinstance(speed, str):
            speed = ast.literal_eval(speed)
            if isinstance(speed, list):
                float_speed = float(speed[0])
            elif isinstance(speed, int):
                float_speed = float(speed)
            else:
                float_speed = default_speed
        else:
            float_speed = default_speed

        return float_speed

    def select_most_dissimilar_by_jaccard(
        self,
        sd_indices_dict: dict[tuple[int, int], list[int]],
        trajectory: Trajectory,
        traj_index: int,
    ) -> int:
        """
        From all trajectories sharing the same SD pair, pick the one whose
        edge set is *most dissimilar* (lowest Jaccard index) to the given
        trajectory.

        Args:
            sd_indices_dict: Mapping from (source, dest) → list of trajectory indices.
            trajectory: The reference trajectory.
            traj_index: Index of the reference trajectory (to exclude self).

        Returns:
            The index of the most dissimilar trajectory, or -1 if no valid candidate exists.
        """
        sd = trajectory.get_sd_pair()
        if sd not in sd_indices_dict:
            return -1

        candidates = [i for i in sd_indices_dict[sd] if i != traj_index]
        if not candidates:
            return -1

        ref_edges = set(trajectory.path)
        best_idx = -1
        best_sim = float("inf")  # we want the minimum Jaccard (most dissimilar)

        for cand_idx in candidates:
            cand_edges = set(self.trajectories[cand_idx].path)
            # Jaccard = |intersection| / |union|
            intersection = len(ref_edges & cand_edges)
            union = len(ref_edges | cand_edges)
            jaccard = intersection / union if union > 0 else 0.0
            if jaccard < best_sim:
                best_sim = jaccard
                best_idx = cand_idx

        return best_idx
