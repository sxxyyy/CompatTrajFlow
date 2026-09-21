import logging
from argparse import Namespace
from datetime import datetime

import networkx as nx
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.trajectory import TokenTrajectory, Trajectory

logger = logging.getLogger(__name__)


class TrajectoryDataset(Dataset):
    """
    A PyTorch Dataset for handling trajectories and their corresponding labels.
    Attributes:
        trajectories (list[Trajectory]): A list of trajectories.
        labels (list[int]): A list of integer labels corresponding to each trajectory.
    Methods:
        __len__(): Returns the number of trajectories in the dataset.
        __getitem__(index): Retrieves the trajectory and label at the specified index.
    Args:
        trajectories (list[Trajectory]): The trajectories to include
        in the dataset.
        labels (list[int]): The labels corresponding to each trajectory.
    """

    def __init__(self, trajectories: list[Trajectory], labels: list[int]):
        assert len(trajectories) == len(labels), (
            "Trajectories and labels must have the same length."
        )
        self.trajectories = trajectories
        self.labels = labels

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, index: int) -> tuple[Trajectory, int]:
        return self.trajectories[index], self.labels[index]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class TokenTrajectoryDataset(Dataset):
    """
    A PyTorch Dataset for handling token trajectories and their corresponding labels.
    Attributes:
        token_trajectories (list[TokenTrajectory]): A list of token trajectories.
        labels (list[int]): A list of integer labels corresponding to each trajectory.
    Methods:
        __len__(): Returns the number of trajectories in the dataset.
        __getitem__(index): Retrieves the trajectory and label at the specified index.
    Args:
        token_trajectories (list[TokenTrajectory]): The token trajectories to include
        in the dataset.
        labels (list[int]): The labels corresponding to each trajectory.
    """

    def __init__(
        self,
        token_trajectories: list[TokenTrajectory],
        labels: list[int],
    ):
        assert len(token_trajectories) == len(labels), (
            "Trajectories and labels must have the same length."
        )
        self.token_trajectories = token_trajectories
        self.labels = labels

    def __len__(self):
        return len(self.token_trajectories)

    def __getitem__(self, index: int) -> tuple[TokenTrajectory, int]:
        return self.token_trajectories[index], self.labels[index]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class GraphTrajectoryDataset(Dataset):
    def __init__(
        self,
        args: Namespace,
        token_trajectories: list[TokenTrajectory],
        labels: list[int],
    ):
        assert len(token_trajectories) == len(labels), (
            "Trajectories and labels must have the same length."
        )
        self.graphs, self.adj_matrices, self.adj_masks, kept_indices = (
            self.build_graphs(token_trajectories)
        )
        self.labels = [labels[i] for i in kept_indices]

    def _build_graph(self, trajectory: TokenTrajectory):
        graph = nx.DiGraph()

        node_ids = {node: i for i, node in enumerate(set(trajectory.path))}
        for pre_node, pre_time, next_node, next_time in zip(
            trajectory.path[:-1],
            trajectory.time_tuple[:-1],
            trajectory.path[1:],
            trajectory.time_tuple[1:],
        ):
            next_time = datetime(*next_time)
            pre_time = datetime(*pre_time)
            edge_attributes = {"travel_time": (next_time - pre_time).total_seconds()}
            if edge_attributes["travel_time"] < 0:
                return None, None, None
            if graph.has_edge(pre_node, next_node):
                attribute_vale = (
                    graph[pre_node][next_node]["travel_time"]
                    + edge_attributes["travel_time"]
                )
                graph[pre_node][next_node]["travel_time"] = attribute_vale
            else:
                graph.add_edge(pre_node, next_node, **edge_attributes)

        adj_matrix = np.zeros((64, 64), dtype=np.float32)
        for u, v in graph.edges():
            u_idx = node_ids[u]
            v_idx = node_ids[v]
            if u_idx < 64 and v_idx < 64:
                adj_matrix[u_idx, v_idx] = graph[u][v]["travel_time"]

        num_nodes = len(node_ids)
        adj_mask = np.zeros((64, 64), dtype=bool)
        for i in range(min(num_nodes, 64)):
            adj_mask[i, :num_nodes] = True
        return graph, adj_matrix, adj_mask

    def build_graphs(self, trajectories: list[TokenTrajectory]):
        graphs = []
        adj_matrices = []
        adj_masks = []
        kept_indices = []
        skipped_count = 0
        for i, trajectory in enumerate(trajectories):
            if i % 1000 == 0:
                logger.info(
                    "Building graph for trajectory %d/%d", i + 1, len(trajectories)
                )
            graph, adj, mask = self._build_graph(trajectory)
            if graph is None:
                logger.warning(
                    "Skipping trajectory %d: negative travel_time detected in "
                    "time_tuple. This indicates non-monotonic timestamps which "
                    "should have been filtered in preprocessing.",
                    i,
                )
                skipped_count += 1
                continue
            graphs.append(graph)
            adj_matrices.append(adj)
            adj_masks.append(mask)
            kept_indices.append(i)
        if skipped_count > 0:
            logger.warning(
                "Skipped %d / %d trajectories due to non-monotonic timestamps.",
                skipped_count,
                len(trajectories),
            )
        return graphs, adj_matrices, adj_masks, kept_indices

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index: int):
        graph = self.graphs[index]
        adj_matrix = self.adj_matrices[index]
        adj_mask = self.adj_masks[index]
        label = self.labels[index]

        node_ids = list(graph.nodes())
        edge_list = list(graph.edges())
        edge_attrs = [graph[u][v].get("travel_time", 0) for u, v in edge_list]
        return (
            torch.tensor(node_ids, dtype=torch.long, device="cpu"),
            torch.tensor(edge_list, dtype=torch.long, device="cpu"),
            torch.tensor(edge_attrs, dtype=torch.long, device="cpu"),
            torch.tensor(adj_matrix, dtype=torch.float, device="cpu"),
            torch.tensor(adj_mask, dtype=torch.bool, device="cpu"),
            torch.tensor(label, dtype=torch.long, device="cpu"),
        )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
