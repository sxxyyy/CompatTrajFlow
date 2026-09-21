from math import ceil


class TokenTrajectory:
    """Represents a tokenized trajectory with a path, time, and tau features."""

    def __init__(
        self,
        path: list[int],
        time: list[int] | None = None,
        time_tuple: list[tuple[int, int, int, int, int, int]] | None = None,
        tau: list[list[int]] | None = None,
        speed: list[float] | None = None,
    ) -> None:
        self.path = path
        self.time = time
        self.time_tuple = time_tuple
        self.tau = tau
        self.speed = speed


class Trajectory:
    """
    Represents a trajectory with a path, timestamps, and speed for each point.

    Attributes:
        path (list[int]): Sequence of node IDs representing the trajectory.
        timestamps (list[int]): Unix timestamps for each node in the path.
        speed (list[float]): Speed values for each point in the path.
    """

    def __init__(
        self, path: list[int], timestamps: list[int], speed: list[float]
    ) -> None:
        """
        Initialize a Trajectory instance.

        Args:
            path (list[int]): Sequence of node IDs.
            timestamps (list[int]): Unix timestamps for each node.
            speed (list[float]): Speed values for each point.
        """
        assert (
            len(path) == len(timestamps) == len(speed)
        ), "Path, timestamps, and speed must have the same length."
        self.path = path
        self.timestamps = timestamps
        self.speed = speed

    def set_observation_ratio(self, observation_ratio: float) -> None:
        """
        Set the observation ratio for the trajectory.

        Args:
            observation_ratio (float): Ratio of observations to keep.
        """
        if not (0 < observation_ratio <= 1):
            raise ValueError("Observation ratio must be between 0 and 1.")
        num_observations = ceil(len(self.path) * observation_ratio)
        self.path = self.path[:num_observations]
        self.timestamps = self.timestamps[:num_observations]
        self.speed = self.speed[:num_observations]

    def get_sd_slice(self, source_node: int, destination_node: int) -> list[slice]:
        """
        Get slices of the trajectory path between source and destination nodes.
        Args:
            source_node (int): ID of the source node.
            destination_node (int): ID of the destination node.
        Returns:
            list[slice]: List of slices representing segments from source to destination.
        """
        slices = []
        start_indices = []
        end_indices = []
        for i, edge in enumerate(self.path):
            if edge == source_node:
                start_indices.append(i)
            if edge == destination_node:
                end_indices.append(i)
        if start_indices and end_indices:
            for start_index in start_indices:
                for end_index in end_indices:
                    if start_index < end_index:
                        slices.append(slice(start_index, end_index + 1))
        return slices

    def get_start_time(self) -> int:
        """
        Get the Unix timestamp of the first node in the trajectory.

        Returns:
            int: Start time as a Unix timestamp.
        """
        return self.timestamps[0]

    def get_sd_pair(self) -> tuple[int, int]:
        """
        Get the start and destination node IDs of the trajectory.

        Returns:
            tuple[int, int]: (start_node, destination_node)
        """
        return self.path[0], self.path[-1]

    def __getitem__(self, index: slice) -> "Trajectory":
        if isinstance(index, slice):
            return Trajectory(
                self.path[index],
                self.timestamps[index],
                self.speed[index],
            )
        raise TypeError(f"Index must be a slice, not {type(index).__name__}")

    def __add__(self, other: "Trajectory") -> "Trajectory":
        """
        Concatenate two trajectories.

        Args:
            other (Trajectory): Another trajectory to concatenate.

        Returns:
            Trajectory: New trajectory with combined path, timestamps, and speed.
        """
        return Trajectory(
            self.path + other.path,
            self.timestamps + other.timestamps,
            self.speed + other.speed,
        )

    def __len__(self) -> int:
        """
        Get the number of nodes in the trajectory.

        Returns:
            int: Length of the trajectory path.
        """
        return len(self.path)

    def __iter__(self):
        """
        Iterate over the path, timestamps, and speed values.

        Yields:
            tuple[int, int, float]: (path token, timestamp, speed) for each point.
        """
        for i in range(len(self)):
            yield (self.path[i], self.timestamps[i], self.speed[i])

    def __str__(self):
        """
        Return a string representation of the trajectory.

        Returns:
            str: String with (path, timestamps, speed).
        """

        return str((self.path, self.timestamps, self.speed))
