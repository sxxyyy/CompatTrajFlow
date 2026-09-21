"Abstract Formatter module"

import logging
from os.path import exists

logger = logging.getLogger(__name__)


class AbstractFormatter:
    """Formatter class"""

    def __init__(self, raw_data_dir: str, trajectory_len_threshold: tuple[int, int]):
        self.filtered_points_file = f"{raw_data_dir}/filtered_points.csv"
        self.boundary_file = f"{raw_data_dir}/boundary.pkl"
        self.threshold = trajectory_len_threshold

    def __call__(self):
        if self.is_filtered_points_file_exist():
            logger.info("Filtered points file exists")
        else:
            logger.info("Filtered points file did not exist")
            self.unzip()
            self.format(self.threshold)

    def unzip(self):
        """Unzip the raw data file"""
        raise NotImplementedError()

    def format(self, threshold: tuple[int, int]):
        """Format raw data to points file in .csv format with head id;x;y;timestamp,
        where x and y are longitude and latitude.

        Args:
            threshold (tuple[int, int]): The minimum and maximum length thresholds for trajectories.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError()

    def init_filtered_point_file(self):
        """create .csv point file with head id;x;y;timestamp"""
        logger.info("Writing header to points file dir: %s", self.filtered_points_file)
        with open(self.filtered_points_file, mode="w", encoding="utf-8") as f:
            f.write("id;x;y;timestamp\n")
            f.close()

    def is_filtered_points_file_exist(self) -> bool:
        """
        Check if the filtered points file exists.
        Returns:
            bool: True if the filtered points file exists, False otherwise.
        """

        return exists(self.filtered_points_file)
