import ast
import logging
from os import remove
from os.path import exists
from zipfile import ZipFile

import pandas as pd

from .abstract_formatter import AbstractFormatter

logger = logging.getLogger(__name__)


class PortoFormatter(AbstractFormatter):
    def __init__(self, raw_data_dir: str, trajectory_len_threshold: tuple[int, int]):
        super().__init__(raw_data_dir, trajectory_len_threshold)
        self.raw_data_dir = raw_data_dir
        self.zip_path = f"{raw_data_dir}/train.csv.zip"
        self.porto_file_path = f"{raw_data_dir}/train.csv"

    def unzip(self):
        """Unzip porto data train.csv.zip"""
        if not exists(self.porto_file_path):
            logger.info("Unzip porto data")

            logger.info("Unzip %s", self.zip_path)
            with ZipFile(self.zip_path, "r") as my_zip:
                my_zip.extractall(self.raw_data_dir)
                my_zip.close()
            logger.info("Remove %s", self.zip_path)
            remove(self.zip_path)

    def format(self, threshold: tuple[int, int]):
        train_csv = pd.read_csv(self.porto_file_path, header=0, index_col="TRIP_ID")

        self.init_filtered_point_file()

        logger.info("Processing %s", self.porto_file_path)
        valid_trajectories_count = 0
        with open(self.filtered_points_file, mode="a", encoding="utf-8") as f:
            for i, (_, trip) in enumerate(train_csv.iterrows()):
                fid = i
                geo_trajectory = ast.literal_eval(trip["POLYLINE"])
                if threshold[0] > len(geo_trajectory):
                    continue
                timestamp = float(trip["TIMESTAMP"])
                for lon, lat in geo_trajectory:
                    line = f"{fid};{lon};{lat};{timestamp}"
                    f.write(line + "\n")
                    timestamp += 15
                valid_trajectories_count += 1
            f.close()

        logger.info(
            "%d / %d valid/total trajectories in %s",
            valid_trajectories_count,
            len(train_csv),
            self.porto_file_path,
        )

        logger.info("Porto data processing completed.")
