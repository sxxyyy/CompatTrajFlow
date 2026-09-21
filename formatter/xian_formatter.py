import logging
import math
from datetime import datetime
from os import listdir
from zoneinfo import ZoneInfo

import pandas as pd

from .abstract_formatter import AbstractFormatter

logger = logging.getLogger(__name__)


class XianFormatter(AbstractFormatter):
    def __init__(self, raw_data_dir: str, trajectory_len_threshold: tuple[int, int]):
        super().__init__(raw_data_dir, trajectory_len_threshold)
        self.raw_data_dir = raw_data_dir
        self.csv_path = f"{raw_data_dir}/2016_xian"

    def unzip(self):
        pass  # No zip file to unzip for Xian data

    @staticmethod
    def transform_lat(x: float, y: float):
        ret = (
            -100.0
            + 2.0 * x
            + 3.0 * y
            + 0.2 * y * y
            + 0.1 * x * y
            + 0.2 * math.sqrt(abs(x))
        )
        ret += (
            (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi))
            * 2.0
            / 3.0
        )
        ret += (
            (20.0 * math.sin(y * math.pi) + 40.0 * math.sin(y / 3.0 * math.pi))
            * 2.0
            / 3.0
        )
        ret += (
            (160.0 * math.sin(y / 12.0 * math.pi) + 320 * math.sin(y * math.pi / 30.0))
            * 2.0
            / 3.0
        )
        return ret

    @staticmethod
    def transform_lon(x: float, y: float):
        ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
        ret += (
            (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi))
            * 2.0
            / 3.0
        )
        ret += (
            (20.0 * math.sin(x * math.pi) + 40.0 * math.sin(x / 3.0 * math.pi))
            * 2.0
            / 3.0
        )
        ret += (
            (
                150.0 * math.sin(x / 12.0 * math.pi)
                + 300.0 * math.sin(x / 30.0 * math.pi)
            )
            * 2.0
            / 3.0
        )
        return ret

    def gcj02_to_wgs84(self, lat: float, lon: float):
        a = 6378245.0
        ee = 0.00669342162296594323
        dLat = self.transform_lat(lon - 105.0, lat - 35.0)
        dLon = self.transform_lon(lon - 105.0, lat - 35.0)
        radLat = lat / 180.0 * math.pi
        magic = math.sin(radLat)
        magic = 1 - ee * magic * magic
        sqrtMagic = math.sqrt(magic)
        dLat = (dLat * 180.0) / ((a * (1 - ee)) / (magic * sqrtMagic) * math.pi)
        dLon = (dLon * 180.0) / (a / sqrtMagic * math.cos(radLat) * math.pi)
        mgLat = lat + dLat
        mgLon = lon + dLon
        wgsLat = lat * 2 - mgLat
        wgsLon = lon * 2 - mgLon
        return wgsLat, wgsLon

    def format(self, threshold: tuple[int, int]):
        # Convert the Xi'an dataset to the point format used by FMM.
        file_list = listdir(self.csv_path)

        self.init_filtered_point_file()
        fid = 0
        logger.info("Processing Xian data")
        with open(self.filtered_points_file, mode="a", encoding="utf-8") as p:
            for file_name in file_list:
                if not file_name.endswith(".csv"):
                    continue
                logger.info("Processing csv file: %s", file_name)
                df = pd.read_csv(f"{self.csv_path}/{file_name}")
                df = df.sort_values(
                    ["车辆ID", "订单ID", "GPS时间"], ascending=[False, False, True]
                )
                grouped = df.groupby(["车辆ID", "订单ID"])
                for _, vehicle_group in grouped:
                    gps_times = vehicle_group["GPS时间"].tolist()
                    if len(gps_times) < threshold[0]:
                        continue
                    lats = vehicle_group["轨迹点纬度"].tolist()
                    lons = vehicle_group["轨迹点经度"].tolist()
                    converted = [
                        self.gcj02_to_wgs84(lat, lon) for lat, lon in zip(lats, lons)
                    ]
                    lats, lons = map(list, zip(*converted))
                    for lat, lon, ts in zip(lats, lons, gps_times):
                        ts = (
                            datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                            .replace(tzinfo=ZoneInfo("Asia/Shanghai"))
                            .timestamp()
                        )
                        line = f"{fid};{lon};{lat};{ts}"
                        p.write(line + "\n")
                    fid += 1
        logger.info(
            "%d valid trajectories in %s",
            fid,
            self.filtered_points_file,
        )
        logger.info("Xian data processing completed.")
