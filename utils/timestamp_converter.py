from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import holidays
from utils.trajectory import Trajectory


class TimestampConverter:
    """
    TimestampConverter is a utility class for converting Unix timestamps in trajectory
    data into integer tokens, taking into account time zones, holidays, and workdays
    for specific locations.

    Attributes:
        timezone (pytz.timezone): The timezone corresponding to the specified location.
        holidays (holidays.HolidayBase): A holidays calendar for the specified location and years.
        num_workdays_tokens (int): The number of tokens representing a full day in seconds (default: 24 * 60 * 60).
        num_time_tokens (int): The number of tokens representing two full days in seconds (default: 24 * 60 * 60 * 2).

    """

    def __init__(self, location: str):
        """
        Initializes the object with timezone and holiday information based on the
        specified location.
        Args:
            location (str): The location for which to set timezone and holidays.
                Supported values are:
                    - "porto": Sets timezone to Europe/Lisbon and holidays for Porto,
                    Portugal (subdivisions "13" and "Ext" for years 2013 and 2014).
                    - "xian": Sets timezone to Asia/Shanghai and Chinese holidays.
        Raises:
            ValueError: If the provided location is not supported.
        """

        match location:
            case "porto":
                self.timezone = ZoneInfo("Europe/Lisbon")
                self.holidays = holidays.country_holidays(
                    "PT", language="en_US", subdiv="13", years={2013, 2014}
                ) + holidays.country_holidays(
                    "PT", language="en_US", subdiv="Ext", years={2013, 2014}
                )
            case "xian":
                self.timezone = ZoneInfo("Asia/Shanghai")
                self.holidays = holidays.country_holidays("CN", years={2016})
            case _:
                raise ValueError("wrong location")
        self.num_workdays_tokens = 24 * 60 * 60
        self.num_time_tokens = 24 * 60 * 60 * 2

    def __call__(self, trajectory: Trajectory) -> list[int]:
        """
        Converts the timestamps of a given Trajectory object into a list of integer tokens.
        Args:
            trajectory (Trajectory): The trajectory object containing timestamps to be converted.
        Returns:
            list[int]: A list of integer tokens corresponding to the input timestamps.
        """

        time_tokens = []
        for timestamp in trajectory.timestamps:
            time_tokens.append(self.timestamp2token(timestamp))
        return time_tokens

    def get_time_tuple(
        self, trajectory: Trajectory
    ) -> list[tuple[int, int, int, int, int, int]]:
        """
        Converts the timestamps of a given Trajectory object into a list of time tuples.
        Each tuple contains (year, month, day, hour, minute, second).
        Args:
            trajectory (Trajectory): The trajectory object containing timestamps to be converted.
        Returns:
            list[tuple[int, int, int, int, int, int]]: A list of time tuples corresponding to the input timestamps.
        """
        time_tuples = []
        for timestamp in trajectory.timestamps:
            dt = datetime.fromtimestamp(timestamp, tz=self.timezone)
            time_tuples.append(dt.timetuple()[:6])
        return time_tuples

    def convert_to_time(
        self, trajectory: Trajectory, sample_second: int = 15
    ) -> list[int]:
        time_tokens = []
        for timestamp in trajectory.timestamps:
            date_time = datetime.fromtimestamp(timestamp, tz=self.timezone)
            time_tokens.append(self.get_base_token(date_time) // sample_second)
        return time_tokens

    def convert_to_tau(self, time: list[int]):
        """
        Converts a list of timestamps into a list of tau features.
        Args:
            time (list[int]): A list of timestamps to be converted.
        Returns:
            list[list[int]]: A list of tau features, where each feature is a list
            containing [hour, minute, second, year, month, day].
        """
        tau_list = []
        for timestamp in time:
            t = datetime.fromtimestamp(timestamp, self.timezone)
            tau_list.append([t.hour, t.minute, t.second, t.year, t.month, t.day])
        return tau_list

    def get_start_hour(self, trajectory: Trajectory) -> int:
        """Give the start hour of this trajectory

        Args:
            trajectory (Trajectory): Given trajectory

        Returns:
            int: (0-23)
        """
        start_time = datetime.fromtimestamp(trajectory.timestamps[0], tz=self.timezone)
        return start_time.hour

    def is_workday(self, trajectory: Trajectory) -> bool:
        """
        Determines whether the start time of a given trajectory falls on a workday.
        Args:
            trajectory (Trajectory): The trajectory object containing timestamps.
        Returns:
            bool: True if the start time is a workday according to the holidays
            calendar, False otherwise.
        """

        start_time = datetime.fromtimestamp(trajectory.timestamps[0], tz=self.timezone)
        return self.holidays.is_working_day(start_time)

    @staticmethod
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

    def time_shift(
        self, trajectory: Trajectory, start_hour: int, new_start_hour: int
    ) -> list[int]:
        """
        Shifts the timestamps of a trajectory by a specified number of hours
        and converts them to tokens.
        Args:
            trajectory (Trajectory): The trajectory object containing timestamps
            to be shifted.
            start_hour (int): The original starting hour of the trajectory.
            new_start_hour (int): The desired new starting hour for the trajectory.
        Returns:
            TokenTrajectory: A new TokenTrajectory object with the same path as
            the input trajectory and time tokens corresponding to the shifted timestamps.
        """
        gap = new_start_hour - start_hour
        time_tokens = []
        for timestamp in trajectory.timestamps:
            date_time = datetime.fromtimestamp(timestamp)
            date_time += timedelta(hours=gap)
            time_tokens.append(self.get_token(date_time))
        return time_tokens

    def timestamp2token_inverse_date(self, timestamp: float):
        """
        Converts a Unix timestamp to a token representing the time of day in seconds,
        with an offset applied if the date is a working day.
        Args:
            timestamp (float): The Unix timestamp to convert.
        Returns:
            int: The token representing the time of day in seconds. If the date is a working day,
                 an offset equal to `self.num_workdays_tokens` is added to the token.
        """

        date_time = datetime.fromtimestamp(timestamp, tz=self.timezone)
        token = date_time.hour * 60 * 60 + date_time.minute * 60 + date_time.second
        if self.holidays.is_working_day(date_time):
            token += self.num_workdays_tokens
        return token

    def timestamp2token(self, timestamp: int) -> int:
        """
        Converts a Unix timestamp to a token representation.
        Args:
            timestamp (int): The Unix timestamp to convert.
        Returns:
            int: The token corresponding to the given timestamp.
        Note:
            The conversion uses the instance's timezone and the `get_token` method.
        """

        date_time = datetime.fromtimestamp(timestamp, tz=self.timezone)
        return self.get_token(date_time)

    def get_token(self, date_time: datetime) -> int:
        """
        Converts a datetime object to a token, adding an offset if the date is not a working day.
        Args:
            date_time (datetime): The datetime object to convert.
        Returns:
            int: The token representing the time of day, with an offset for non-working days.
        """
        token = self.get_base_token(date_time)
        if not self.holidays.is_working_day(date_time):
            token += self.num_workdays_tokens
        return token

    def get_base_token(self, date_time: datetime) -> int:
        """
        Convert a datetime object to a base time token (seconds since midnight).
        Args:
            date_time (datetime): The datetime object to convert.
        Returns:
            int: The number of seconds since midnight.
        """
        return date_time.hour * 60 * 60 + date_time.minute * 60 + date_time.second
