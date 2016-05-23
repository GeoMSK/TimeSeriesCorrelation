from .DatasetDatabase import DatasetDatabase
from Dataset.DatasetDatabase import DATE_FORMAT
import h5py
import numpy as np
import datetime as dt

__author__ = 'gm'

delta1sec = dt.timedelta(seconds=1)
delta_zero = dt.timedelta(seconds=0)


class DatasetDB2HDF5:
    """
    Provides functionality to convert a dataset stored in a sqlite3 database to a hdf5 database.
    """

    def __init__(self, db_name, hdf5_name):
        """
        :param db_name: the sqlite3 database file
        :param hdf5_name: the hdf5 database file to be created
        """
        self.db_name = db_name
        self.hdf5_name = hdf5_name
        self.first_datetime = None
        self.last_datetime = None
        self.db = None  # sqlite database
        self.h5 = None  # hdf5 database
        self.first_datetime_of_ts = None  # temp variable holding the first datetime of a time series
        self.last_datetime_of_ts = None  # temp variable, holding the last datetime of a time series
        self.ts = []  # temp variable, holding a time series

    def convert(self, range=None, compression_level=None, point_threshold=None):
        """
        convert a dataset stored in a sqlite3 database to a hdf5 database. Data interval in every time series is
        one second. For those seconds that the dataset has no data we put the previous available data to fill the gaps.

        range is used to filter the time series to write to the new database
        range = [start_date, end_date] date: '%m/%d/%Y-%H:%M:%S'

        point_threshold if not None will determine if a time series name will not be displayed based on
        the number of data it has.
        eg. if point_threshold is 100 and a time series has 90 data points then it will be discarded.
        It can also be a percentage of the max data points in the range specified
        eg. point_threshold="%50"
        """
        assert compression_level in [None, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.db = DatasetDatabase(self.db_name)
        self.db.connect()

        # get the globally first and last date-times (of all time series)
        self.first_datetime = dt.datetime.strptime(self.db.get_first_datetime(None), DATE_FORMAT)
        self.last_datetime = dt.datetime.strptime(self.db.get_last_datetime(None), DATE_FORMAT)

        self.h5 = h5py.File(self.hdf5_name, mode='w')

        i = 1
        for ts_name in self.db.get_distinct_names(range=range, point_threshold=point_threshold):
            self._convert_time_series(ts_name, compression_level)
            if i % 100 == 0:
                print("processed %d time series" % i)
            i += 1

        self.h5.close()
        self.db.disconnect()

    def _convert_time_series(self, ts_name, compression_level):
        self.ts = self.db.get_time_series(ts_name).fetchall()
        # ts --> [[date, time, data1, data2], ...]
        gap_filled_ts = []

        # get the first and last date-times of this time series
        self.first_datetime_of_ts = dt.datetime.strptime(self.ts[0][0] + "-" + self.ts[0][1], DATE_FORMAT)
        self.last_datetime_of_ts = dt.datetime.strptime(self.ts[-1][0] + "-" + self.ts[-1][1], DATE_FORMAT)

        # FS: First Segment
        # MS: Middle Segment
        # LS: Last Segment
        #
        # |--FS--| |--MS--| |--LS--|
        # ........ ........ ........  <-- time series
        # |---the globally first datetime(self.first_datetime)
        #          |---the first datetime of the time series(self.first_datetime_of_ts)
        #                 |---the last datetime of the time series(self.last_datetime_of_ts)
        #                          |---the globally last datetime(self.last_datetime)

        # First segment filling
        # There is no First segment if self.first_datetime_of_ts == self.first_datetime
        if self.first_datetime_of_ts != self.first_datetime:
            self._fill_first_segment(gap_filled_ts)

        # Middle segment filling
        # This is always present
        self._fill_middle_segment(gap_filled_ts)

        # Last segment filling
        # There is no Last segment if self.last_datetime_of_ts == self.last_datetime
        if self.last_datetime_of_ts != self.last_datetime:
            self._fill_last_segment(gap_filled_ts)

        ts_array = np.array(gap_filled_ts, dtype='float32')
        if compression_level:
            self.h5.create_dataset(ts_name, (len(gap_filled_ts),), data=ts_array, dtype='float32', compression="gzip",
                                   compression_opts=compression_level)
        else:
            self.h5.create_dataset(ts_name, (len(gap_filled_ts),), data=ts_array, dtype='float32')

    def _fill_first_segment(self, gap_filled_ts):
        """
        fills the first segment, copies the first point of self.ts to all missing points from global start timedate
        assume global start timedate is 00:00:00 and start of timeseries is 00:00:05 with data 2.1

        00:00:05
           2.1

        after filling:

        00:00:00  00:00:01  00:00:02  00:00:03  00:00:04  00:00:05
           2.1       2.1       2.1       2.1       2.1       2.1

        """
        assert self.first_datetime_of_ts > self.first_datetime
        delta = self.first_datetime_of_ts - self.first_datetime
        first_ts_data = self.ts[0][2]
        seconds = delta.days * 86400 + delta.seconds
        DatasetDB2HDF5._fill_N_points(gap_filled_ts, first_ts_data, seconds)

    def _fill_middle_segment(self, gap_filled_ts):
        """
        fills the middle segment, puts the first point of self.ts in gap_filled_ts and then does the following:
        if the next point to insert is ahead of the previous for more than one second, then fill every missing second
        by copying the data of the previous point
        00:00:00  00:00:05
           1.1       2.3

        after filling:

        00:00:00  00:00:01  00:00:02  00:00:03  00:00:04  00:00:05
           1.1       1.1       1.1       1.1       1.1       2.3
        """
        prev_datetime = None
        prev_data = None
        assert len(self.ts) != 0
        for row in self.ts:  # for every row in the current time series
            date = row[0]
            time = row[1]
            cur_data = row[2]
            cur_datetime = dt.datetime.strptime(date + "-" + time, DATE_FORMAT)
            if prev_datetime is not None:
                assert cur_datetime > prev_datetime
                delta = cur_datetime - prev_datetime
                assert delta != delta_zero
                if delta > delta1sec:
                    # fill gaps
                    seconds = delta.days * 86400 + delta.seconds
                    assert prev_data
                    assert seconds - 1 > 0
                    DatasetDB2HDF5._fill_N_points(gap_filled_ts, prev_data, seconds - 1)

            gap_filled_ts.append(cur_data)
            prev_datetime = cur_datetime
            prev_data = cur_data

    def _fill_last_segment(self, gap_filled_ts):
        """
        fills the last segment, copies the last point of self.ts to all missing points from time series
        last point till global last timedate
        assume global last timedate is 00:00:10 and last of timeseries is 00:00:05 with data 2.1

        00:00:05
           2.1

        after filling:

        00:00:05  00:00:06  00:00:07  00:00:08  00:00:09  00:00:10
           2.1       2.1       2.1       2.1       2.1       2.1

        """
        assert self.last_datetime_of_ts < self.last_datetime
        delta = self.last_datetime - self.last_datetime_of_ts
        last_ts_data = self.ts[-1][2]
        seconds = delta.days * 86400 + delta.seconds
        DatasetDB2HDF5._fill_N_points(gap_filled_ts, last_ts_data, seconds)

    @staticmethod
    def _fill_N_points(gap_filled_ts: list, data: float, N: int):
        """
        append data N times to gap_filled_ts
        """
        assert N > 0
        for i in range(N):
            gap_filled_ts.append(data)
