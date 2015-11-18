from .DatasetDatabase import DatasetDatabase
import h5py
import numpy as np
import datetime as dt

__author__ = 'gm'


class DatasetDB2HDF5:
    def __init__(self, db_name, hdf5_name):
        self.db_name = db_name
        self.hdf5_name = hdf5_name

    def convert(self):
        db = DatasetDatabase(self.db_name)
        db.connect()

        first_datetime = dt.datetime.strptime(db.get_first_datetime(None), '%m/%d/%Y-%H:%M:%S')
        last_datetime = dt.datetime.strptime(db.get_last_datetime(None), '%m/%d/%Y-%H:%M:%S')
        delta1sec = dt.timedelta(seconds=1)

        with h5py.File(self.hdf5_name, mode='w') as h5:

            for name in db.get_distinct_names():  # for every time series
                prev_datetime = first_datetime
                ts = db.get_time_series(name).fetchall()
                # ts --> [[date, time, data1, data2], ...]
                gap_filled_ts = []

                first_datetime_of_ts = dt.datetime.strptime(ts[0][0] + "-" + ts[0][1], '%m/%d/%Y-%H:%M:%S')
                last_datetime_of_ts = dt.datetime.strptime(ts[-1][0] + "-" + ts[-1][1], '%m/%d/%Y-%H:%M:%S')
                if first_datetime_of_ts != first_datetime:
                    prev_data = ts[0][2]
                    prev_datetime = prev_datetime - delta1sec
                if last_datetime_of_ts != last_datetime:
                    ts.append([last_datetime.date(), last_datetime.time(), ts[-1][2]])

                for row in ts:  # for every row in the current time series
                    date = row[0]
                    time = row[1]
                    cur_data = row[2]
                    cur_datetime = dt.datetime.strptime(date + "-" + time, '%m/%d/%Y-%H:%M:%S')
                    delta = cur_datetime - prev_datetime
                    if not delta == delta1sec:
                        # fill gaps
                        for i in range(delta.seconds):
                            assert prev_data
                            gap_filled_ts.append(prev_data)

                    gap_filled_ts.append(cur_data)
                    prev_datetime = cur_datetime
                    prev_data = cur_data
                ts_array = np.array(gap_filled_ts, dtype='float32')
                h5.create_dataset(name, (len(gap_filled_ts),), data=ts_array, dtype='float32', compression="gzip",
                                  compression_opts=9)

        db.disconnect()
