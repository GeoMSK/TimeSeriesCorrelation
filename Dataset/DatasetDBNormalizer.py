import numpy as np

from .DatasetDatabase import DatasetDatabase
from Dataset.DatasetDatabase import DATE_FORMAT
import datetime as dt
import h5py

__author__ = 'gm'


class DatasetDBNormalizer:
    def __init__(self):
        pass

    @staticmethod
    def normalize_sqlite(db_name, data_normalize=False, tick="time-based"):
        """
        NOT TESTED thoroughly
        """
        db = DatasetDatabase(db_name)
        write_buffer = []

        assert tick in ["incremental", "time-based"]
        db.connect()
        tsnames_list = [t[0] for t in db.get_distinct_names()]
        data1 = []
        data2 = []
        for name in tsnames_list:
            ts_list = db.get_time_series(name).fetchall()
            # Columns:
            #   name | tick |  ( date | time | data1 | data2 )
            for row in ts_list:
                data1.append(float(row[2]))
                data2.append(float(row[3]))

            if data_normalize:
                d = np.array(data1)
                if np.std(d) == 0:
                    data1_norm = data1
                else:
                    data1_norm = (d - np.mean(d)) / np.std(d)

                d = np.array(data2)
                if np.std(d) == 0:
                    data2_norm = data1
                else:
                    data2_norm = (d - np.mean(d)) / np.std(d)
            else:
                data1_norm = data1
                data2_norm = data2

            if tick == "incremental":
                i = 0
                for row in ts_list:
                    write_buffer.append((name, i, row[0], row[1], data1_norm[i], data2_norm[i]))
                    i += 1
            elif tick == "time-based":
                first_datetime = dt.datetime.strptime(db.get_first_datetime(None), DATE_FORMAT)
                # last_datetime = dt.datetime.strptime(self.db.get_last_datetime(None), DATE_FORMAT)

                for row in ts_list:
                    date = row[0]
                    time = row[1]
                    datetime = dt.datetime.strptime(date + "-" + time, DATE_FORMAT)
                    i = (datetime - first_datetime).seconds
                    write_buffer.append((name, i, date, time, data1_norm[i], data2_norm[i]))

            db.store_multiple_data(write_buffer, table="dataset_normalized")
            write_buffer.clear()

        db.disconnect()

    @staticmethod
    def normalize_hdf5(h5db, h5db_normalized, compression_level=None):
        h5 = h5py.File(h5db, mode='r')
        h5_norm = h5py.File(h5db_normalized, mode='w')

        for ts in h5:
            ts_norm = DatasetDBNormalizer._normalize_time_series(h5[ts][:])
            if compression_level:
                h5_norm.create_dataset(ts, (len(ts_norm),), data=ts_norm, dtype='float32', compression="gzip",
                                       compression_opts=compression_level)
            else:
                h5_norm.create_dataset(ts, (len(ts_norm),), data=ts_norm, dtype='float32')
        h5.close()
        h5_norm.close()

    @staticmethod
    def _normalize_time_series(time_series_data: np.array):
        """
        normalize an array of float values
        :param time_series_data: a numpy array with float values
        :return: a numpy array with normalized data
        """
        assert isinstance(time_series_data, np.ndarray)
        d = time_series_data
        if np.std(d) == 0:
            data_norm = d
        else:
            data_norm = (d - np.mean(d)) / np.std(d)
        return data_norm
