import numpy as np

from .DatasetDatabase import DatasetDatabase
import datetime as dt

__author__ = 'gm'


class DatasetDBNormalizer:
    def __init__(self, db_name):
        self.db_name = db_name
        self.db = DatasetDatabase(self.db_name)
        self.write_buffer = []

    def normalize(self, data_normalize=False, tick="time-based"):
        assert tick in ["incremental", "time-based"]
        self.db.connect()
        tsnames_list = [t[0] for t in self.db.get_distinct_names()]
        data1 = []
        data2 = []
        for name in tsnames_list:
            ts_list = self.db.get_time_series(name).fetchall()
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
                    self.write_buffer.append((name, i, row[0], row[1], data1_norm[i], data2_norm[i]))
                    i += 1
            elif tick == "time-based":
                first_datetime = dt.datetime.strptime(self.db.get_first_datetime(None), '%m/%d/%Y-%H:%M:%S')
                # last_datetime = dt.datetime.strptime(self.db.get_last_datetime(None), '%m/%d/%Y-%H:%M:%S')

                for row in ts_list:
                    date = row[0]
                    time = row[1]
                    datetime = dt.datetime.strptime(date+"-"+time, '%m/%d/%Y-%H:%M:%S')
                    i = (datetime - first_datetime).seconds
                    self.write_buffer.append((name, i, date, time, data1_norm[i], data2_norm[i]))

            self.db.store_multiple_data(self.write_buffer, table="dataset_normalized")
            self.write_buffer.clear()

        self.db.disconnect()
