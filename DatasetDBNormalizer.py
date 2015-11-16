from DatasetDatabase import DatasetDatabase
import numpy as np

__author__ = 'gm'


class DatasetDBNormalizer:
    def __init__(self, db_name):
        self.db_name = db_name
        self.db = DatasetDatabase(self.db_name)
        self.write_buffer = []

    def normalize(self):
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

            # normalize
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

            i = 0
            for row in ts_list:
                self.write_buffer.append((name, i, row[0], row[1], data1_norm[i], data2_norm[i]))
                i += 1

            self.db.store_multiple_data(self.write_buffer, table="dataset_normalized")
            self.write_buffer.clear()

        self.db.disconnect()
