__author__ = 'gm'

from DatasetReader import DatasetReader
from DatasetDatabase import DatasetDatabase
import logging


class DatasetConverter:
    """
    Converts a given dataset to an sqlite database with a single table, "dataset", with columns
    name | tick | date | time | data1 | data2
    
    data with the same date-time are averaged
    TODO: missing data?
    """

    def __init__(self, dataset_name, database_name, write_buffer_size=100000):
        """
        :param dataset_name: the name of the dataset
        :param database_name: the name of the database
        :param write_buffer_size: gather this many ticks, then write to database
        """
        self.dataset = dataset_name
        self.dbname = database_name
        self.write_buffer = []  # holds data to be written to the database
        self.write_buffer_size = write_buffer_size
        # last data per time series
        # {"time-series name", [latest-data, tick]}
        self.ldpt = {}
        self.db = None
        self.dreader = None
        self.logger = logging.getLogger("DatasetConverter")

    def convert(self):
        """
        parses the dataset and stores all information in the database
        """
        assert isinstance(self.dataset, str)
        assert isinstance(self.dbname, str)

        self.dreader = DatasetReader(self.dataset)
        self.dreader.open_dataset()

        self.db = DatasetDatabase(self.dbname)
        self.db.connect()

        for data in self.dreader:
            name = data[0]
            date = data[1]
            time = data[2]

            if name not in self.ldpt:
                # first time seeing this time-series
                # keep the data in ldpt and initialize tick to zero
                self.ldpt[name] = [data, 0]
            elif self.ldpt[name][0][1] == date and self.ldpt[name][0][2] == time:
                # known time series
                # the date-time from the latest tick of this time-series is the same with the current
                # overwrite (data is incrementally averaged while read, so no need to compute avg here)
                self.ldpt[name] = [data, self.ldpt[name][1]]
            else:
                # known time series
                # - put the latest [data, tick] of this time-series from ldpt in the write buffer
                self._append_to_write_buffer(self.ldpt[name])

                # - overwrite ldpt [data, tick] with the newly arrived data and increment tick
                self.ldpt[name] = [data, self.ldpt[name][1] + 1]
        # write the last data for each time-series that are left in ldpt
        for key, value in self.ldpt.items():
            self._append_to_write_buffer(value)

        # flush the write buffer
        if len(self.write_buffer) > 0:
            self.db.store_multiple_data(self.write_buffer)
            self.write_buffer.clear()

        self.db.disconnect()
        self.dreader.close_dataset()

    def _append_to_write_buffer(self, data: list):
        """
        append data to the write buffer, if the buffer has reached a predefined size then the data are written
        into the database

        :param data: the data to append to the write buffer
                     [ [name, tick, date, time, data1, data2], tick ]
        """
        assert isinstance(data, list)
        assert isinstance(data[0], tuple)
        assert isinstance(data[1], int)

        prev_name = data[0][0]
        prev_date = data[0][1]
        prev_time = data[0][2]
        prev_data1 = data[0][3]
        prev_data2 = data[0][4]
        self.write_buffer.append([prev_name, data[1], prev_date, prev_time, prev_data1, prev_data2])

        if len(self.write_buffer) == self.write_buffer_size:
            self.db.store_multiple_data(self.write_buffer)
            self.write_buffer.clear()
