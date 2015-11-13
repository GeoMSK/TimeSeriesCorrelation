__author__ = 'gm'

import io
import logging


class DatasetReader:
    """
    Provides functionality to read data from a specific dataset
    """

    def __init__(self, dataset_path: str):
        """
        :param dataset_path: the path to the dataset we want read data
        """
        self.dataset_path = dataset_path
        self.dataset_handle = None
        """:type: io.TextIOWrapper"""
        self.time_buffer = {}
        self.logger = logging.getLogger("DatasetReader")

    def open_dataset(self):
        """
        opens the dataset specified in dataset_path for reading
        """
        if self.dataset_handle is None:
            self.dataset_handle = open(self.dataset_path, 'r')
            self.logger.info("Open dataset file \"%s\" for reading" % self.dataset_path)

    def close_dataset(self):
        """
        closes the dataset specified in dataset_path
        """
        if self.dataset_handle is not None:
            self.dataset_handle.close()
            self.dataset_handle = None
            self.logger.info("Close dataset file \"%s\"" % self.dataset_path)

    def __iter__(self):
        return self

    def __next__(self):
        if self.dataset_handle is None:
            raise StopIteration
        t = self.get_next_data_averaged()
        if t is None:
            raise StopIteration
        else:
            return t

    def get_next_data(self):
        """
        get the data of the next line in the dataset

        :return: (name, date, time, data1, data2) or None if reached EOF
        :rtype: tuple
        """
        assert isinstance(self.dataset_handle, io.TextIOWrapper)
        line = self.dataset_handle.readline().rstrip('\n')
        if line == "":
            return None
        name, date, time, data1, data2 = line.split(",")
        return name, date, time, float(data1), float(data2)

    def get_next_data_averaged(self):
        """
        get the data of the next line in the dataset
        if the date time in the data fetched is exactly the same as the previous fetch of the same time-series
        then the new data1, data2 are averaged with the previous values. The average is computed in an incremental
        manner

        :return: (name, date, time, data1, data2) or None if reached EOF
        :rtype: tuple
        """
        t = self.get_next_data()
        if t is None:
            return None
        name, date, time, data1, data2 = t
        if name not in self.time_buffer:
            # first time seeing this time-series
            self.time_buffer[name] = [date, time, data1, data2, 1]
        else:
            # known time-series, check for same timepoint
            temp_data = self.time_buffer[name]
            """:type: list """
            if temp_data[0] == date and temp_data[1] == time:
                # same time point, average data (incremental average)
                n = temp_data[4] + 1
                old_avg1 = float(temp_data[2])
                old_avg2 = float(temp_data[3])
                avg1 = (n - 1) * old_avg1 / n + data1 / n
                avg2 = (n - 1) * old_avg2 / n + data2 / n
                temp_data[2] = avg1
                temp_data[3] = avg2
                temp_data[4] = n

                data1 = avg1
                data2 = avg2
            else:
                # overwrite old record, we have new timepoint
                self.time_buffer[name] = [date, time, data1, data2, 1]

        return name, date, time, data1, data2
