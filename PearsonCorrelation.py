import logging
import numpy as np
from Dataset.DatasetH5 import DatasetH5
from Util import calc_limit

__author__ = 'gm'


class PearsonCorrelation:
    def __init__(self, normalized_f_dataset_path: str, limit_ts_num=None, limit_ts_len=None):
        """
        :param normalized_f_dataset_path: normalized dataset path
        :param limit_ts_num: limit the number of time series to be processed. May be an integer (eg 1000) or a string (eg %70)
        in case it is a string it represents a percentage on the number of time series
        :param limit_ts_len: limit the length of the time series, similar to limit_ts_num
        """
        self.limit_ts_num = limit_ts_num
        self.limit_ts_len = limit_ts_len
        self.norm_ds_path = normalized_f_dataset_path
        self.norm_ds = DatasetH5(self.norm_ds_path)
        self.size = calc_limit(limit_ts_num, len(self.norm_ds))
        self.max_ts_len = calc_limit(limit_ts_len,
                                            len(self.norm_ds[0]))  # assuming that every ts has the same length
        self.correlation_matrix = np.zeros((self.size, self.size), dtype="float32", order="C")
        self.cache = [None] * self.size
        self.logger = logging.getLogger("PearsonCorrelation")

    def get_ts(self, i):
        if self.cache[i] is None:
            self.cache[i] = self.norm_ds[i].value[:self.max_ts_len]

        return self.cache[i]

    def find_correlations(self):
        n = self.size
        self.logger.debug("Begin correlation computation. N:%d" % n)
        for i in range(n):
            self.logger.debug("Computing %d..." % i)
            for j in range(i + 1, n):
                self.correlation_matrix[i][j] = self.corr(i, j)
        return self.correlation_matrix

    def corr(self, t1, t2):
        ts1 = self.get_ts(t1)
        ts2 = self.get_ts(t2)
        return np.average(ts1 * ts2)  # ts1, ts2 should be normalized
