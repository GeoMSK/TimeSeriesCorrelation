from Dataset.DatasetH5 import DatasetH5
import numpy as np
import logging
import time

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
        self.size = self.__calc_limit(limit_ts_num, len(self.norm_ds))
        self.max_ts_len = self.__calc_limit(limit_ts_len,
                                            len(self.norm_ds[0]))  # assuming that every ts has the same length
        self.correlation_matrix = np.zeros((self.size, self.size), dtype="float32", order="C")
        self.cache = [None] * self.size
        self.logger = logging.getLogger("PearsonCorrelation")

    @staticmethod
    def __calc_limit(limit, num: int) -> int:
        """
        limits the number "num" based on the given limit
        :param limit: this is the limit to enforce on num, it may be an integer or a string. In case of an integer
        num is set to limit, or is left untouched if limit>num. In case of a string (eg format: %70) num is set to
        the percentage indicated by limit, in the example given it will be num * 0,7. If this is None the num is returned
        :param num: the number to limit
        :return: the num with the limit applied, this will be an int at all cases
        """
        ret = None
        if limit is None:
            ret = num
        elif isinstance(limit, int):
            ret = num if limit > num else limit
        elif isinstance(limit, str):
            if limit[0] == "%":
                l = float(limit[1:]) / 100
                assert 0 <= l <= 1
                ret = round(num * l)
            else:
                ret = num if int(limit) > num else int(limit)
        return ret

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
        self.logger.debug("Avg Pearson Correlation computation time: %.3f ms" % (PearsonCorrelation.avg * 1000))
        return self.correlation_matrix

    avg = 0
    n = 0

    def corr(self, t1, t2):
        PearsonCorrelation.n += 1
        ts1 = self.get_ts(t1)
        ts2 = self.get_ts(t2)
        begin = time.time()
        pearson_correlation = np.average(ts1 * ts2)  # ts1, ts2 should be normalized
        end = time.time()
        dur = end - begin
        PearsonCorrelation.avg = (PearsonCorrelation.n - 1) * PearsonCorrelation.avg / PearsonCorrelation.n + \
                                 dur / PearsonCorrelation.n

        return pearson_correlation
