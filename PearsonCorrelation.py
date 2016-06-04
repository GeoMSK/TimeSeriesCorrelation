from Dataset.DatasetH5 import DatasetH5
import numpy as np
import logging
import time

__author__ = 'gm'


class PearsonCorrelation:
    def __init__(self, normalized_f_dataset_path: str):
        """
        :param normalized_f_dataset_path: normalized dataset path
        """
        self.norm_ds_path = normalized_f_dataset_path
        self.norm_ds = DatasetH5(self.norm_ds_path)
        self.correlation_matrix = np.zeros((len(self.norm_ds), len(self.norm_ds)), dtype="float32", order="C")
        self.cache = [None] * len(self.norm_ds)
        self.logger = logging.getLogger("PearsonCorrelation")

    def get_ts(self, i):
        if self.cache[i] is None:
            self.cache[i] = self.norm_ds[i].value

        return self.cache[i]

    def find_correlations(self):
        n = len(self.norm_ds)
        self.logger.debug("Begin correlation computation. N:%d" % n)
        for i in range(n):
            self.logger.debug("Computing %d..." % i)
            for j in range(i+1, n):
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
