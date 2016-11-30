from Dataset.DatasetH5 import DatasetH5
from Util import calc_limit
import numpy as np
import logging
import time
import sys
from profilehooks import profile
from PearsonCorrelation import PearsonCorrelation

__author__ = 'gm'


class BooleanCorrelation:
    def __init__(self, t_dataset_path: str, validation=False, limit_ts_num=None, limit_ts_len=None):
        """
        :param t_dataset_path: original dataset path
        :param limit_ts_num: limit the number of time series to be processed. May be an integer (eg 1000) or a string (eg %70)
        in case it is a string it represents a percentage on the number of time series
        :param limit_ts_len: limit the length of the time series, similar to limit_ts_num
        """
        self.norm_ds_path = t_dataset_path
        self.norm_ds = DatasetH5(t_dataset_path)
        self.size = calc_limit(limit_ts_num, len(self.norm_ds))
        self.max_ts_len = calc_limit(limit_ts_len,
                                     len(self.norm_ds[0]))  # assuming that every ts has the same length
        self.UB = np.full(shape=(self.size, self.size), fill_value=sys.maxsize, dtype="float32",
                          order="C")
        self.LB = np.zeros(shape=(self.size, self.size), dtype="float32", order="C")
        self.CB = np.zeros(shape=(self.size, self.size), dtype="b1", order="C")

        self.cache = [None] * self.size
        self.logger = logging.getLogger("BooleanCorrelation")
        if validation:
            self.c = PearsonCorrelation(self.norm_ds_path)
        self.validation = validation

    def get_ts(self, i):
        if self.cache[i] is None:
            self.cache[i] = self.norm_ds[i].value[:self.max_ts_len]

        return self.cache[i]

    @profile(filename="boolean_profile.data", immediate="False", stdout=False)
    def boolean_approximation(self, T: float):
        m = self.max_ts_len
        n = self.size
        theta = np.sqrt(2 * m * (1 - T))

        self.logger.debug("m: %d  n: %d  theta:%f" % (m, n, theta))

        UB = self.UB
        LB = self.LB
        CB = self.CB
        d = self.d

        self.logger.debug("Processing diagonal... (n: %d)" % n)

        for i in range(n - 1):
            ed = d(i, i + 1)
            UB[i,i+1] = LB[i,i+1] = ed
            if ed <= theta:
                CB[i,i+1] = 1
                if self.validation and self.c.corr(i, i + 1) < T:
                    print("[%d,%d]:%f bool:%d  (ed)%f <= %f(theta)" %
                          (i, i + 1, self.c.corr(i, i + 1), CB[i,i+1], ed, theta))
            else:
                if self.validation and self.c.corr(i, i + 1) >= T:
                    print("[%d,%d]:%f bool:%d  (ed)%f <= %f(theta)" %
                          (i, i + 1, self.c.corr(i, i + 1), CB[i,i+1], ed, theta))
        self.logger.debug("Initial Processing of diagonal finished")
        s = 0
        total = 0
        for k in range(2, n):
            self.logger.debug("Processing diagonal %d/%d..." % (k, n - 1))
            for i in range(n - k):
                total += 1
                j = i + k
                # UB[i,j] = min([UB[i,u] + UB[u,j] for u in range(i + 1, j)])
                UB[i,j] = np.min(  UB[i, i+1:j] + UB[i+1:j, j]   )
                # LB[i,j] = max([max(LB[i,u] - UB[u,j], LB[u,j] - UB[i,u]) for u in range(i + 1, j)])
                LB1 = np.max(LB[i, i+1:j] - UB[i+1:j, j])
                LB2 = np.max(LB[i+1:j, j] - UB[i, i+1:j])
                LB[i,j] = max(LB1, LB2)
                if UB[i,j] <= theta:
                    CB[i,j] = 1
                    if self.validation and self.c.corr(i, j) < T:
                        print("[%d,%d]:%f bool:%d  (UB)%f <= %f(theta)" %
                              (i, j, self.c.corr(i, j), CB[i,j], UB[i,j], theta))
                elif LB[i,j] > theta:
                    CB[i,j] = 0
                    if self.validation and self.c.corr(i, j) >= T:
                        print("[%d,%d]:%f bool:%d  (LB)%f > %f(theta)" %
                              (i, j, self.c.corr(i, j), CB[i,j], LB[i,j], theta))
                else:
                    s += 1
                    ed = d(i, j)
                    UB[i,j] = LB[i,j] = ed
                    if ed <= theta:
                        CB[i,j] = 1
                        if self.validation and self.c.corr(i, j) < T:
                            print("[%d,%d]:%f bool:%d  (ed)%f <= %f(theta)" %
                                  (i, j, self.c.corr(i, j), CB[i,j], ed, theta))
                    else:
                        if self.validation and self.c.corr(i, j) >= T:
                            print("[%d,%d]:%f bool:%d  (ed)%f <= %f(theta)" %
                                  (i, j, self.c.corr(i, j), CB[i,j], ed, theta))
        self.logger.debug("Exact distance computations: %d/%d" % (s, total))
        return CB

    def d(self, t1: int, t2: int):
        ts1 = self.get_ts(t1)
        ts2 = self.get_ts(t2)
        euclidean_distance = np.linalg.norm(ts1 - ts2)
        return euclidean_distance
