from Dataset.DatasetH5 import DatasetH5
import numpy as np
import logging
import time
import sys
from PearsonCorrelation import PearsonCorrelation

__author__ = 'gm'


class BooleanCorrelation:
    def __init__(self, t_dataset_path: str, validation=False):
        """
        :param t_dataset_path: original dataset path
        """
        self.norm_ds_path = t_dataset_path
        self.norm_ds = DatasetH5(t_dataset_path)
        self.UB = np.full(shape=(len(self.norm_ds), len(self.norm_ds)), fill_value=sys.maxsize, dtype="float32",
                          order="C")
        self.LB = np.zeros(shape=(len(self.norm_ds), len(self.norm_ds)), dtype="float32", order="C")
        self.CB = np.zeros(shape=(len(self.norm_ds), len(self.norm_ds)), dtype="b1", order="C")
        self.cache = [None] * len(self.norm_ds)
        self.logger = logging.getLogger("Correlation2")
        if validation:
            self.c = PearsonCorrelation(self.norm_ds_path)
        self.validation = validation

    def get_ts(self, i):
        if self.cache[i] is None:
            self.cache[i] = self.norm_ds[i].value

        return self.cache[i]

    def boolean_approximation(self, T: float):
        m = len(self.norm_ds[0])
        n = len(self.norm_ds)
        theta = np.sqrt(2 * m * (1 - T))

        self.logger.debug("m: %d  n: %d  theta:%f" % (m, n, theta))

        UB = self.UB
        LB = self.LB
        CB = self.CB
        d = self.d

        self.logger.debug("Processing diagonal... (n: %d)" % n)

        for i in range(n - 1):
            # self.logger.debug("Processing %d,%d..." % (i, i + 1))
            ed = d(i, i + 1)
            UB[i, i + 1] = LB[i, i + 1] = ed
            # self.logger.debug("%f <= %f" % (ed, theta))
            if ed <= theta:
                CB[i, i + 1] = 1
                if self.validation and self.c.corr(i, i + 1) < T:
                    print("[%d,%d]:%f bool:%d  (ed)%f <= %f(theta)" %
                          (i, i + 1, self.c.corr(i, i + 1), CB[i, i + 1], ed, theta))
            else:
                if self.validation and self.c.corr(i, i + 1) >= T:
                    print("[%d,%d]:%f bool:%d  (ed)%f <= %f(theta)" %
                          (i, i + 1, self.c.corr(i, i + 1), CB[i, i + 1], ed, theta))
        self.logger.debug("Initial Processing of diagonal finished")
        s = 0
        total = 0
        for k in range(2, n):
            self.logger.debug("Processing diagonal %d/%d..." % (k, n - 1))
            for i in range(n - k):
                total += 1
                j = i + k
                UB[i, j] = min([UB[i, u] + UB[u, j] for u in range(i + 1, j)])
                LB[i, j] = max([max(LB[i, u] - UB[u, j], LB[u, j] - UB[i, u]) for u in range(i + 1, j)])
                if UB[i, j] <= theta:
                    CB[i, j] = 1
                    if self.validation and self.c.corr(i, j) < T:
                        print("[%d,%d]:%f bool:%d  (UB)%f <= %f(theta)" %
                              (i, j, self.c.corr(i, j), CB[i, j], UB[i, j], theta))
                elif LB[i, j] > theta:
                    CB[i, j] = 0
                    if self.validation and self.c.corr(i, j) >= T:
                        print("[%d,%d]:%f bool:%d  (LB)%f > %f(theta)" %
                              (i, j, self.c.corr(i, j), CB[i, j], LB[i, j], theta))
                else:
                    s += 1
                    ed = d(i, j)
                    UB[i, j] = LB[i, j] = ed
                    if ed <= theta:
                        CB[i, j] = 1
                        if self.validation and self.c.corr(i, j) < T:
                            print("[%d,%d]:%f bool:%d  (ed)%f <= %f(theta)" %
                                  (i, j, self.c.corr(i, j), CB[i, j], ed, theta))
                    else:
                        if self.validation and self.c.corr(i, j) >= T:
                            print("[%d,%d]:%f bool:%d  (ed)%f <= %f(theta)" %
                                  (i, j, self.c.corr(i, j), CB[i, j], ed, theta))
        self.logger.debug("Exact distance computations: %d/%d" % (s, total))
        self.logger.debug("Avg Euclidean distance computation time: %.3f ms" % (BooleanCorrelation.avg * 1000))
        return CB

    avg = 0
    n = 0

    def d(self, t1: int, t2: int):
        BooleanCorrelation.n += 1
        ts1 = self.get_ts(t1)
        ts2 = self.get_ts(t2)
        begin = time.time()
        euclidean_distance = np.linalg.norm(ts1 - ts2)
        end = time.time()
        dur = end - begin
        BooleanCorrelation.avg = (BooleanCorrelation.n - 1) * BooleanCorrelation.avg / BooleanCorrelation.n + dur / BooleanCorrelation.n

        return euclidean_distance
