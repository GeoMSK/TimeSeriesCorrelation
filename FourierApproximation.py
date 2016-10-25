from PruningMatrix import PruningMatrix
from Caching import Caching
from Dataset.DatasetH5 import DatasetH5
from Dataset.DatasetDBNormalizer import DatasetDBNormalizer
from Util import calc_limit
import numpy as np
import logging
import time
from profilehooks import profile
import pickle

__author__ = 'gm'


class FourierApproximation:
    def __init__(self, normalized_f_dataset_path: str, limit_ts_num=None, limit_ts_len=None):
        """
        :param normalized_f_dataset_path: normalized dataset path
        :param limit_ts_num: limit the number of time series to be processed. May be an integer (eg 1000) or a string (eg %70)
        in case it is a string it represents a percentage on the number of time series
        :param limit_ts_len: limit the length of the time series, similar to limit_ts_num
        """
        self.norm_ds_path = normalized_f_dataset_path
        self.norm_ds = DatasetH5(normalized_f_dataset_path)
        self.pruning_matrix = None
        """:type pruning_matrix: np.ndarray """
        self.batches = None
        """:type batches: list"""
        self.size = calc_limit(limit_ts_num, self.size)
        self.max_ts_len = calc_limit(limit_ts_len,
                                     len(self.norm_ds[0]))  # assuming that every ts has the same length
        self.correlation_matrix = np.zeros(shape=(self.size, self.size), dtype="float", order="C")
        self.norm_cache = [None] * self.size
        self.coeff_cache = [None] * self.size
        self.m = self.max_ts_len

        logging.debug("Begin computation of fourier coefficients...")
        for i in range(self.size):
            self.coeff_cache[i] = self.norm_ds.compute_fourier(i)

            # with open("pearson_correlation_matrix.pickle", "rb") as f:
            #     self.pearson = pickle.load(f)

    def __load_batch_to_cache(self, batch: list):
        """
        loads given batch to the cache
        """
        assert isinstance(batch, list)
        for ts in batch:
            self.__load_ts_to_cache(ts)

    def __load_ts_to_cache(self, ts: int):
        """
        loads given time-series to the cache
        """
        assert isinstance(ts, int)
        if self.norm_cache[ts] is None:
            self.norm_cache[ts] = self.norm_ds[ts].value

    def __clear_cache(self):
        """
        clears the cache
        """
        self.norm_cache = [None] * self.size

    def __get_pruning_matrix(self, k: int, T: float, recompute=False) -> np.ndarray:
        """
        Compute the Pruning Matrix for the given dataset in self.dataset_path. with k coefficients and T threshold.
        If the pruning matrix has been computed previously it is returned without recomputation, unless
        recompute is set to True
        """
        if self.pruning_matrix is not None and recompute is False:
            return self.pruning_matrix
        pmatrix = PruningMatrix(self.norm_ds_path)
        self.pruning_matrix = pmatrix.compute_pruning_matrix(k, T)
        return self.pruning_matrix

    def __get_batches(self, cache_capacity: int, recompute=False) -> list:
        """
        Compute the batches using the "caching strategy" described in the paper. This calls Fiduccia Mattheyses
        recursively.
        If the batches have been computed previously it is returned without recomputation, unless
        recompute is set to True
        """
        self.batches = [[x for x in range(self.size)]]  # bypass batch computation with Fiduccia
        # assert self.pruning_matrix is not None
        # if self.batches is not None and recompute is False:
        #     return self.batches
        # c = Caching(self.pruning_matrix, self.norm_ds_path, cache_capacity)
        # self.batches = c.calculate_batches()
        # return self.batches

    def find_correlations(self, k: int, T: float, B: int, e: float, recompute=False):
        """
        find correlations between timeseries of the given dataset

        :param k: the number of fourier coefficients to use in the PruningMatrix
        :type k: int
        :param T: the threshold to use in the PruningMatrix
        :type T: float
        :param B: the cache capacity for the caching strategy (capacity=number of time-series that fit in memory)
        :type B: int
        :param e: bounds the approximation error of the correlation when calculated using fourier coefficients,
         this value controls how many fourier coefficients will be used in order to guarantee that the
          approximation error will be less than e
        :type e: float
        :return: the correlation matrix
        :rtype: np.ndarray
        """
        logging.info("Begin computation of Pruning Matrix...")
        self.__get_pruning_matrix(k, T, recompute)
        n = self.pruning_matrix.shape[0]
        nets = 0
        pins = 0
        for i in range(n):
            for j in range(i + 1, n):
                if self.pruning_matrix[i][j] == 1:
                    nets += 1
                    pins += 2
        logging.info("cells: %d nets: %d pins: %d" % (n, nets, pins))
        logging.info("Begin computation of Batches...")
        self.__get_batches(B, recompute)
        logging.info("Batches computation finished. Total batches: %d" % len(self.batches))
        bno = 0
        for b in range(len(self.batches)):
            bno += 1
            logging.info("Begin processing batch %d" % bno)
            batch = self.batches[b]
            self.__load_batch_to_cache(batch)
            logging.debug("Within batch correlations....")
            # compute correlation of time-series within the batch
            for i in range(len(batch)):
                logging.debug("Processing ts %d of batch %d" % (i, bno))
                # t1 = time.time()
                ts_i = batch[i]
                for j in range(i + 1, len(batch)):
                    ts_j = batch[j]
                    self.correlation_matrix[ts_i][ts_j] = self.__correlate(ts_i, ts_j, e, T)

            logging.debug("Remaining batch correlations...")
            # fetch one by one remaining time-series in other batches and compute correlation with every
            # time-series in the current batch
            for tb in range(b + 1, len(self.batches)):  # for every remaining batch
                logging.debug("Processing remaining batch %d" % tb)
                tbatch = self.batches[tb]
                for i in range(len(tbatch)):  # for every ts in the (remaining) batch loaded
                    logging.debug("Processing ts %d of remaining batch %d" % (i, tb))
                    ts_i = tbatch[i]
                    possibly_correlated = self.__get_edges(batch, ts_i)
                    if len(possibly_correlated) > 0:
                        self.__load_ts_to_cache(ts_i)
                        for ts_j in possibly_correlated:  # for every ts in current batch that is possibly correlated with the newly cached ts
                            if self.pruning_matrix[ts_i][ts_j] == 1:  # check pruning matrix
                                self.correlation_matrix[ts_i][ts_j] = self.__correlate(ts_i, ts_j, e, T)
            self.__clear_cache()
        return self.correlation_matrix

    # @profile(filename="profiler.data", immediate="True", stdout=False)
    def __correlate(self, t1: int, t2: int, e: float, T: float) -> float:
        """
        compute the correlation between time-series t1 and t2
        """
        assert t1 is not None
        assert t2 is not None
        k, fft1, fft2 = self.compute_fourrier_coeff_for_ts_pair(t1, t2, e, T)
        if k is None:
            return 0
        assert len(fft1) == k
        assert len(fft2) == k
        approx_corr = self.__aprox_correlation(fft1, fft2)

        # if approx_corr >= 0.7+e and self.pearson[t1][t2] < 0.7 or approx_corr < 0.7-e and self.pearson[t1][t2] >= 0.7:
        #     raise Exception("[%d,%d]: %f(real) %f(approx) k: %d e:%f" % (t1, t2, self.pearson[t1][t2], approx_corr, k, e))

        return approx_corr

    def __aprox_correlation(self, fft1: np.ndarray, fft2: np.ndarray):
        return 1 - (np.linalg.norm(fft1 - fft2) ** 2)

    def __true_correlation(self, t1: int, t2: int):
        """
        compute the true correlation between two time-series. That is the pearson correlation between them
        """
        assert t1 is not None
        assert t2 is not None
        ts1 = self.norm_cache[t1]
        ts2 = self.norm_cache[t2]

        return np.average(ts1 * ts2)  # ts1 and ts2 should be already normalized

    def __get_edges(self, current_batch: list, ts: int) -> list:
        """
        find the "edges" in the Pruning Matrix that the given ts has with every other ts in current batch

        :param current_batch: the batch whose time-series will be checked for connection with given ts
        :type current_batch: list
        :param ts: the time-series to be checked for connection with ts in the batch
        :type ts: int
        :return: a list all time-series connected to the given one
        :rtype: list
        """
        edges = []
        for i in range(len(current_batch)):  # for every ts in current batch
            if self.pruning_matrix[current_batch[i]][ts] == 1:
                edges.append(current_batch[i])
        return edges

    def compute_fourrier_coeff_for_ts_pair(self, ts1: int, ts2: int, e: float, T=None):
        """
        Compute that many fourier coefficients for time-series ts1 and ts2, so that the approximation error is <= e.
        If threshold T is given then at every iteration it is checked whether the euclidean distance d^2(X,Y) exceeds
        sqrt(2m(1-T)) implying that corr(x,y) < T, if this happens then computation stops and None is returned

        :param ts1: the time series to compute the fourier coefficients for
        :type ts1: int
        :param e: the approximation error bound the user wants to set
        :type e: float
        :return: the number of coefficients needed to satisfy the error bound and the coefficients
        :rtype: int, np.ndarray, np.ndarray
        """
        k = 0
        # fft1 = self.norm_ds.compute_fourier(ts1, self.m)
        # fft2 = self.norm_ds.compute_fourier(ts2, self.m)
        fft1 = self.coeff_cache[ts1]
        fft2 = self.coeff_cache[ts2]

        assert sum(np.abs(fft1) ** 2) - 1 < 0.000001
        assert sum(np.abs(fft2) ** 2) - 1 < 0.000001
        s1 = 0
        s2 = 0
        if T:
            theta = np.sqrt(2 * (1 - T))
        while k < self.m:
            k += 1
            if T and np.linalg.norm(fft1[0:k] - fft2[0:k]) > theta:
                return None, None, None

            s1 += np.power(np.abs(fft1[k - 1]), 2)
            s2 += np.power(np.abs(fft2[k - 1]), 2)
            # logging.debug("k: %d  s1: %.6f  s2: %.6f  %.2f  m: %d" % (k, s1, s2, 1 - (e / 2), m))
            # logging.debug("\t%f >= %f" % (min(s1 * 2, s2 * 2), 1 - (e / 2)))
            if min(s1 * 2, s2 * 2) >= 1 - (e / 2):
                break
        assert k <= self.m / 2
        # logging.debug("k: " + str(k))
        return k, fft1[0:k], fft2[0:k]
