import logging
import numpy as np
from Util import time_it
from profilehooks import profile
from Dataset.DatasetH5 import DatasetH5
from PruningMatrix import PruningMatrix
from Util import calc_limit, euclidean_distance_squared, euclidean_distance

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
        self.size = calc_limit(limit_ts_num, len(self.norm_ds))
        self.max_ts_len = calc_limit(limit_ts_len,
                                     len(self.norm_ds[0]))  # assuming that every ts has the same length
        self.correlation_matrix = np.zeros(shape=(self.size, self.size), dtype="float", order="C")
        self.norm_cache = [None] * self.size
        self.coeff_cache = [None] * self.size
        self.m = self.max_ts_len
        self.min_k = [None] * self.size

        logging.debug("Begin computation of fourier coefficients...")
        for i in range(self.size):
            self.coeff_cache[i] = self.norm_ds.compute_fourier(i)
            assert abs(sum(np.abs(self.coeff_cache[i]) ** 2) - self.m) < 0.000001 * self.m

        logging.debug("End computation of fourier coefficients...")

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
            self.norm_cache[ts] = self.norm_ds[ts].value[:self.max_ts_len]

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
        pmatrix = PruningMatrix(self.norm_ds_path, self.coeff_cache)
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

    @profile(filename="fourier_profile.data", immediate="False", stdout=False)
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
        # nets = 0
        # pins = 0
        # for i in range(n):
        #     for j in range(i + 1, n):
        #         if self.pruning_matrix[i][j] == 1:
        #             nets += 1
        #             pins += 2
        # logging.info("cells: %d nets: %d pins: %d" % (n, nets, pins))
        logging.info("Computing min_k for every time series...")
        self.__compute_k(e)
        logging.debug("Computation of min_k finished")
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
                    possibly_correlated = self.__get_edges(batch, ts_i)  # TODO: remove get_edges
                    if len(possibly_correlated) > 0:
                        self.__load_ts_to_cache(ts_i)
                        for ts_j in possibly_correlated:  # for every ts in current batch that is possibly correlated with the newly cached ts
                            self.correlation_matrix[ts_i][ts_j] = self.__correlate(ts_i, ts_j, e, T)
            self.__clear_cache()
        return self.correlation_matrix

    # @profile(filename="profiler.data", immediate="True", stdout=False)
    def __correlate(self, t1: int, t2: int, e: float, T: float) -> float:
        """
        compute the correlation between time-series t1 and t2, consulting the PruningMatrix
        """
        assert t1 is not None
        assert t2 is not None
        if self.pruning_matrix[t1, t2] == 0:  # check pruning matrix first
            return 0
        K = max(self.min_k[t1], self.min_k[t2])
        assert K > 0
        fft1 = self.coeff_cache[t1]
        fft2 = self.coeff_cache[t2]
        assert len(fft1) == len(fft2)
        theta = (1 + e - T) * self.m
        k = 1
        dist = 0
        while k <= K:
            dist += np.abs(fft1[k - 1] - fft2[k - 1]) ** 2
            if dist > theta:
                return 0
            k += 1
        return 1 - dist / self.m

        # return 1 - (np.linalg.norm(fft1 - fft2) ** 2)

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

    @time_it
    def __compute_k(self, e: float):
        """
        Compute the minimum k for every timeseries, according to the paper, that will guarantee an approximation error e
        """
        const_val = (1 - e / 2) * self.m / 2.
        for i in range(len(self.coeff_cache)):
            fft = self.coeff_cache[i]
            k = 1
            s = 0
            while k < self.m:
                s += np.power(np.abs(fft[k - 1]), 2)
                if s >= const_val:
                    break
                k += 1
            assert k <= self.m / 2
            self.min_k[i] = k
