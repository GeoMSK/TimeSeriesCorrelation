from PruningMatrix import PruningMatrix
from Caching import Caching
from Dataset.DatasetH5 import DatasetH5
import numpy as np
import logging

__author__ = 'gm'


class Correlation:
    def __init__(self, normalized_f_dataset_path: str, t_dataset_path: str):
        """
        :param normalized_f_dataset_path: normalized dataset in the frequency domain
        :param t_dataset_path: dataset in the time domain
        """
        self.norm_f_dataset_path = normalized_f_dataset_path
        self.t_dataset_path = t_dataset_path
        self.f_ds = DatasetH5(normalized_f_dataset_path)
        self.t_ds = DatasetH5(t_dataset_path)
        self.pruning_matrix = None
        """:type pruning_matrix: np.ndarray """
        self.batches = None
        """:type batches: list"""
        self.correlation_matrix = np.zeros(shape=(len(self.f_ds), len(self.f_ds)), dtype="float", order="C")
        self.f_cache = [None] * len(self.f_ds)
        self.t_cache = [None] * len(self.f_ds)

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
        if self.f_cache[ts] is None:
            self.f_cache[ts] = self.f_ds[ts].value
        if self.t_cache[ts] is None:
            self.t_cache[ts] = self.t_ds[ts].value

    def __clear_cache(self):
        """
        clears the cache
        """
        self.f_cache = [None] * len(self.f_ds)
        self.t_cache = [None] * len(self.f_ds)

    def __get_pruning_matrix(self, k: int, T: float, recompute=False) -> np.ndarray:
        """
        Compute the Pruning Matrix for the given dataset in self.dataset_path. with k coefficients and T threshold.
        If the pruning matrix has been computed previously it is returned without recomputation, unless
        recompute is set to True
        """
        if self.pruning_matrix is not None and recompute is False:
            return self.pruning_matrix
        pmatrix = PruningMatrix(self.norm_f_dataset_path)
        self.pruning_matrix = pmatrix.compute_pruning_matrix(k, T)
        return self.pruning_matrix

    def __get_batches(self, cache_capacity: int, recompute=False) -> list:
        """
        Compute the batches using the "caching strategy" described in the paper. This calls Fiduccia Mattheyses
        recursively.
        If the batches have been computed previously it is returned without recomputation, unless
        recompute is set to True
        """
        assert self.pruning_matrix is not None
        if self.batches is not None and recompute is False:
            return self.batches
        c = Caching(self.pruning_matrix, self.norm_f_dataset_path, cache_capacity)
        self.batches = c.calculate_batches()
        return self.batches

    def find_correlations(self, k: int, T: float, B: int, recompute=False):
        """
        find correlations between timeseries of the given dataset

        :param k: the number of fourier coefficients to use in the PruningMatrix
        :type k: int
        :param T: the threshold to use in the PruningMatrix
        :type T: float
        :param B: the cache capacity for the caching strategy (capacity=number of time-series that fit in memory)
        :type B: int
        :return: the correlation matrix
        :rtype: np.ndarray
        """
        logging.info("Begin computation of Pruning Matrix...")
        self.__get_pruning_matrix(k, T, recompute)
        logging.info("Begin computation of Batches...")
        self.__get_batches(B, recompute)
        bno = 0
        for b in range(len(self.batches)):
            bno += 1
            logging.info("Begin processing batch %d" % bno)
            batch = self.batches[b]
            self.__load_batch_to_cache(batch)
            # compute correlation of time-series within the batch
            for i in range(len(batch)):
                ts_i = batch[i]
                for j in range(i + 1, len(batch)):
                    ts_j = batch[j]
                    self.__correlate(self.f_cache[ts_i], self.f_cache[ts_j])

            # fetch one by one remaining time-series in other batches and compute correlation with every
            # time-series in the current batch
            for tb in range(b + 1, len(self.batches)):  # for every remaining batch
                tbatch = self.batches[tb]
                for i in range(len(tbatch)):  # for every ts in the (remaining) batch loaded
                    ts_i = tbatch[i]
                    possibly_correlated = self.__get_edges(batch, ts_i)
                    if len(possibly_correlated) > 0:
                        self.__load_ts_to_cache(ts_i)
                        for ts_j in possibly_correlated:  # for every ts in current batch that is possibly correlated with the newly cached ts
                            if self.pruning_matrix[ts_i][ts_j] == 1:  # check pruning matrix
                                self.__correlate(self.f_cache[ts_i], self.f_cache[ts_j])
            self.__clear_cache()
        return self.correlation_matrix

    def __correlate(self, t1: int, t2: int, e: float) -> float:
        """
        compute the correlation between time-series t1 and t2
        """
        assert t1 is not None
        assert t2 is not None
        k, fft1, fft2 = self.compute_fourrier_coeff_for_ts_pair(t1, t2, e)
        print("found k: %d" % k)

    def __true_correlation(self, t1: int, t2: int):
        """
        computer the true correlation between two time-series. That is the euclidean distance between them
        """
        assert t1 is not None
        assert t2 is not None
        ts1 = self.t_cache[t1]
        ts2 = self.t_cache[t2]
        euclidean_distance = np.linalg.norm(ts1-ts2)
        return euclidean_distance

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

    def compute_fourrier_coeff_for_ts_pair(self, ts1: int, ts2: int, e: float):
        """
        Compute that many fourier coefficients for time-series ts1 and ts2, so that the approximation error is <= e

        :param ts1: the time series to compute the fourier coefficients for
        :type ts1: int
        :param e: the approximation error bound the user wants to set
        :type e: float
        :return: the number of coefficients needed to satisfy the error bound and the coefficients
        :rtype: int, np.ndarray, np.ndarray
        """
        k = 0
        fft1 = self.f_ds.compute_fourier(ts1, self.f_ds[ts1])
        fft2 = self.f_ds.compute_fourier(ts2, self.f_ds[ts2])
        s1 = 0
        s2 = 0
        while k < max(len(self.f_ds[ts1]), len(self.f_ds[ts1])):
            k += 1
            i = k - 1
            s1 += 2 * (abs(fft1[i]) ** 2)
            s2 += 2 * (abs(fft2[i]) ** 2)
            if min(s1, s2) >= 1 - (e / 2):
                break
        return k, fft1[0:k], fft2[0:k]
