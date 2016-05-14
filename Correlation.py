from PruningMatrix import PruningMatrix
from Caching import Caching
from Dataset.DatasetH5 import DatasetH5
import numpy as np

__author__ = 'gm'


class Correlation:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.ds = DatasetH5(dataset_path)
        self.pruning_matrix = None
        """:type pruning_matrix: np.ndarray """
        self.batches = None
        """:type batches: list"""
        self.correlation_matrix = np.zeros(shape=(len(self.ds), len(self.ds)), dtype="float", order="C")
        self.cache = [None] * len(self.ds)

    def __load_batch_to_cache(self, batch: list):
        assert isinstance(batch, list)
        for ts in batch:
            self.__load_ts_to_cache(ts)

    def __load_ts_to_cache(self, ts: int):
        assert isinstance(ts, int)
        if self.cache[ts] is not None:
            self.cache[ts] = self.ds[ts].value

    def __clear_cache(self):
        self.cache = [None] * len(self.ds)

    def __get_pruning_matrix(self, k: int, T: float, recompute=False) -> np.ndarray:
        """
        Compute the Pruning Matrix for the given dataset in self.dataset_path. with k coefficients and T threshold.
        If the pruning matrix has been computed previously it is returned without recomputation, unless
        recompute is set to True
        """
        if self.pruning_matrix is not None and recompute is False:
            return self.pruning_matrix
        pmatrix = PruningMatrix(self.dataset_path)
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
        c = Caching(self.pruning_matrix, self.dataset_path, cache_capacity)
        self.batches = c.calculate_batches()
        return self.batches

    def find_correlations(self, k: int, T: float, B: int):
        """
        find correlations between timeseries of the given dataset
        :param k: the number of fourier coefficients to use
        in the PruningMatrix
        :type k: int
        :param T: the threshold to use in the PruningMatrix
        :type T: float
        :param B: the cache capacity for the caching strategy (capacity=number of time-series that fit in memory)
        :type B: int
        :return: the correlation matrix
        :rtype: np.ndarray
        """
        self.__get_pruning_matrix(k, T)
        self.__get_batches(B)
        for b in range(len(self.batches)):
            batch = self.batches[b]
            self.__load_batch_to_cache(batch)
            # compute correlation of time-series within the batch
            for i in range(len(batch)):
                for j in range(i + 1, len(batch)):
                    self.__correlate(self.cache[batch[i]], self.cache[batch[j]])

            # fetch one by one remaining time-series in other batches and compute correlation with every
            # time-series in the current batch
            for tb in range(b + 1, len(self.batches)):  # for every remaining batch
                tbatch = self.batches[tb]
                for i in range(len(tbatch)):  # for every ts in the (remaining) batch loaded
                    ts_i = tbatch[i]
                    possibly_correlated = self.__get_edges(batch, ts_i)
                    if len(possibly_correlated) > 0:
                        self.__load_ts_to_cache(ts_i)
                        for j in possibly_correlated:  # for every ts in current batch that is possibly correlated with the newly cached ts
                            ts_j = batch[j]
                            if self.pruning_matrix[ts_i][ts_j] == 1:  # check pruning matrix
                                self.__correlate(self.cache[ts_i], self.cache[ts_j])
                self.__clear_cache()
        return self.correlation_matrix

    def __correlate(self, t1, t2):
        assert t1 is not None
        assert t2 is not None

    def __get_edges(self, current_batch, ts) -> list:
        edges = []
        for j in range(len(current_batch)):  # for every ts in current batch
            if self.pruning_matrix[current_batch[j]][ts] == 1:
                edges.append(ts)
        return edges
