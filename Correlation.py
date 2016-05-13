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
        if self.batches is not None and recompute is False:
            return self.batches
        c = Caching(self.pruning_matrix, self.dataset_path, cache_capacity)
        self.batches = c.calculate_batches()
        return self.batches

    def find_correlations(self):
        for b in range(len(self.batches)):
            batch = self.batches[b]
            self.__load_batch_to_cache(batch)
            # compute correlation of time-series within the batch
            for i in range(len(batch)):
                for j in range(i+1, len(batch)):
                    self.__correlate(self.cache[batch[i]], self.cache[batch[j]])

            # fetch one by one remaining time-series in other batches and compute correlation with every
            # time-series in the current batch
            for tb in range(b+1, len(self.batches)):  # for every remaining batch
                tbatch = self.batches[tb]
                for i in range(len(tbatch)):  # for every ts in the (remaining) batch loaded
                    ts_i = tbatch[i]
                    self.__load_ts_to_cache(ts_i)
                    for j in range(len(batch)):  # for every ts in current batch
                        ts_j = batch[j]
                        if self.pruning_matrix[ts_i][ts_j] is True:  # check pruning matrix
                            self.__correlate(self.cache[ts_i], self.cache[ts_j])
                self.__clear_cache()

    def __correlate(self, t1, t2):
        assert t1 is not None
        assert t2 is not None
