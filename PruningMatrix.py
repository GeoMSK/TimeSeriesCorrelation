import logging
from itertools import count

import numpy as np
from Util import time_it
from math import sqrt
from Dataset.DatasetH5 import DatasetH5

__author__ = 'gm'


class PruningMatrix:
    def __init__(self, h5dataset_name: str, coeff_cache=None):
        self.h5dataset_name = h5dataset_name
        self.pruning_matrix = None
        self.coeff_cache = coeff_cache

    @time_it
    def compute_pruning_matrix(self, k: int, T: float, disable_store=False) -> np.ndarray:
        """
        compute the pruning matrix for the given hdf5 dataset.
        use only k fourier coefficients for every time-series to perform the computation.
        T is the threshold. returns the pruning matrix as a numpy array
        """
        logging.debug("Computing PruningMatrix for k=%d" % k)
        if self.coeff_cache:
            N = len(self.coeff_cache)
            m = len(self.coeff_cache[0])
            assert k < m / 2
            self.pruning_matrix = np.zeros((N, N), dtype="b1", order='C')

            fourier = np.empty((N, k), dtype=np.complex_)

            # get fourier transform from coeff_cache
            for ts in range(N):
                fourier[ts, :] = self.coeff_cache[ts][:k]
        else:
            with DatasetH5(self.h5dataset_name) as ds:
                N = len(ds)
                m = len(ds[0])
                assert k < m / 2
                self.pruning_matrix = np.zeros((N, N), dtype="b1", order='C')

                fourier = np.empty((N, k), dtype=np.complex_)

                # compute fourier transform for every time-series
                for ts in range(N):
                    coeff = ds.compute_fourier(ts, k)
                    assert len(coeff) == k
                    fourier[ts, :] = np.array(coeff, dtype=np.complex_)

        logging.debug("Begin second part of PruningMatrix Computation")
        # compute the pruning matrix
        # dk = 0            
        t = sqrt(2 * m * (1 - T))
        countPairs = 0
        for i in range(N):
            j = i
            self.pruning_matrix[i, j] = True
        for i in range(N):
            print("PruningMatrix: computing %d/%d" % (i + 1, N), end="\r")
            for j in range(i, N):
                # if i == j:
                #     self.pruning_matrix[i, i] = True
                #     continue

                # dk = np.linalg.norm(fourier[i] - fourier[j])
                bval = 1  # (dk <= t)
                self.pruning_matrix[j, i] = self.pruning_matrix[i, j] = bval
                if bval: countPairs += 1
        print("")

        total = N * (N - 1) / 2
        logging.debug("There have been %d out of %d, %.2f %% pruning" % (countPairs, total,
                                                                         (total - countPairs) / total * 100))
        return self.pruning_matrix
