import numpy as np
from Dataset.DatasetH5 import DatasetH5

__author__ = 'gm'


class PruningMatrix:
    def __init__(self, h5dataset_name):
        self.h5dataset_name = h5dataset_name
        self.pruning_matrix = None

    def compute_pruning_matrix(self, k, T):
        """
        compute the pruning matrix for the given hdf5 dataset
        use only k fourier coefficients for every time-series to perform the computation
        T is the threshold
        """
        with DatasetH5(self.h5dataset_name) as ds:
            N = len(ds)
            m = len(ds[0])
            assert k < m / 2
            self.pruning_matrix = np.empty((N, N), dtype="b1", order='C')
            fourier = []  # [np.empty(k)] * N
            # compute fourier transform for every time-series
            for ts in range(N):
                coeff = ds.compute_fourier(ts, k)
                assert len(coeff) == k
                fourier.append(np.array(coeff))
            # compute the pruning matrix
            for i in range(N):
                for j in range(N):
                    dk = np.linalg.norm(fourier[i] - fourier[j])
                    self.pruning_matrix[i][j] = 1 if dk <= (2 * m * (1 - T)) ** (1 / 2) else 0

