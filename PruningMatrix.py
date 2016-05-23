import numpy as np
import matplotlib.pyplot as plt
from Dataset.DatasetH5 import DatasetH5
from Dataset.DatasetDBNormalizer import DatasetDBNormalizer

__author__ = 'gm'


class PruningMatrix:
    def __init__(self, h5dataset_name: str):
        self.h5dataset_name = h5dataset_name
        self.pruning_matrix = None

    def compute_pruning_matrix(self, k: int, T: float) -> np.ndarray:
        """
        compute the pruning matrix for the given hdf5 dataset.
        use only k fourier coefficients for every time-series to perform the computation.
        T is the threshold. returns the pruning matrix as a numpy array
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
            # dk = 0
            for i in range(N):
                # fi = fourier[i]
                for j in range(N):
                    # fj = fourier[j]
                    # for w in range(k):
                    #     dk += (fi[w] - fj[w]) * np.conjugate(fi[w] - fj[w])
                    # dk **= 1 / 2
                    dk = np.linalg.norm(fourier[i] - fourier[j])
                    t = (2 * m * (1 - T)) ** (1 / 2)
                    self.pruning_matrix[i][j] = 1 if dk <= t else 0
                    # print(str(dk) + " " + str((2 * m * (1 - T)) ** (1 / 2)))

                    with DatasetH5("database1.h5") as f:
                        plt.figure(1)

                        plt.subplot(211)
                        plt.title("corr=%.2f "
                                  "dk=%d  "
                                  "T=%.2f  "
                                  "m=%d  "
                                  "%d<=%d ??" % (np.average(DatasetDBNormalizer.normalize_time_series(f[i]) *
                                                            DatasetDBNormalizer.normalize_time_series(f[j])),
                                                 dk, T, m, dk, t))
                        plt.plot(f[i])
                        plt.plot(f[j])

                        plt.subplot(212)
                        plt.plot(ds[i])
                        plt.plot(ds[j])

                        plt.show()
                        plt.close()
        return self.pruning_matrix
