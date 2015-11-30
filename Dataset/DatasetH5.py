import h5py
import os
import numpy as np

__author__ = 'gm'


class DatasetH5():
    def __init__(self, dataset_name):
        self.name = dataset_name
        self.ts_names = []
        assert os.path.exists(dataset_name)
        self.f = h5py.File(self.name, 'r')
        for ts in self.f:
            self.ts_names.append(ts)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.f.close()

    def get_ts_names(self):
        """
        return a list with all time-series names found in the hd5 dataset
        """
        assert len(self.ts_names) != 0
        return self.ts_names

    def __len__(self):
        return len(self.f)

    def __getitem__(self, item):
        return self.f[self.ts_names[item]]

    def __iter__(self):
        return self.f.__iter__()

    def compute_fourier(self, time_series, k):
        """
        compute the fourier transform of the given time-series, return only the k coefficients in a list
        time-series may either be the name of the time series or the index of self.ts_names
        """
        assert isinstance(time_series, str) or isinstance(time_series, int)
        assert isinstance(k, int)

        if isinstance(time_series, int):
            time_series = self.ts_names[time_series]

        d = self.f[time_series]
        fft = np.fft.fft(d)  # TODO: pad with zeros to reach length power of 2, needed ??
        if k > len(fft):
            k = len(fft)
        return fft[0:k]
