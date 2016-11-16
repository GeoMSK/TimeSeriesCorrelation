import h5py
import os
import numpy as np

__author__ = 'gm'


class DatasetH5:
    def __init__(self, dataset_name):
        self.name = dataset_name
        self.ts_names = []
        assert os.path.exists(dataset_name)
        self.f = h5py.File(self.name, 'a')
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

    def compute_fourier(self, time_series, k=-1, disable_store=True):
        """
        compute the fourier transform of the given time-series, return only the k coefficients in a list.
        time-series may either be the name of the time series or the index of self.ts_names
        disable_store controls whether the computed coefficients will be store and read from the hdf5 database
        """
        assert isinstance(time_series, str) or isinstance(time_series, int)
        assert isinstance(k, int)

        if isinstance(time_series, int):
            time_series = self.ts_names[time_series]

        if not disable_store:
            coeff_db = h5py.File(self.name.rstrip(".h5") + "_coeff.h5", "a")
            ts_coeff = coeff_db.get(time_series)
            if ts_coeff is not None:
                if k > len(ts_coeff):
                    k = len(ts_coeff)
                return ts_coeff[0:k]

        d = self.f[time_series]
        fft = np.fft.fft(d, norm="ortho")
        if k > len(fft) or k == -1:
            k = len(fft)
        if not disable_store:
            coeff_db.create_dataset(time_series, (k,), data=fft[0:k], compression="gzip", compression_opts=9)
            coeff_db.close()
        return fft[0:k]

    @staticmethod
    def __get_next_power_of_2(x: int) -> int:
        return 2 ** (x - 1).bit_length()

    def close(self):
        """
        close the hdf5 database
        """
        self.f.close()
