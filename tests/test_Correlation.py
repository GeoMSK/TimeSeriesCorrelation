import numpy as np
from Dataset.DatasetH5 import DatasetH5
from tests.test_generic import corr, normalize

__author__ = 'gm'


def test_approx_correlation(testfiles):
    name = testfiles["dataset1_normalized.h5"]
    name2 = testfiles["database1.h5"]

    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    b = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])
    m = len(a)
    a_fft = np.fft.fft(normalize(a), norm="ortho")
    b_fft = np.fft.fft(normalize(b), norm="ortho")
    assert abs(sum(abs(a_fft) ** 2) - m) < 0.000000001 * m
    assert abs(sum(abs(b_fft) ** 2) - m) < 0.000000001 * m

    approx_corr = 1 - (np.linalg.norm(a_fft - b_fft) ** 2)/m
    assert abs(corr(a, b) - approx_corr) < 0.0000000001 * approx_corr

    orig_ds = DatasetH5(name2)
    t1 = 0
    t2 = 3
    t1_orig = orig_ds[t1][:]
    t2_orig = orig_ds[t2][:]
    m = len(t1_orig)
    t1_fft = np.fft.fft(normalize(t1_orig), norm="ortho")
    t2_fft = np.fft.fft(normalize(t2_orig), norm="ortho")

    assert abs(sum(np.abs(t1_fft) ** 2) - m) < 0.000001 * m
    assert abs(sum(np.abs(t2_fft) ** 2) - m) < 0.000001 * m

    approx_corr = 1 - (np.linalg.norm(t1_fft - t2_fft) ** 2)/2/m
    assert abs(corr(t1_orig, t2_orig) - approx_corr) < 0.0000001 * approx_corr
