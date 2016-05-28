import logging
import numpy as np
from Correlation import Correlation
from tests.test_generic import corr, normalize
__author__ = 'gm'


def test_Correlation(testfiles):
    logging.basicConfig(level=logging.DEBUG)
    name = testfiles["h5100"]
    # name = "/home/george/msc/workspaces/PyCharmWorkspace/TimeSeriesCorrelation/test_resources/h5100.db"
    c = Correlation(name, name)
    correlation_matrix = c.find_correlations(1, 0.7, 20, 0.04)

    assert True


def test_get_edges(testfiles):
    name = testfiles["h5100"]  # doesn't matter for this test
    c = Correlation(name, name)
    batch = [0, 1, 2]
    c.pruning_matrix = np.array([[1, 0, 1],
                                 [0, 1, 0],
                                 [1, 0, 1]
                                 ])

    assert c._Correlation__get_edges(batch, 0) == [0, 2]
    assert c._Correlation__get_edges(batch, 1) == [1]
    assert c._Correlation__get_edges(batch, 2) == [0, 2]


def test_true_correlation(testfiles):
    name = testfiles["h5100"]  # we just need valid names to instantiate Correlation, the data is not used
    c = Correlation(name, name)

    a = np.array([3, 4])
    b = np.array([1, 2])

    m1 = np.mean(a)
    m2 = np.mean(b)
    s1 = np.std(a)
    s2 = np.std(b)

    c.norm_cache[0] = (a - np.mean(a)) / np.std(a)
    c.norm_cache[1] = (b - np.mean(b)) / np.std(b)

    cor = (((3-m1)/s1)*((1-m2)/s2) + ((4-m1)/s1)*((2-m2)/s2)) / 2

    pearson_correlation = c._Correlation__true_correlation(0, 1)

    assert pearson_correlation == cor

def test_approx_correlation(testfiles):
    name = testfiles["dataset1_normalized.h5"]

    c = Correlation(name, name)

    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    b = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])
    a_fft = np.fft.fft(normalize(a), norm="ortho")
    b_fft = np.fft.fft(normalize(b), norm="ortho")

    approx_corr = c._Correlation__aprox_correlation(a_fft, b_fft)
    assert corr(a, b) - approx_corr < 0.00001