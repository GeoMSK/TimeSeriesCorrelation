import logging
import numpy as np
from Correlation import Correlation

__author__ = 'gm'


def test_Correlation(testfiles):
    logging.basicConfig(level=logging.DEBUG)
    name = testfiles["h5100"]
    # name = "/home/george/msc/workspaces/PyCharmWorkspace/TimeSeriesCorrelation/test_resources/h5100.db"
    c = Correlation(name)
    correlation_matrix = c.find_correlations(1, 0.7, 20)

    assert True


def test_get_edges(testfiles):
    name = testfiles["h5100"]  # doesn't matter for this test
    c = Correlation(name)
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

    c.t_cache[0] = (a-np.mean(a))/np.std(a)
    c.t_cache[1] = (b-np.mean(b))/np.std(b)

    cor = (((3-m1)/s1)*((1-m2)/s2) + ((4-m1)/s1)*((2-m2)/s2)) / 2

    pearson_correlation = c._Correlation__true_correlation(0, 1)

    assert pearson_correlation == cor
