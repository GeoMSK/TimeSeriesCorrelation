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

