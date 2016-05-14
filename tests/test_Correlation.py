import logging
from Correlation import Correlation

__author__ = 'gm'


def test_Correlation(testfiles):
     logging.basicConfig(level=logging.DEBUG)
     name = testfiles["h5100"]
     c = Correlation(name)
     correlation_matrix = c.find_correlations(1, 0.7, 20)


