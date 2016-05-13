from Caching import Caching
from PruningMatrix import PruningMatrix
import pickle


__author__ = 'gm'


def test_Caching(testfiles):
    name = testfiles["h5100"]
    # name = "/home/george/msc/workspaces/PyCharmWorkspace/TimeSeriesCorrelation/test_resources/h5100.db"
    pm = PruningMatrix(name)
    pm.compute_pruning_matrix(1, 0.7)

    c = Caching(pm.pruning_matrix, name, 20)
    c.calculate_batches()

    for b in c.batches:
        print(b)

    assert False
