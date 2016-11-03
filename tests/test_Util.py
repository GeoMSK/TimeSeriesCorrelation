import numpy as np
from math import isclose
from Util import euclidean_distance, euclidean_distance_squared

__author__ = 'gm'


def test_euclidean():
    l1 = np.array([1 + 3j, 2 - 1j, 4 + 2j, 2 + 8j, 4 - 3j])
    l2 = np.array([4 + 7j, 43 - 3j, 6 + 9j, 23 + 6j, 2 - 8j])

    assert euclidean_distance(l1, l2) == np.linalg.norm(l1 - l2)
    assert euclidean_distance_squared(l1, l2) == np.linalg.norm(l1 - l2) ** 2

    assert isclose(euclidean_distance(l1, l2, 3), np.linalg.norm(l1[:3] - l2[:3]))
    assert isclose(euclidean_distance_squared(l1, l2, 3), np.linalg.norm(l1[:3] - l2[:3]) ** 2)
