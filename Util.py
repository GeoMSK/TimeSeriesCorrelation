__author__ = 'gm'

import numpy as np


def calc_limit(limit, num: int) -> int:
    """
    limits the number "num" based on the given limit
    :param limit: this is the limit to enforce on num, it may be an integer or a string. In case of an integer
    num is set to limit, or is left untouched if limit>num. In case of a string (eg format: %70) num is set to
    the percentage indicated by limit, in the example given it will be num * 0,7. If this is None the num is returned
    :param num: the number to limit
    :return: the num with the limit applied, this will be an int at all cases
    """
    ret = None
    if limit is None:
        ret = num
    elif isinstance(limit, int):
        ret = num if limit > num else limit
    elif isinstance(limit, str):
        if limit[0] == "%":
            l = float(limit[1:]) / 100
            assert 0 <= l <= 1
            ret = round(num * l)
        else:
            ret = num if int(limit) > num else int(limit)
    return ret


def euclidean_distance(l1: np.ndarray, l2: np.ndarray, k=None) -> float:
    """
    Calculate the euclidean distance between l1 and l2. Limit their size to k, if k is specified
    :return: the euclidean distance of l1[0:k] and l2[0:k] if k is given or l1 and l2 if not
    """
    assert len(l1) == len(l2)
    s = 0
    if k is None or k > len(l1):
        k = len(l1)
    for i in range(k):
        s += np.abs(l1[i] - l2[i]) ** 2  # np.abs is needed for complex numbers
    return np.sqrt(s)


def euclidean_distance_squared(l1: np.ndarray, l2: np.ndarray, k=None) -> float:
    """
    Calculate the squared euclidean distance between l1 and l2. Limit their size to k, if k is specified
    :return: the euclidean distance of l1[0:k] and l2[0:k] if k is given or l1 and l2 if not
    """
    assert len(l1) == len(l2)
    s = 0
    if k is None or k > len(l1):
        k = len(l1)
    for i in range(k):
        s += np.abs(l1[i] - l2[i]) ** 2  # np.abs is needed for complex numbers
    return s
