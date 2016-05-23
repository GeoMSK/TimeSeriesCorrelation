import numpy as np
from Dataset.DatasetDBNormalizer import DatasetDBNormalizer
from Dataset.DatasetH5 import DatasetH5

__author__ = 'gm'


def normalize(data: np.ndarray) -> np.ndarray:
    assert isinstance(data, np.ndarray)
    return DatasetDBNormalizer.normalize_time_series(data)


def corr(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    compute the pearson correlation between time-series ts1 and ts2
    """
    assert isinstance(ts1, np.ndarray)
    assert isinstance(ts2, np.ndarray)

    return np.average(normalize(ts1) * normalize(ts2))


def test_numpy_average():
    d = np.array([5, 5, 20, 10])
    assert np.average(d) == 10


def test_normalize():
    data = np.array([4, 8, 12])
    m = (4 + 8 + 12) / 3
    assert np.mean(data) == m
    s = (((4 - m) ** 2 + (8 - m) ** 2 + (12 - m) ** 2) / 3) ** 0.5
    assert np.std(data) == s
    n1 = (data - m) / s
    n2 = normalize(data)

    assert np.array_equal(n1, n2)


def test_corr():
    t1 = np.array([4, 8, 12])
    t2 = np.array([3, 7, 11])

    m1 = np.mean(t1)
    m2 = np.mean(t2)
    s1 = np.std(t1)
    s2 = np.std(t2)

    c1 = corr(t1, t2)
    c2 = (((4-m1)/s1)*((3-m2)/s2) + ((8-m1)/s1)*((7-m2)/s2) + ((12-m1)/s1)*((11-m2)/s2)) / 3

    assert c1 == c2


def test_lemma2():
    """
    corr(x,y)>=T => dk(X, Y) <= sqrt(2m(1-T))
    x,y is the original time-series
    X,Y are the fourier coefficients of the normalized time-series
    """


def test_dataset_normalization(testfiles):

    orig = DatasetH5(testfiles["database1.h5"])
    norm = DatasetH5(testfiles["dataset1_normalized.h5"])

    for i in range(len(orig)):
        ts = orig[i][:]
        t = normalize(ts)
        assert np.array_equal(t, norm[i][:])

