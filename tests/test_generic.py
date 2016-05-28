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
    c2 = (((4 - m1) / s1) * ((3 - m2) / s2) + ((8 - m1) / s1) * ((7 - m2) / s2) + ((12 - m1) / s1) * (
        (11 - m2) / s2)) / 3

    assert c1 == c2


def test_dataset_normalization(testfiles):
    orig = DatasetH5(testfiles["database1.h5"])
    norm = DatasetH5(testfiles["dataset1_normalized.h5"])

    for i in range(len(orig)):
        ts = orig[i][:]
        t = normalize(ts)
        assert np.array_equal(t, norm[i][:])


def test_euclidean_distance_preserved_by_FFT(testfiles):
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    b = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])
    a_fft = np.fft.fft(a, norm="ortho")
    b_fft = np.fft.fft(b, norm="ortho")

    # euclidean_distance of a and b using np.linalg.norm
    d1 = np.linalg.norm(a - b)
    # euclidean distance of a_fft and b_fft using np.linalg.norm
    d2 = np.linalg.norm(a_fft - b_fft)
    assert d1 == d2

    norm = DatasetH5(testfiles["dataset1_normalized.h5"])

    ts1 = norm[0][:]
    ts2 = norm[1][:]

    f1 = norm.compute_fourier(0, len(ts1), disable_store=True)
    f2 = norm.compute_fourier(1, len(ts2), disable_store=True)
    assert len(ts1) == len(f1)
    assert len(ts2) == len(f2)
    assert np.linalg.norm(ts1 - ts2) - np.linalg.norm(f1 - f2) < 0.1


def test_lemma2(testfiles):
    """
    corr(x,y)>=T => dk(X, Y) <= sqrt(2m(1-T))
    x,y is the original time-series
    X,Y are the fourier coefficients of the normalized time-series
    """
    orig = DatasetH5(testfiles["database1.h5"])
    norm = DatasetH5(testfiles["dataset1_normalized.h5"])

    T = 0.5
    k = 5
    m = len(orig[0])
    const = np.sqrt(2 * m * (1 - T))
    assert k <= 2 * m
    # for i in range(len(orig)):  #  takes too long to complete
    for i in range(1):
        for j in range(i + 1, len(orig)):
            c = corr(orig[i][:], orig[j][:])
            if c >= T:
                fi = norm.compute_fourier(i, k)
                fj = norm.compute_fourier(j, k)
                dk = np.linalg.norm(fi - fj)
                assert dk <= const
