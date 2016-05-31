import logging
import numpy as np

from Correlation import Correlation
from Dataset.DatasetH5 import DatasetH5
from tests.test_generic import corr, normalize

__author__ = 'gm'

#
# def test_Correlation(testfiles):
#     # FIXME: normalization and lemma 3 validity for constant signals
#     logging.basicConfig(level=logging.DEBUG)
#     # name = testfiles["dataset1_normalized.h5"]
#     name = testfiles["h5100_norm"]
#     name2 = testfiles["h5100"]
#
#     orig = DatasetH5(testfiles["h5100"])
#     norm = DatasetH5(testfiles["h5100_norm"])
#
#     print(orig[5][:])
#     for i in range(len(orig)):
#         ts = orig[i][:]
#         t = normalize(ts)
#         assert np.array_equal(t, norm[i][:])
#         fft = np.fft.fft(norm[i])/len(norm[i])
#         print(i)
#         assert abs(sum(np.abs(fft) ** 2) - 1) < 0.00001
#
#     c = Correlation(name, name)
#     correlation_matrix = c.find_correlations(1, 0.7, 20, 0.04)
#
#     assert True


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

    cor = (((3 - m1) / s1) * ((1 - m2) / s2) + ((4 - m1) / s1) * ((2 - m2) / s2)) / 2

    pearson_correlation = c._Correlation__true_correlation(0, 1)

    assert pearson_correlation == cor


def test_approx_correlation(testfiles):
    name = testfiles["dataset1_normalized.h5"]
    name2 = testfiles["database1.h5"]

    c = Correlation(name, name)

    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    b = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])
    m = len(a)
    a_fft = np.fft.fft(normalize(a)) / m
    b_fft = np.fft.fft(normalize(b)) / m
    assert abs(sum(abs(a_fft) ** 2) - 1) < 0.000000001
    assert abs(sum(abs(b_fft) ** 2) - 1) < 0.000000001

    approx_corr = 1 - (np.linalg.norm(a_fft - b_fft) ** 2)/2
    assert abs(corr(a, b) - approx_corr) < 0.0000000001

    orig_ds = DatasetH5(name2)
    t1 = 0
    t2 = 3
    t1_orig = orig_ds[t1][:]
    t2_orig = orig_ds[t2][:]
    m = len(t1_orig)
    t1_fft = np.fft.fft(normalize(t1_orig)) / m
    t2_fft = np.fft.fft(normalize(t2_orig)) / m

    approx_corr = 1 - (np.linalg.norm(t1_fft - t2_fft) ** 2)/2
    assert abs(corr(t1_orig, t2_orig) - approx_corr) < 0.00000001


def test_approx_correlation_error(testfiles):
    name = testfiles["dataset1_normalized.h5"]
    name2 = testfiles["database1.h5"]

    c = Correlation(name, name)
    orig_ds = DatasetH5(name2)

    e = 0.04
    for t1 in range(5):
        for t2 in range(5):
            t1_norm = c.norm_ds[t1][:]
            t2_norm = c.norm_ds[t2][:]
            m = len(t1_norm)
            t1_orig = orig_ds[t1][:]
            t2_orig = orig_ds[t2][:]
            real_corr = np.average(t1_norm * t2_norm)
            real_corr_verify = corr(t1_orig, t2_orig)
            real_corr_verify2 = np.average(normalize(t1_orig) * normalize(t2_orig))
            assert abs(real_corr - real_corr_verify) < 0.0000001
            assert abs(real_corr - real_corr_verify2) < 0.0000001

            t1_fft = np.fft.fft(t1_norm) / m
            t2_fft = np.fft.fft(t2_norm) / m

            assert abs(sum(abs(t1_fft) ** 2) - 1) < 0.00001
            assert abs(sum(abs(t2_fft) ** 2) - 1) < 0.00001

            approx_corr_all_coeff = 1 - (np.linalg.norm(t1_fft - t2_fft) ** 2)/2

            approx_corr = c._Correlation__correlate(t1, t2, e)

            print("Real correlation:   " + str(real_corr))
            print("Approx correlation: " + str(approx_corr_all_coeff) + " (all coefficients)")
            print("Approx correlation: " + str(approx_corr) + " (k coefficients)")

            assert abs(real_corr - approx_corr) <= e
