from Dataset.DatasetH5 import  DatasetH5
import numpy as np
import sys

__author__ = 'gm'

if len(sys.argv) != 4:
    print("usage: %s i j k" % sys.argv[0])
    exit()

i = int(sys.argv[1])
j = int(sys.argv[2])
k = int(sys.argv[3])
e = 0.04

ds = DatasetH5("test_resources/dataset1_normalized.h5")
n = len(ds)
m = len(ds[0])
fft1 = ds.compute_fourier(i, m)
fft2 = ds.compute_fourier(j, m)

if k == -1:
    assert sum(np.abs(fft1) ** 2) - 1 < 0.000001
    assert sum(np.abs(fft2) ** 2) - 1 < 0.000001
    s1 = 0
    s2 = 0
    k = 0
    while k < m:
        k += 1
        s1 += np.power(np.abs(fft1[k - 1]), 2)
        s2 += np.power(np.abs(fft2[k - 1]), 2)
        if min(s1 * 2, s2 * 2) >= 1 - (e / 2):
            break
    assert k <= m / 2
    fft1 = fft1[0:k]
    fft2 = fft2[0:k]

approx_corr = 1 - (np.linalg.norm(fft1 - fft2) ** 2)

print(str(approx_corr) + " k: " + str(k))
