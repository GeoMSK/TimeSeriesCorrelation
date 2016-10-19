from Dataset.DatasetH5 import  DatasetH5
import numpy as np
import sys

__author__ = 'gm'

if len(sys.argv) != 3:
    print("usage: %s i j" % sys.argv[0])
    exit()

i = int(sys.argv[1])
j = int(sys.argv[2])

ds = DatasetH5("test_resources/dataset1_normalized.h5")
n = len(ds)
m = len(ds[0])

ts1 = ds[i][:]
ts2 = ds[j][:]


real_corr = np.average(ts1*ts2)

print("%.16f" % real_corr)
