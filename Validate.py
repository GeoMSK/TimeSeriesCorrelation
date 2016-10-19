import numpy as np
import pickle
import argparse
from Dataset.DatasetH5 import DatasetH5

__author__ = 'gm'

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--fourier-approximation-file", default="fourier_approximation_correlation_matrix.pickle",
                    help="the path to the fourier approximation correlation pickle file")
parser.add_argument("-b", "--boolean-correlation-file", default="boolean_correlation_matrix.pickle",
                    help="the path to the boolean correlation pickle file")
parser.add_argument("-p", "--pearson-correlation-file", default="pearson_correlation_matrix.pickle",
                    help="the path to the pearson correlation pickle file")
parser.add_argument("-T", type=float, default=0.7,
                    help="The threshold")
parser.add_argument("-e", type=float, default=0.04,
                    help="The approximation error")

args = parser.parse_args()

fourier_approximation_file = args.fourier_approximation_file
boolean_approximation_file = args.boolean_correlation_file
pearson_correlation_file = args.pearson_correlation_file

# h5_database_file = "./test_resources/database1.h5"  # original h5 database
h5_database_file = "./database2.h5"  # original h5 database

with open(fourier_approximation_file, "rb") as f:
    fourier_approximation = pickle.load(f)

with open(boolean_approximation_file, "rb") as f:
    boolean_approximation = pickle.load(f)

with open(pearson_correlation_file, "rb") as f:
    pearson_correlation = pickle.load(f)

orig_db = DatasetH5(h5_database_file)


def normalize(ts):
    d = ts
    if np.std(d) == 0:
        data_norm = d / d
    else:
        data_norm = (d - np.mean(d)) / np.std(d)
    return data_norm


cache = [None] * len(orig_db)


def get_ts(i):
    if cache[i] is None:
        cache[i] = orig_db[i].value
    return cache[i]


def assert_pearson():
    table = pearson_correlation
    n = table.shape[0]
    print("Begin pearson correlation validation")
    for i in range(n):
        ts1 = get_ts(i)
        print("validating %d/%d" % (i + 1, n))
        for j in range(i + 1, n):
            ts2 = get_ts(j)
            corr = np.average(normalize(ts1) * normalize(ts2))
            assert table[i][j] == corr
    print("Finished pearson correlation validation\n")


def num_corr(table, T=None):
    s = 0
    n = table.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            # assert -1 <= table[i][j] <= 1
            if not (-1 <= table[i][j] <= 1):
                print("pearson[%d][%d]: %f" % (i,j, table[i][j]))
            if T:
                if table[i][j] >= T:
                    s += 1
            else:
                if table[i][j] == 1:
                    s += 1
    return s


def assert_diagonal(table):
    n = table.shape[0]
    for i in range(n):
        for j in range(i + 1):
            assert table[i][j] == 0


f_erroneous_positives = 0
f_erroneous_negatives = 0
f_false_positives = 0
f_false_negatives = 0


def assertFourier(T, e, v=False):
    global f_erroneous_positives
    global f_erroneous_negatives
    global f_false_positives
    global f_false_negatives
    fourier = fourier_approximation
    pear = pearson_correlation
    n = fourier.shape[0]

    for i in range(n):
        for j in range(i + 1, n):
            if fourier[i][j] + e < T and pear[i][j] >= T:
                f_erroneous_negatives += 1
                if v:
                    print("[%d,%d]: %f(real) %f(approx)" % (i, j, pear[i][j], fourier[i][j]))
            if fourier[i][j] - e >= T and pear[i][j] < T:
                f_erroneous_positives += 1
                if v:
                    print("[%d,%d]: %f(real) %f(approx)" % (i, j, pear[i][j], fourier[i][j]))
            if fourier[i][j] >= T and pear[i][j] < T:
                f_false_positives += 1
            elif fourier[i][j] < T and pear[i][j] >= T:
                f_false_negatives += 1


b_erroneous_positives = 0
b_erroneous_negatives = 0


def assertBoolean(T, v=False):
    global b_erroneous_positives
    global b_erroneous_negatives
    bool = boolean_approximation
    pear = pearson_correlation
    n = bool.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if bool[i][j] == 1 and pear[i][j] < T:
                b_erroneous_positives += 1
                if v:
                    print("[%d,%d]: %f  bool: %d" % (i, j, pear[i][j], bool[i][j]))
            if bool[i][j] == 0 and pear[i][j] >= T:
                b_erroneous_negatives += 1
                if v:
                    print("[%d,%d]: %f  bool: %d" % (i, j, pear[i][j], bool[i][j]))


#
#  Begin validations
#

print("diagonal check fourier...", end="")
assert_diagonal(fourier_approximation)
print(" ok")
print("diagonal check boolean...", end="")
assert_diagonal(boolean_approximation)
print(" ok")
print("diagonal check pearson...", end="")
assert_diagonal(pearson_correlation)
print(" ok")

T = args.T
e = args.e

print("Computing num_fourier...", end="")
num_fourier = num_corr(fourier_approximation, T)
print(" done")
print("Computing num_boolean...", end="")
num_boolean = num_corr(boolean_approximation)
print(" done")
print("Computing num_pearson...", end="")
num_pearson = num_corr(pearson_correlation, T)
print(" done")

print("assertFourier...")
assertFourier(T, e)
print("assertBoolean...")
assertBoolean(T)
print("assertPearson...")
assert_pearson()  # takes too long

print("")
print("Threshold T: %.4f  error e: %.4f" % (T, e))
print("Correlated pairs based on \033[1mfourier approximation\033[0m: %d\n"
      "                                false positives: %d\n"
      "                                false negatives: %d\n"
      "                                errors(pos: %d  neg: %d)\n"
      "                                %d-%d+%d = %d" %
      (num_fourier, f_false_positives, f_false_negatives, f_erroneous_positives, f_erroneous_negatives,
       num_fourier, f_false_positives, f_false_negatives, num_fourier - f_false_positives + f_false_negatives))
print("Correlated pairs based on \033[1mboolean approximation\033[0m: %d  errors(pos: %d  neg: %d)" %
      (num_boolean, b_erroneous_positives, b_erroneous_negatives))
print("Correlated pairs based on \033[1mpearson correlation\033[0m: \033[32m%d\033[0m" % num_pearson)
