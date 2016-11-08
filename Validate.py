import numpy as np
import pickle
import argparse
import os
import time
from Dataset.DatasetH5 import DatasetH5

__author__ = 'gm'

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--fourier-approximation-file", default="fourier_approximation_correlation_matrix.pickle",
                    help="the path to the fourier approximation correlation pickle file. "
                         "Default fourier_approximation_correlation_matrix.pickle")
parser.add_argument("-b", "--boolean-correlation-file", default="boolean_correlation_matrix.pickle",
                    help="the path to the boolean correlation pickle file. "
                         "Default boolean_correlation_matrix.pickle")
parser.add_argument("-p", "--pearson-correlation-file", default="pearson_correlation_matrix.pickle",
                    help="the path to the pearson correlation pickle file. "
                         "Default pearson_correlation_matrix.pickle")
parser.add_argument("-k", type=int, default=5,
                    help="The fourier coefficients to be used for the pruning matrix. Default 5")
parser.add_argument("-T", type=float, default=0.7,
                    help="The threshold. Default 0.7")
parser.add_argument("-e", type=float, default=0.04,
                    help="The approximation error. Default 0.04")
parser.add_argument("--original-dataset", default="./test_resources/database1.h5",
                    help="The original h5 dataset file (non normalized). Default ./test_resources/database1.h5")
parser.add_argument("--normalized-dataset", default="./test_resources/dataset1_normalized.h5",
                    help="The normalized h5 dataset file. Default ./test_resources/dataset1_normalized.h5")
parser.add_argument("--results-folder", default="results",
                    help="the folder where the results will be stored. Default results")
parser.add_argument("--skip-processing", action="store_true", default=False,
                    help="If this is set the TimeSeries Correlation does not perform any processing and "
                         "the validation will occur in existing files")
parser.add_argument("-df", action="store_true",
                    help="Disable fourier approximation correlation matrix validation")
parser.add_argument("-db", action="store_true",
                    help="Disable boolean correlation matrix validation")
parser.add_argument("-dp", action="store_true",
                    help="Disable pearson correlation matrix validation")
parser.add_argument("-spp", action="store_true",
                    help="Disable pearson correlation matrix computation")
parser.add_argument("-spb", action="store_true",
                    help="Disable boolean correlation matrix computation")
parser.add_argument("-spf", action="store_true",
                    help="Disable fourier approximation correlation matrix computation")
parser.add_argument("-v", action="store_true",
                    help="Produce more output")
parser.add_argument("-O", action="store_true",
                    help="Add the python optimization flag")
args = parser.parse_args()
results_folder = args.results_folder
if not os.path.exists(results_folder):
    os.mkdir(results_folder)

fourier_approximation_file = results_folder + "/" + args.fourier_approximation_file
boolean_approximation_file = results_folder + "/" + args.boolean_correlation_file
pearson_correlation_file = results_folder + "/" + args.pearson_correlation_file

h5_dataset_orig = args.original_dataset
h5_dataset_norm = args.normalized_dataset

k = args.k
T = args.T
e = args.e
opt = "-O" if args.O else ""
if not args.skip_processing:
    if not args.spp:
        print("Executing Pearson... ")
        t = time.time()
        cmd = "python3 %s TimeSeriesCorrelation.py corr --alg 0 -k %d -T %f -e %f --out %s %s" % (
            opt, k, T, e, pearson_correlation_file, h5_dataset_norm)
        print("cmd: " + cmd)
        os.system(cmd)
        print("time: %.3f min" % ((time.time() - t) / 60.0))
    if not args.spf:
        print("Executing Fourier... ")
        t = time.time()
        cmd = "python3 %s TimeSeriesCorrelation.py corr --alg 1 -k %d -T %f -e %f --out %s %s" % (
            opt, k, T, e, fourier_approximation_file, h5_dataset_norm)
        print("cmd: " + cmd)
        os.system(cmd)
        print("time: %.3f min" % ((time.time() - t) / 60.0))
    if not args.spb:
        print("Executing Boolean... ")
        t = time.time()
        cmd = "python3 %s TimeSeriesCorrelation.py corr --alg 2 -k %d -T %f -e %f --out %s %s" % (
            opt, k, T, e, boolean_approximation_file, h5_dataset_norm)
        print("cmd: " + cmd)
        os.system(cmd)
        print("time: %.3f min" % ((time.time() - t) / 60.0))

if not args.df:
    with open(fourier_approximation_file, "rb") as f:
        fourier_approximation = pickle.load(f)

if not args.db:
    with open(boolean_approximation_file, "rb") as f:
        boolean_approximation = pickle.load(f)

with open(pearson_correlation_file, "rb") as f:
    pearson_correlation = pickle.load(f)

orig_db = DatasetH5(h5_dataset_orig)


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
    n = table.shape[0] if isinstance(table, np.ndarray) else len(table)
    for i in range(n):
        for j in range(i + 1, n):
            # assert -1 <= table[i][j] <= 1
            if not (-1 <= table[i][j] <= 1):
                print("pearson[%d][%d]: %f" % (i, j, table[i][j]))
            if T:
                if table[i][j] >= T:
                    s += 1
            else:
                if table[i][j] == 1:
                    s += 1
    return s


def assert_diagonal(table):
    n = table.shape[0] if isinstance(table, np.ndarray) else len(table)
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
    n = bool.shape[0] if isinstance(bool, np.ndarray) else len(bool)
    for i in range(n):
        for j in range(i + 1, n):
            if bool[i][j] == 1 and pear[i][j] < T:
                b_erroneous_positives += 1
                if v:
                    print("[%d,%d]: %.8f  bool: %d" % (i, j, pear[i][j], bool[i][j]))
            if bool[i][j] == 0 and pear[i][j] >= T:
                b_erroneous_negatives += 1
                if v:
                    print("[%d,%d]: %.8f  bool: %d" % (i, j, pear[i][j], bool[i][j]))


#
#  Begin validations
#

if not args.df:
    print("diagonal check fourier...", end="")
    assert_diagonal(fourier_approximation)
    print(" ok")
if not args.db:
    print("diagonal check boolean...", end="")
    assert_diagonal(boolean_approximation)
    print(" ok")
if not args.dp:
    print("diagonal check pearson...", end="")
    assert_diagonal(pearson_correlation)
    print(" ok")

if not args.df:
    print("Computing num_fourier...", end="")
    num_fourier = num_corr(fourier_approximation, T)
    print(" done")
if not args.db:
    print("Computing num_boolean...", end="")
    num_boolean = num_corr(boolean_approximation)
    print(" done")

print("Computing num_pearson...", end="")
num_pearson = num_corr(pearson_correlation, T)
print(" done")

if not args.df:
    print("assertFourier...")
    assertFourier(T, e, args.v)
if not args.db:
    print("assertBoolean...")
    assertBoolean(T, args.v)
if not args.dp:
    print("assertPearson...")
    assert_pearson()  # takes too long

print("")
print("Threshold T: %.4f  error e: %.4f" % (T, e))
if not args.df:
    print("Correlated pairs based on \033[1mfourier approximation\033[0m: %d\n"
          "                                false positives: %d\n"
          "                                false negatives: %d\n"
          "                                errors(pos: %d  neg: %d)\n"
          "                                %d-%d+%d = %d" %
          (num_fourier, f_false_positives, f_false_negatives, f_erroneous_positives, f_erroneous_negatives,
           num_fourier, f_false_positives, f_false_negatives, num_fourier - f_false_positives + f_false_negatives))
if not args.db:
    print("Correlated pairs based on \033[1mboolean approximation\033[0m: %d  errors(pos: %d  neg: %d)" %
          (num_boolean, b_erroneous_positives, b_erroneous_negatives))
print("Correlated pairs based on \033[1mpearson correlation\033[0m: \033[32m%d\033[0m" % num_pearson)
