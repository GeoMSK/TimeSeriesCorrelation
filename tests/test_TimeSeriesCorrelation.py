from TimeSeriesCorrelation import *
import pytest
import os
__author__ = 'gm'


class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


@pytest.mark.usefixtures("cleandir")
def test_dates(testfiles):
    database_file = "dataset100"
    args = Args(action=print_max_datetimes, range=None, database_file=testfiles[database_file])
    dates(args)

    args = Args(action=print_min_datetimes, range=None, database_file=testfiles[database_file])
    dates(args)

    args = Args(action=print_start_end_datetimes, range=None, database_file=testfiles[database_file])
    dates(args)

    args = Args(action=plot_dates, range=None, all=False, use_file=False, database_file=testfiles[database_file])
    dates(args)

    args = Args(action=plot_dates, range=None, all=True, use_file=False, database_file=testfiles[database_file])
    dates(args)

    args = Args(action=plot_dates, range='07/08/2015-00:05:12--07/08/2015-00:05:13',
                all=False, use_file=False, database_file=testfiles[database_file])
    dates(args)


@pytest.mark.usefixtures("cleandir")
def test_dataset2db(testfiles):
    args = Args(dataset_file=testfiles["data100"], database_file="./test.db")
    dataset2db(args)

    assert os.path.exists("./test.db")


@pytest.mark.usefixtures("cleandir")
def test_db2h5(testfiles):
    args = Args(database_file=testfiles["dataset100"], hdf5_file="./test_hdf.db", compress=None)
    db2h5(args)

    assert os.path.exists("./test_hdf.db")


@pytest.mark.usefixtures("cleandir")
def test_calc(testfiles):
    args = Args(database_file=testfiles["dataset100"])
    calc(args)


@pytest.mark.usefixtures("cleandir")
def test_h5norm(testfiles):
    args = Args(h5database=testfiles["h5100"], h5normalized="testh5.db", compress=9)
    h5norm(args)

    assert os.path.exists("testh5.db")

