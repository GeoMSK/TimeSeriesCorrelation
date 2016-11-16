import h5py
import pytest
import datetime as dt
from Dataset.DatasetConverter import DatasetConverter
from Dataset.DatasetDB2HDF5 import DatasetDB2HDF5
from Dataset.DatasetDatabase import DatasetDatabase
from Dataset.DatasetDatabase import DATE_FORMAT

__author__ = 'gm'


@pytest.mark.usefixtures("cleandir")
def test(testfiles):
    dataset = testfiles["data10000"]
    sqlite_db = "dataset10000.db"
    h5_db = "h510000.db"

    dc = DatasetConverter(dataset, sqlite_db)
    dc.convert()

    h5conv = DatasetDB2HDF5(sqlite_db, h5_db)
    h5conv.convert()

    db = DatasetDatabase(sqlite_db)
    db.connect()

    first_datetime = dt.datetime.strptime(db.get_first_datetime(None), DATE_FORMAT)
    last_datetime = dt.datetime.strptime(db.get_last_datetime(None), DATE_FORMAT)
    delta = last_datetime - first_datetime
    pnum = delta.days * 3600 * 24 + delta.seconds + 1

    db.disconnect()

    with h5py.File(h5_db, 'r') as f:
        for name in f.keys():
            assert f[name].len() == pnum
