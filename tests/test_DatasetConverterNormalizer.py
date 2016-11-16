import os

from Dataset.DatasetConverter import DatasetConverter
from Dataset.DatasetDatabase import DatasetDatabase
from Dataset.DatasetDBNormalizer import DatasetDBNormalizer
import pytest

__author__ = 'gm'


@pytest.mark.usefixtures("cleandir")
def test_converter(testfiles):
    dc = DatasetConverter(testfiles["data100"], "./test_database.db")
    dc.convert()

    db = DatasetDatabase("./test_database.db")
    db.connect()

    ts = db.get_time_series("Forex·EURSEK·NoExpiry")

    assert ts.fetchall() == [("07/08/2015", "00:05:12", "9.37086666666667", "1.0"),
                             ("07/08/2015", "00:05:13", "9.3714", "1.0"),
                             ("07/08/2015", "00:05:14", "9.3713", "1.0")
                             ]
    db.disconnect()
    os.remove("./test_database.db")
