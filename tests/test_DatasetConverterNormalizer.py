import os

from Dataset.DatasetConverter import DatasetConverter
from Dataset.DatasetDatabase import DatasetDatabase
from Dataset.DatasetDBNormalizer import DatasetDBNormalizer


__author__ = 'gm'


def test_converter():
    dc = DatasetConverter("./test_resources/data100.txt", "./test_database.db")
    dc.convert()

    db = DatasetDatabase("./test_database.db")
    db.connect()

    ts = db.get_time_series("Forex·EURSEK·NoExpiry")

    assert ts.fetchall() == [("07/08/2015", "00:05:12", "9.37086666666667", "1.0"),
                             ("07/08/2015", "00:05:13", "9.3714", "1.0"),
                             ("07/08/2015", "00:05:14", "9.3713", "1.0")
                             ]

    normalizer = DatasetDBNormalizer("./test_database.db")
    normalizer.normalize()

    db.disconnect()
    os.remove("./test_database.db")
