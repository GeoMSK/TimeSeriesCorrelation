from Dataset.DatasetDatabase import DatasetDatabase
import sqlite3 as sql
import pytest
import os

__author__ = 'gm'


@pytest.mark.usefixtures("cleandir")
def test_DatasetDatabase():
    #
    # Part1 (insert 1 row)
    #
    test_db_filename = "test_db"
    db = DatasetDatabase(test_db_filename)
    db.connect()

    db.store_data("time-series1", 0, "11-11-2015", "19:12:00", 123.4, 1)

    assert isinstance(db.conn, sql.Connection)
    c = db.conn.cursor()
    assert isinstance(c, sql.Cursor)
    c.execute("SELECT * from dataset")
    assert c.fetchone() == ("time-series1", 0, "11-11-2015", "19:12:00", "123.4", "1")

    iterator = db.get_time_series("time-series1")
    assert iterator is not None
    for row in iterator:
        assert row == ("11-11-2015", "19:12:00", "123.4", "1")

    db.disconnect()
    assert db.conn is None

    os.remove(test_db_filename)

    #
    # Part 2 (insert multiple rows)
    #
    db = DatasetDatabase(test_db_filename)
    db.connect()

    data = [("time-series1", 1, "11-11-2015", "19:12:01", "123.5", "1"),
            ("time-series1", 2, "11-11-2015", "19:12:02", "123.6", "1"),
            ("time-series1", 3, "11-11-2015", "19:12:03", "123.7", "1"),
            ("time-series1", 4, "11-11-2015", "19:12:04", "123.8", "1"),
            ("time-series1", 5, "11-11-2015", "19:12:05", "123.9", "1"),
            ("time-series1", 6, "11-11-2015", "19:12:06", "123.5", "1")]

    db.store_multiple_data(data)

    iterator = db.get_time_series("time-series1")

    assert iterator.fetchall() == [("11-11-2015", "19:12:01", "123.5", "1"),
                                   ("11-11-2015", "19:12:02", "123.6", "1"),
                                   ("11-11-2015", "19:12:03", "123.7", "1"),
                                   ("11-11-2015", "19:12:04", "123.8", "1"),
                                   ("11-11-2015", "19:12:05", "123.9", "1"),
                                   ("11-11-2015", "19:12:06", "123.5", "1")]

    db.disconnect()
    assert db.conn is None

    os.remove(test_db_filename)
