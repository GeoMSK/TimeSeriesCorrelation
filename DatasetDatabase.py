__author__ = 'gm'

import sqlite3 as sql
import logging
import os


class DatasetDatabase:
    """
    Sqlite database that holds dataset information in table "dataset"
    Columns:
    name | tick | data | time | data1 | data2
    """

    def __init__(self, db_name):
        self.db_name = db_name
        self.conn = None
        self.logger = logging.getLogger("DatasetDatabase")

    def is_connected(self):
        return self.conn is not None

    def connect(self):
        """
        connects to database specified in self.db_name
        if it is already connected then this has no effect
        if the database does not exist it is created
        """
        if not self.is_connected():
            if os.path.exists(self.db_name) and os.path.isfile(self.db_name):
                self.conn = sql.connect(self.db_name)
            else:
                self.conn = sql.connect(self.db_name)
                self.create_database()

    def create_database(self):
        """
        create a new database with the name specified in self.db_name
        create table "dataset"
        """
        if not self.is_connected():
            self.connect()

        assert isinstance(self.conn, sql.Connection)
        c = self.conn.cursor()
        assert isinstance(c, sql.Cursor)
        create_query = "CREATE TABLE dataset(" \
                       "name varchar(255)," \
                       "tick int," \
                       "date varchar(255)," \
                       "time varchar(255)," \
                       "data1 varchar(255)," \
                       "data2 varchar(255)," \
                       "CONSTRAINT mypk PRIMARY KEY (name, tick)" \
                       ");"
        c.execute(create_query)

    def store_data(self, name, tick, date, time, data1, data2):
        """
        stores a new row in table "dataset"

        :param name: the name of the time-series
        :type name: str
        :param tick: the i-th timepoint
        :type tick: int
        :param date: the date this data arrived
        :type date: str
        :param time: the time this data arrived
        :type time: str
        :param data1: the data of this timepoint for this time-series
        :type data1: float
        :param data2: the data of this timepoint for this time-series
        :type data2: int
        """
        if self.is_connected():
            assert isinstance(self.conn, sql.Connection)
            c = self.conn.cursor()
            assert isinstance(c, sql.Cursor)
            store_query = "INSERT INTO dataset VALUES (?,?,?,?,?,?);"
            try:
                c.execute(store_query, (
                    name, tick, date, time, data1, data2))
            except sql.IntegrityError as e:
                self.logger.error(e)

            self.conn.commit()
        else:
            raise Exception("Not connected to database")

