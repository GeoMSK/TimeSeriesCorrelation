__author__ = 'gm'

import sqlite3 as sql
import logging
import os


class DatasetDatabase:
    """
    Sqlite database that holds dataset information in table "dataset" and "dataset_normalized
    Columns:
    name | tick | date | time | data1 | data2
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
                self.logger.info("connected to database \"%s\"" % self.db_name)
            else:
                self.conn = sql.connect(self.db_name)
                self._create_database()
                self.logger.info("Created database \"%s\"" % self.db_name)

    def disconnect(self):
        """
        close the sql.Connection (self.conn) if it is connected
        """
        if self.is_connected():
            self.conn.close()
            self.conn = None
            self.logger.info("Disconnected from database \"%s\"" % self.db_name)

    def _create_database(self):
        """
        create a new database with the name specified in self.db_name
        create table "dataset"
        """

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

        create_query = "CREATE TABLE dataset_normalized(" \
                       "name varchar(255)," \
                       "tick int," \
                       "date varchar(255)," \
                       "time varchar(255)," \
                       "data1 varchar(255)," \
                       "data2 varchar(255)," \
                       "CONSTRAINT mypk PRIMARY KEY (name, tick)" \
                       ");"
        c.execute(create_query)

    def store_data(self, name, tick, date, time, data1, data2, table="dataset"):
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
            store_query = "INSERT INTO %s VALUES (?,?,?,?,?,?);" % table
            try:
                c.execute(store_query, (name, tick, date, time, data1, data2))
                self.conn.commit()
            except sql.IntegrityError as e:
                self.logger.exception(e)
        else:
            raise Exception("Not connected to database")

    def store_multiple_data(self, multi_data, table="dataset"):
        """
        stores data in list multi_data to database. multi_data is of the form:
        [(name, tick, date, time, data1, data2), (...), ...]
        """
        if self.is_connected():
            assert isinstance(self.conn, sql.Connection)
            c = self.conn.cursor()
            assert isinstance(c, sql.Cursor)
            store_query = "INSERT INTO %s VALUES (?,?,?,?,?,?);" % table
            try:
                c.executemany(store_query, multi_data)
                self.conn.commit()
            except sql.IntegrityError as e:
                self.logger.exception(e)
        else:
            raise Exception("Not connected to database")

    def get_time_series(self, name):
        """
        :param name: the time-series name
        :return: None if error occurred else a sql.Cursor that can be treated as an iterator, call the
        cursor’s fetchone() method to retrieve a single matching row, or call fetchall() to get a list
        of the matching rows.
        """
        if self.is_connected():
            assert isinstance(self.conn, sql.Connection)
            c = self.conn.cursor()
            assert isinstance(c, sql.Cursor)
            query = "SELECT date, time, data1, data2 FROM dataset WHERE name=?"
            try:
                return c.execute(query, (name,))
            except sql.IntegrityError as e:
                self.logger.exception(e)
        else:
            raise Exception("Not connected to database")

        return None

    def get_distinct_names(self):
        """
        get all time-series names
        :return: a list with all time-series names
        """
        if self.is_connected():
            assert isinstance(self.conn, sql.Connection)
            c = self.conn.cursor()
            assert isinstance(c, sql.Cursor)
            query = "SELECT distinct name FROM dataset"
            try:
                return [t[0] for t in c.execute(query).fetchall()]
            except sql.IntegrityError as e:
                self.logger.exception(e)
        else:
            raise Exception("Not connected to database")

        return None

    def execute_query(self, query):
        """
        execute the specified query.
        return None if error occurred else a sql.Cursor that can be treated as an iterator, call the
        cursor’s fetchone() method to retrieve a single matching row, or call fetchall() to get a list
        of the matching rows.
        """
        if self.is_connected():
            assert isinstance(self.conn, sql.Connection)
            c = self.conn.cursor()
            assert isinstance(c, sql.Cursor)
            try:
                return c.execute(query)
            except sql.IntegrityError as e:
                self.logger.exception(e)
        else:
            raise Exception("Not connected to database")

        return None