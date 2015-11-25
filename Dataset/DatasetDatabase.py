import datetime as dt
import numpy as np
import sqlite3 as sql
import logging
import os

__author__ = 'gm'

DATE_FORMAT = '%m/%d/%Y-%H:%M:%S'


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
        self.start_end_dates = None  # dictionary used by get_distinct_names for range filtering
        # {"time-series name": [start_datetime, end_datetime]}

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
        return self

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
            query = "SELECT date, time, data1, data2 FROM dataset WHERE name=? order by date, time"
            try:
                return c.execute(query, (name,))
            except sql.IntegrityError as e:
                self.logger.exception(e)
        else:
            raise Exception("Not connected to database")

        return None

    def get_distinct_names(self, range=None):
        """
        get all time-series names, filter by range if not None.
        range = [start_date, end_date]
        so a time series name is added only id all of its points are within start_date - end_date
        date format is '%m/%d/%Y-%H:%M:%S'
        :return: a list with all time-series names
        """
        if self.is_connected():
            assert isinstance(self.conn, sql.Connection)
            c = self.conn.cursor()
            assert isinstance(c, sql.Cursor)
            query = "SELECT distinct name FROM dataset"

            if range:
                self.start_end_dates = self.get_start_end_points()
            try:
                ts_names = [t[0] for t in c.execute(query).fetchall()]
                if range:
                    ts_names = list(filter(lambda x: self.time_series_within_range(x, range[0], range[1]), ts_names))
                return ts_names
            except sql.IntegrityError as e:
                self.logger.exception(e)
        else:
            raise Exception("Not connected to database")

        return None

    def time_series_within_range(self, ts_name, start_date, end_date):
        """
        returns true if all points of the time series with name ts_name are within start_date - end_date
        :param ts_name: the name of the time series
        :param start_date: '%m/%d/%Y-%H:%M:%S'
        :param end_date: '%m/%d/%Y-%H:%M:%S'
        :return:
        """
        if self.start_end_dates is not None:
            threshold_start_date = dt.datetime.strptime(start_date, DATE_FORMAT)
            threshold_end_date = dt.datetime.strptime(end_date, DATE_FORMAT)
            s, e = self.start_end_dates[ts_name]
            ts_start_date = dt.datetime.strptime(s, DATE_FORMAT)
            ts_end_date = dt.datetime.strptime(e, DATE_FORMAT)
            return threshold_start_date <= ts_start_date and ts_end_date <= threshold_end_date
        else:
            raise Exception("start_end_dates not populated")

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

    def get_first_datetime(self, name):
        """
        get the first date-time of the specified time-series, for which we have data
        if name is None, get the global first datetime (from all time-series)
        :param name: the time-series name
        :return: the first date-time "month/day/year-hours:minutes:seconds"
        """
        if self.is_connected():
            assert isinstance(self.conn, sql.Connection)
            c = self.conn.cursor()
            assert isinstance(c, sql.Cursor)
            if name is None:
                query = "select min(date), min(time) from dataset"
            else:
                query = "select min(date), min(time) from dataset where name=?"
            try:
                if name is None:
                    c = c.execute(query)
                else:
                    c = c.execute(query, (name,))
                res = c.fetchone()
                date_time = res[0] + "-" + res[1]
                return date_time
            except sql.IntegrityError as e:
                self.logger.exception(e)
        else:
            raise Exception("Not connected to database")

        return None

    def get_last_datetime(self, name):
        """
        get the last date-time of the specified time-series, for which we have data
        if name is None, get the global last datetime (from all time-series)
        :param name: the time-series name
        :return: the last date-time "month/day/year-hours:minutes:seconds"
        """
        self.assert_connected()
        c = self.conn.cursor()
        assert isinstance(c, sql.Cursor)
        if name is None:
            query = "select max(date), max(time) from dataset"
        else:
            query = "select max(date), max(time) from dataset where name=?"
        try:
            if name is None:
                c = c.execute(query)
            else:
                c = c.execute(query, (name,))
            res = c.fetchone()
            date_time = res[0] + "-" + res[1]
            return date_time
        except sql.IntegrityError as e:
            self.logger.exception(e)

        return None

    def assert_connected(self):
        if self.is_connected():
            assert isinstance(self.conn, sql.Connection)
        else:
            raise Exception("Not connected to database")
    
    def check(self):
        """
        NOT USED
        assert that every time series in table dataset of the database as the same min(time)
        """
        self.assert_connected()
        tsnames_list = self.db_name.get_distinct_names()

        c = self.db_name.execute_query("select min(time) from dataset where name='%s'" % tsnames_list.pop(0))
        prev = c.fetchone()
        for name in tsnames_list:
            c = self.db_name.execute_query("select min(time) from dataset where name='%s'" % name)
            cur = c.fetchone()
            assert prev == cur
            # prev = cur

    def print_min_date_times(self):
        """
        print the min (start) date-time of every time-series
        """
        self.assert_connected()
        tsnames_list = self.get_distinct_names()

        dt = {}

        for name in tsnames_list:
            date_time = self.get_first_datetime(name)
            if date_time in dt:
                dt[date_time] += 1
                continue
            else:
                dt[date_time] = 1

        for key, value in sorted(dt.items()):
            print(key + " " + str(value))

    def print_max_date_times(self):
        """
        print the max (end) date-time of every time-series
        """
        self.assert_connected()
        tsnames_list = self.get_distinct_names()

        dt = {}

        for name in tsnames_list:
            date_time = self.get_last_datetime(name)
            if date_time in dt:
                dt[date_time] += 1
                continue
            else:
                dt[date_time] = 1

        for key, value in sorted(dt.items()):
            print(key + " " + str(value))

    def print_start_end_points(self, range=None):
        """
        print the start date-time and end date-time of every time-series
        """
        for key, value in self.get_start_end_points(range=range).items():
            print(key + " " + value[0] + "---" + value[1])

    def get_start_end_points(self, range=None, use_file=False):
        """
        get the start date-time and end date-time of every time-series
        returns a dictionary of the form {"time-series name": [start_datetime, end_datetime]}
        start_datetime and end_datetime are of the form "month/day/year-hours:minutes:seconds"

        range is used to filter the time series whose points will be returned
        range = [start_date, end_date]
        """
        if not use_file or not os.path.exists("date-time-pairs"):
            self.assert_connected()
            tsnames_list = self.get_distinct_names(range=range)

            dt = {}

            for name in tsnames_list:
                c = self.execute_query("select name, min(date), min(time), max(date), max(time) from dataset "
                                          "where name='%s'" % name)
                res = c.fetchone()

                name = res[0]
                min_date_time = res[1] + "-" + res[2]
                max_date_time = res[3] + "-" + res[4]

                dt[name] = [min_date_time, max_date_time]

            if use_file:
                with open("date-time-pairs", 'w') as f:
                    for key, value in dt.items():
                        print(key + "," + value[0] + "," + value[1], file=f)
        else:
            dt = {}
            with open("date-time-pairs", 'r') as f:
                for line in f:
                    line = line[:-1]
                    split_line = line.split(",")
                    dt[split_line[0]] = [split_line[1], split_line[2]]
        return dt

    def get_all_points(self, range=None, use_file=False):
        """
        get all points of every time series. Point is a date-time that the time-series has data for
        returns a dictionary of the form {"time-series name": np.array([d1,d2,d3, ...])}
        dx are of the form "month/day/year-hours:minutes:seconds"
        dx are ordered

        range is used to filter the time series whose points will be returned
        range = [start_date, end_date]
        """
        if not use_file or not os.path.exists("all-date-time-points"):
            self.assert_connected()
            tsnames_list = self.get_distinct_names(range=range)

            dt = {}

            for name in tsnames_list:
                c = self.execute_query("select name, date, time from dataset "
                                          "where name='%s' order by date, time" % name)

                # d = self.db_name.execute_query("select count(date) from dataset "
                #                           "where name='%s'" % name).fetchone()[0]
                # d = int(d)
                for res in c:
                    name = res[0]
                    date_time = res[1] + "-" + res[2]

                    if name in dt:
                        dt[name].append(date_time)
                    else:
                        dt[name] = [date_time]
                # print(d, len(dt[name]))
                # assert d == len(dt[name])
            if use_file:
                with open("all-date-time-points", 'w') as f:
                    for key, value in dt.items():
                        print(key + "," + ",".join(value), file=f)
        else:
            dt = {}
            with open("all-date-time-points", 'r') as f:
                for line in f:
                    line = line[:-1]
                    split_line = line.split(",")
                    dt[split_line[0]] = split_line[1:]

        for key, value in dt.items():
            dt[key] = np.array(value)
        return dt
