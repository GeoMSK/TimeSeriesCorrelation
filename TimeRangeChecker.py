from DatasetDatabase import DatasetDatabase
import os

__author__ = 'gm'


class TimeRangeChecker:
    def __init__(self, db_name):
        self.db_name = db_name
        self.db = None

    def check(self):
        self.db = DatasetDatabase(self.db_name)
        self.db.connect()
        tsnames_list = self.db.get_distinct_names()

        c = self.db.execute_query("select min(time) from dataset where name='%s'" % tsnames_list.pop(0))
        prev = c.fetchone()
        for name in tsnames_list:
            c = self.db.execute_query("select min(time) from dataset where name='%s'" % name)
            cur = c.fetchone()
            assert prev == cur
            # prev = cur

        self.db.disconnect()

    def print_min_date_times(self):
        """
        print the min (start) date-time of every time-series
        """
        self.db = DatasetDatabase(self.db_name)
        self.db.connect()
        tsnames_list = self.db.get_distinct_names()

        dt = {}

        for name in tsnames_list:
            c = self.db.execute_query("select min(date), min(time) from dataset where name='%s'" % name)
            res = c.fetchone()
            date_time = res[0] + "-" + res[1]
            if date_time in dt:
                dt[date_time] += 1
                continue
            else:
                dt[date_time] = 1
                print(date_time)
        self.db.disconnect()

    def print_max_date_times(self):
        """
        print the max (end) date-time of every time-series
        """
        self.db = DatasetDatabase(self.db_name)
        self.db.connect()
        tsnames_list = self.db.get_distinct_names()

        dt = {}

        for name in tsnames_list:
            c = self.db.execute_query("select max(date), max(time) from dataset where name='%s'" % name)
            res = c.fetchone()
            date_time = res[0] + "-" + res[1]
            if date_time in dt:
                dt[date_time] += 1
                continue
            else:
                dt[date_time] = 1

        for key, value in sorted(dt.items()):
            print(key + " " + str(value))
        self.db.disconnect()

    def print_start_end_points(self):
        """
        print the start date-time and end date-time of every time-series
        """
        for key, value in self.get_start_end_points():
            print(key + " " + value[0] + "---" + value[1])

    def get_start_end_points(self, use_file=False):
        """
        get the start date-time and end date-time of every time-series
        returns a dictionary of the form {"time-series name": [start_datetime, end_datetime]}
        start_datetime and end_datetime are of the form "month/day/year-hours:minutes:seconds"
        """
        if not use_file or not os.path.exists("date-time-pairs"):
            self.db = DatasetDatabase(self.db_name)
            self.db.connect()
            tsnames_list = self.db.get_distinct_names()

            dt = {}

            for name in tsnames_list:
                c = self.db.execute_query("select name, min(date), min(time), max(date), max(time) from dataset "
                                          "where name='%s'" % name)
                res = c.fetchone()

                name = res[0]
                min_date_time = res[1] + "-" + res[2]
                max_date_time = res[3] + "-" + res[4]

                dt[name] = [min_date_time, max_date_time]

            self.db.disconnect()
            if use_file:
                with open("date-time-pairs", 'w') as f:
                    for key, value in dt:
                        print(key + "," + value[0] + "," + value[1], file=f)
        else:
            dt = {}
            with open("date-time-pairs", 'r') as f:
                for line in f:
                    line = line[:-1]
                    split_line = line.split(",")
                    dt[split_line[0]] = [split_line[1], split_line[2]]
        return dt
