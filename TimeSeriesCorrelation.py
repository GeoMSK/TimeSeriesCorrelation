#!/usr/bin/python3

__author__ = 'gm'

import logging
import argparse
from DatasetReader import DatasetReader
from DatasetDatabase import DatasetDatabase


def dataset2db(dataset, dbname):
    """
    parses a dataset and stores all information in a database

    :param dataset: the dataset name
    :type dataset: str
    :param dbname: the database name
    :type dbname: str
    """
    assert isinstance(dataset, str)
    assert isinstance(dbname, str)

    dreader = DatasetReader(dataset)
    dreader.open_dataset()

    db = DatasetDatabase(dbname)
    db.connect()

    input_buffer = []
    # last data per time series
    ldpt = {}

    for data in dreader:
        name = data[0]
        date = data[1]
        time = data[2]

        if name not in ldpt:
            ldpt[name] = [data, 0]
        elif ldpt[name][0][1] == date and ldpt[name][0][2] == time:
            ldpt[name] = [data, ldpt[name][1]]
        else:
            # name, tick, date, time, data1, data2
            prev_name = ldpt[name][0][0]
            prev_date = ldpt[name][0][1]
            prev_time = ldpt[name][0][2]
            prev_data1 = ldpt[name][0][3]
            prev_data2 = ldpt[name][0][4]
            input_buffer.append([prev_name, ldpt[name][1], prev_date, prev_time, prev_data1, prev_data2])
            if len(input_buffer) == 100:
                db.store_multiple_data(input_buffer)
                input_buffer.clear()

            ldpt[name] = [data, ldpt[name][1] + 1]
    for key, value in ldpt.items():
        prev_name = value[0][0]
        prev_date = value[0][1]
        prev_time = value[0][2]
        prev_data1 = value[0][3]
        prev_data2 = value[0][4]
        input_buffer.append([prev_name, value[1], prev_date, prev_time, prev_data1, prev_data2])
        if len(input_buffer) == 100:
                db.store_multiple_data(input_buffer)
                input_buffer.clear()

    if len(input_buffer) > 0:
        db.store_multiple_data(input_buffer)
        input_buffer.clear()

    db.disconnect()
    dreader.close_dataset()


def main():
    logging.basicConfig(filename='TimeSeriesCorrelation.log', level=logging.DEBUG, filemode="w")
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset2db", metavar="dataset_file",
                        help="parse the given dataset and store all information to a database")
    parser.add_argument("--database", default="dataset.db", metavar="database_name",
                        help="the database name to store the dataset, default='dataset.db'")

    args = parser.parse_args()

    if args.dataset2db:
        dataset2db(args.dataset2db, args.database)


if __name__ == '__main__':
    main()