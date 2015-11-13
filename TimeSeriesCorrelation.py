#!/usr/bin/python3

__author__ = 'gm'

import logging
import argparse
from DatasetConverter import DatasetConverter


def main():
    logging.basicConfig(filename='TimeSeriesCorrelation.log', level=logging.DEBUG, filemode="w",
                        format="%d-%m-%Y %H:%M:%S -- %(message)s")
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset2db", metavar="dataset_file",
                        help="parse the given dataset and store all information to a database")
    parser.add_argument("--database", default="dataset.db", metavar="database_name",
                        help="the database name to store the dataset, default='dataset.db'")

    args = parser.parse_args()

    if args.dataset2db:
        dc = DatasetConverter(args.dataset2db, args.database)
        dc.convert()

if __name__ == '__main__':
    main()