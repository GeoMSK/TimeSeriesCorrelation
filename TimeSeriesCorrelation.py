#!/usr/bin/python3
import logging
import argparse

from Dataset.TimeRangeChecker import TimeRangeChecker
from Dataset.DatasetPlotter import DatasetPlotter
from Dataset.DatasetConverter import DatasetConverter


__author__ = 'gm'


def main():
    logging.basicConfig(filename='TimeSeriesCorrelation.log', level=logging.DEBUG, filemode="w",
                        format="%(asctime)s -- %(message)s", datefmt="%d-%m-%Y %H:%M:%S")

    parser = argparse.ArgumentParser()
    parser.set_defaults(func=False)

    subparsers = parser.add_subparsers(title="subcommands", help="")
    parser_a = subparsers.add_parser('dataset2db',
                                     help="parse the given dataset and store all information to a database")

    parser_a.add_argument("dataset_file",
                          help="the dataset file")
    parser_a.add_argument("database_file",
                          help="the database name, default='dataset.db'")
    parser_a.set_defaults(func=dataset2db)

    parser_b = subparsers.add_parser('check-dates',
                                     help="check if the time-series have similar date-time range")
    parser_b.set_defaults(func=check_dates)
    parser_b.add_argument("database_file",
                          help="the database file")

    args = parser.parse_args()

    if args.func:
        args.func(args)
    else:
        parser.print_help()


def check_dates(args):
    # TimeRangeCheck(args.database_path).check()
    # TimeRangeCheck(args.database_path).print_max_date_times()
    # TimeRangeChecker(args.database_path).print_start_end_points()
    # point_dic = TimeRangeChecker(args.database_file).get_start_end_points(use_file=True)
    # datetime_pairs = []
    # for key, value in point_dic.items():
    #     datetime_pairs.append(value)
    # DatasetPlotter.plot_start_end_points(sorted(datetime_pairs))


    point_dic = TimeRangeChecker(args.database_file).get_all_points(use_file=False)
    points = []
    for key, value in point_dic.items():
        points.append(value)
    DatasetPlotter.plot_all_points(sorted(points))


def dataset2db(args):
    dc = DatasetConverter(args.dataset_file, args.database_file)
    dc.convert()

if __name__ == '__main__':
    main()