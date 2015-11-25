#!/usr/bin/python3
import logging
import argparse
import datetime as dt
from Dataset.DatasetPlotter import DatasetPlotter
from Dataset.DatasetConverter import DatasetConverter
from Dataset.DatasetDatabase import DatasetDatabase
from Dataset.DatasetDB2HDF5 import DatasetDB2HDF5
from Dataset.DatasetDatabase import DATE_FORMAT
from Dataset.DatasetDBNormalizer import DatasetDBNormalizer

__author__ = 'gm'

print_max_datetimes = "print-max-datetimes"
print_min_datetimes = "print-min-datetimes"
print_start_end_datetimes = "print-start-end-datetimes"
plot_dates = "plot-dates"


def main():
    logging.basicConfig(filename='TimeSeriesCorrelation.log', level=logging.DEBUG, filemode="w",
                        format="%(asctime)s -- %(message)s", datefmt="%d-%m-%Y %H:%M:%S")

    parser = argparse.ArgumentParser()
    parser.set_defaults(func=False)

    subparsers = parser.add_subparsers(title="subcommands", help="")
    parser_dataset2db = subparsers.add_parser('dataset2db',
                                              help="parse the given dataset and store all information to a database")

    parser_dataset2db.add_argument("dataset_file",
                                   help="the dataset file")
    parser_dataset2db.add_argument("database_file",
                                   help="the database name, default='dataset.db'")
    parser_dataset2db.set_defaults(func=dataset2db)

    parser_dates = subparsers.add_parser('dates',
                                         help="plot all time-series date-time range in one graph")
    parser_dates.set_defaults(func=dates)
    parser_dates.add_argument("database_file",
                              help="the database file")
    parser_dates.add_argument("action", choices=[print_max_datetimes,
                                                 print_min_datetimes,
                                                 print_start_end_datetimes,
                                                 plot_dates],
                              help="print-max-date-times: print the last date-time for every time-series.\n"
                                   "print-min-date-times: print the first date-time for every time-series.\n"
                                   "print-start-end-points: print the first and last date-time for every time-series.\n"
                                   "plot-points: plot a line for every time-series, the leftmost point is the first "
                                   "date-time of the time-series and the rightmost is the last.")
    parser_dates.add_argument("--all", action="store_true", default=False,
                              help="plot all time series points in one graph. A point is a date-time for which"
                                   " the time-series has data")
    parser_dates.add_argument("-f", "--use-file", action="store_true", default=False,
                              help="store the query result in a file so that next time the query is not performed, "
                                   "data are read from the file")
    parser_dates.add_argument("--range", default=None,
                              help="Only time series whose points are within start_date-end_date range are considered. "
                                   "format: '%m/%d/%Y-%H:%M:%S--%m/%d/%Y-%H:%M:%S "
                                   "eg. --range '01/01/2016-00:00:00--01/01/2016-20:00:00'")

    parser_calc = subparsers.add_parser('calc',
                                        help="calculate total points if we fill each second with data. From the "
                                             "globally first date-time to the the last")
    parser_calc.set_defaults(func=calc)
    parser_calc.add_argument("database_file",
                             help="the database file")

    parser_db2h5 = subparsers.add_parser('db2h5',
                                         help="convert the database to hdf5")
    parser_db2h5.set_defaults(func=db2h5)
    parser_db2h5.add_argument("database_file",
                              help="the database file")
    parser_db2h5.add_argument("hdf5_file",
                              help="the HDF5 file")
    parser_db2h5.add_argument("-c", "--compress", type=int, default=None,
                              help="compress on the fly the HDF5 file, using gzip. Supply a number 1-9. 1 is low"
                                   "compression, 9 is high")
    parser_h5norm = subparsers.add_parser('h5norm',
                                          help="normalize a hdf5 database")
    parser_h5norm.set_defaults(func=h5norm)
    parser_h5norm.add_argument("h5database",
                               help="the database file to normalize")
    parser_h5norm.add_argument("h5normalized",
                               help="the normalizedHDF5 file")
    parser_h5norm.add_argument("-c", "--compress", type=int, default=None,
                               help="compress on the fly the HDF5 file, using gzip. Supply a number 1-9. 1 is low"
                                    "compression, 9 is high")

    args = parser.parse_args()

    if args.func:
        args.func(args)
    else:
        parser.print_help()


def dates(args):
    if args.range:
        args.range = args.range.split("--")
    if args.action == print_max_datetimes:
        DatasetDatabase(args.database_file).connect().print_max_date_times()
    elif args.action == print_min_datetimes:
        DatasetDatabase(args.database_file).connect().print_min_date_times()
    elif args.action == print_start_end_datetimes:
        DatasetDatabase(args.database_file).connect().print_start_end_points(range=args.range)
    elif args.action == plot_dates:
        if not args.all:
            point_dic = DatasetDatabase(args.database_file).connect() \
                .get_start_end_points(range=args.range, use_file=args.use_file)
            datetime_pairs = []
            for key, value in point_dic.items():
                datetime_pairs.append(value)
            DatasetPlotter.plot_start_end_points(sorted(datetime_pairs, key=lambda x: x[0] + x[-1]))
        else:
            point_dic = DatasetDatabase(args.database_file).connect() \
                .get_all_points(range=args.range, use_file=args.use_file)
            points = []
            for key, value in point_dic.items():
                points.append(value)
            DatasetPlotter.plot_all_points(sorted(points, key=lambda x: x[0] + x[-1]))


def dataset2db(args):
    dc = DatasetConverter(args.dataset_file, args.database_file)
    dc.convert()


def db2h5(args):
    conv = DatasetDB2HDF5(args.database_file, args.hdf5_file)
    if args.compress:
        conv.convert(compression_level=args.compress)
    else:
        conv.convert()


def calc(args):
    db = DatasetDatabase(args.database_file)
    db.connect()
    first_datetime = dt.datetime.strptime(db.get_first_datetime(None), DATE_FORMAT)
    last_datetime = dt.datetime.strptime(db.get_last_datetime(None), DATE_FORMAT)
    ts_names = db.get_distinct_names()
    delta = last_datetime - first_datetime
    pnum = delta.days * 3600 * 24 + delta.seconds + 1
    total_points = pnum * len(ts_names)
    print(first_datetime.strftime("%m/%d/%Y-%H:%M:%S") + " - " + last_datetime.strftime("%m/%d/%Y-%H:%M:%S"))
    print("delta: " + str(delta))
    print("points per time series: %d" % pnum)
    print("total points in interpolated dataset: " + str(total_points))
    print("Estimated size (4 bytes per point): %f MB" % (total_points * 4.0 / 1024.0 / 1024.0))

    db.disconnect()


def h5norm(args):
    DatasetDBNormalizer.normalize_hdf5(args.h5database, args.h5normalized, args.compress)


if __name__ == '__main__':
    main()
