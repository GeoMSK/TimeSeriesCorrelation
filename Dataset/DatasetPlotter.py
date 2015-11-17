import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import matplotlib.dates as mdates

__author__ = 'gm'


class DatasetPlotter:
    def __init__(self):
        pass

    @staticmethod
    def plot_start_end_points(datetime_pairs: list):
        """
        datetime_pairs is a list of the form [start date-time, end date-time]
        a row holds the date-time of the first data, as well as the last, of a time series
        the list holds each pair for every time-series

        date-time is of type str --> "month/day/year-hours:minutes:seconds"
        """
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y\n%H:%M:%S'))
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=6))
        plt.title("Time Series Duration\n"
                  "Leftmost point of line is first point of time-series\n"
                  "Rightmost point of line is last point of time-series")
        plt.ylabel("Time Series (sorted) No")
        t = 0
        for pair in datetime_pairs:
            dates = pair
            x = [dt.datetime.strptime(d, '%m/%d/%Y-%H:%M:%S') for d in dates]
            y = [t, t]
            plt.plot(x, y, color="blue")
            t += 1

        plt.ylim([-1, t+1])

        plt.show()