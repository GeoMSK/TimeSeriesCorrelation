from Dataset.DatasetH5 import DatasetH5
from enum import Enum
from numpy.random import normal
__author__ = 'gm'


class RandomWalkType(Enum):
    """
    Supported RandomWalk types
    """
    Gaussian = 0


class RandomWalk:
    """
    This class provides Random Walk functionality. The type determines how each step is calculated in the random walk.
    All supported types should be in the RandomWalkType enumeration
    """
    def __init__(self, type=RandomWalkType.Gaussian):
        self.type = type

    @staticmethod
    def gaussian_random_walk(m: float, s: float, size=None):
        """
        :param m: Mean (“centre”) of the distribution
        :param s: Standard deviation (spread or “width”) of the distribution
        :param size: Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn.
        Default is None, in which case a single value is returned
        :return: samples from the normal distribution, as defined by the parameters given
        """
        return normal(m, s, size)

    def generate_dataset(self, path: str, signals_no: int, signals_size: int):
        """
        Generate an hdf5 dataset with signals generated using RandomWalk
        :param path: the path to the dataset
        :param signals_no: the number of signals to generate
        :param signals_size: the size of each signal
        """
        # TODO: implement
        pass