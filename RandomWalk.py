from enum import Enum
from numpy.random import normal
import h5py
import argparse

__author__ = 'gm'


class RandomWalkType(Enum):
    """
    Supported RandomWalk types
    """
    Gaussian = 0


rw_string2int = {"gaussian": RandomWalkType.Gaussian}
rw_int2string = {RandomWalkType.Gaussian: "gaussian"}


class RandomWalk:
    """
    This class provides Random Walk functionality. The type determines how each step is calculated in the random walk.
    All supported types should be in the RandomWalkType enumeration
    """

    def __init__(self, type=RandomWalkType.Gaussian, compression=9, **rw_args):
        """
        :param type: The RandomWalk type. Available types in RandomWalkType enumeration
        :param compression: the compression level to use for the hdf5 dataset. 0 is lowest 9 is highest
        :param rw_args: parameters to pass to the random walk function
        """
        self.type = type
        self.rw_args = rw_args
        self.compression = compression

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

        print("Generating %d signals of %d length each, using %s RandomWalk..." %
              (signals_no, signals_size, rw_int2string[self.type]))
        with h5py.File(path, "w") as d:
            for i in range(signals_no):
                signal = self.random_walk(size=signals_size, **self.rw_args)
                d.create_dataset("ts" + str(i), (len(signal),), data=signal, dtype='float32', compression="gzip",
                                 compression_opts=self.compression)
                print("Signals generated so far: %d" % (i + 1), end="\r" if i < signals_no - 1 else "\n")

        print("Generation complete")

    def random_walk(self, **kwargs):
        if self.type == RandomWalkType.Gaussian:
            return RandomWalk.gaussian_random_walk(**kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path",
                        help="The path to the HDF5 dataset to be created")
    parser.add_argument("--type", choices=["gaussian"], default="gaussian",
                        help="The type of RandomWalk to use. Available types: gaussian")
    parser.add_argument("-N", type=int, required=True,
                        help="The total number of signals to generate")
    parser.add_argument("--len", type=int, required=True,
                        help="The length of each signal")
    parser.add_argument("-m", type=float, default=0.0,
                        help="Mean (“centre”) of the distribution. Applicable only to Gaussian RandomWalk. Default 0")
    parser.add_argument("-s", type=float, default=10.0,
                        help="Standard deviation (spread or “width”) of the distribution. "
                             "Applicable only to Gaussian RandomWalk. Default 10")
    parser.add_argument("-c", type=int, default=9,
                        help="Compression level. 0 is the lowest 9 is the highest")

    args = parser.parse_args()
    rw = RandomWalk(type=rw_string2int[args.type], compression=args.c, m=args.m, s=args.s)
    rw.generate_dataset(args.path, args.N, args.len)
