__author__ = 'gm'

from DatasetReader import DatasetReader


def test_DatasetReader():
    dr = DatasetReader("./resources/data.txt")
    dr.open_dataset()

    for i in range(0, 100):
        print(dr.get_next_data_averaged())

    assert False