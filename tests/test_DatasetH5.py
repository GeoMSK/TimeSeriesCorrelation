from Dataset.DatasetH5 import DatasetH5

__author__ = 'gm'


def test_init(testfiles):
    name = testfiles["h5100"]

    with DatasetH5(name) as ds:
        assert len(ds.get_ts_names()) != 0
        assert len(ds) == len(ds.get_ts_names())

        for ts in ds.get_ts_names():
            fourier = ds.compute_fourier(ts, 100)
            assert len(fourier) != 0
