from Dataset.DatasetReader import DatasetReader

__author__ = 'gm'


def test_DatasetReader():
    dr = DatasetReader("./test_resources/data100.txt")
    dr.open_dataset()

    ds = {}

    # Columns:
    #   name | tick | date | time | data1 | data2

    # dr.get_next_data_averaged()
    # return (name, date, time, data1, data2)

    for i in range(0, 100):
        data = dr.get_next_data_averaged()
        name = data[0]
        date = data[1]
        time = data[2]
        data1 = data[3]
        data2 = data[4]

        if name not in ds:
            ds[name] = {date+time: [data1, data2]}
        else:
            ds[name][date+time] = [data1, data2]

    # for key, value in ds.items():
    #     print(key, value)

    assert abs(ds["Forex·EURJPY·NoExpiry"]["07/08/201500:05:12"][0] - 134.7380) <= 1e-6  # 134.738

    assert abs(ds["Forex·EURCHF·NoExpiry"]["07/08/201500:05:12"][0] - 1.042033) <= 1e-6  # 1.0420333333333334

    dr.close_dataset()
    dr.open_dataset()

    for data in dr:
        assert(isinstance(data, tuple))
