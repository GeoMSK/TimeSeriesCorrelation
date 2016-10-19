import pytest
import tempfile
import os

__author__ = 'gm'


@pytest.fixture()
def cleandir(request):
    curdir = os.getcwd()
    newpath = tempfile.mkdtemp()
    os.chdir(newpath)

    def fin():
        os.chdir(curdir)

    request.addfinalizer(fin)


@pytest.fixture(scope="session", autouse=True)
def testfiles():
    f = {
        "data100": "./test_resources/data100.txt",
        "data10000": "./test_resources/data10000.txt",
        "dataset100": "./test_resources/dataset100.db",
        "h5100": "./test_resources/h5100.db",
        "h5100_norm": "./test_resources/h5100_normalized.db",
        "database1.h5": "./test_resources/database1.h5",
        "dataset1_normalized.h5": "./test_resources/dataset1_normalized.h5"
    }
    for name in f.keys():
        f[name] = os.path.abspath(f[name])

    return f
