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
        "data10000": "./test_resources/data10000.txt"
    }
    for name in f.keys():
        f[name] = os.path.abspath(f[name])

    return f