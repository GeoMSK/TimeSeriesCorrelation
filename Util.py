__author__ = 'gm'


def calc_limit(limit, num: int) -> int:
    """
    limits the number "num" based on the given limit
    :param limit: this is the limit to enforce on num, it may be an integer or a string. In case of an integer
    num is set to limit, or is left untouched if limit>num. In case of a string (eg format: %70) num is set to
    the percentage indicated by limit, in the example given it will be num * 0,7. If this is None the num is returned
    :param num: the number to limit
    :return: the num with the limit applied, this will be an int at all cases
    """
    ret = None
    if limit is None:
        ret = num
    elif isinstance(limit, int):
        ret = num if limit > num else limit
    elif isinstance(limit, str):
        if limit[0] == "%":
            l = float(limit[1:]) / 100
            assert 0 <= l <= 1
            ret = round(num * l)
        else:
            ret = num if int(limit) > num else int(limit)
    return ret
