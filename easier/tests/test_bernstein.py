from ..bernstein import Bernstein, Compress
import pytest


def get_xy():
    import numpy as np
    x = np.linspace(0, np.pi / 4, 300)
    y = np.sin(x)
    dydx = np.cos(x)
    return x, y, dydx


def test_bernstein_simple():
    import numpy as np
    x, y, dydx = get_xy()

    yf = Bernstein(x=x, y=y, N=1000).predict(x)
    dydxf = Bernstein(x=x, y=y, N=1000).predict_derivative(x)
    max_delta = np.max(np.abs(yf - y))
    max_delta_deriv = np.max(np.abs(dydxf - dydx))
    assert max_delta < 1e-4
    assert max_delta_deriv < 1e-3


def test_bernstein_limits():
    import numpy as np
    x, y, dydx = get_xy()

    yf = Bernstein(x=x, y=y, N=1000, xlim=(0, np.pi / 4)).predict(x)
    max_delta = np.max(np.abs(yf - y))
    assert max_delta < 1e-4


def test_bernstein_bad_compress_type():

    with pytest.raises(ValueError):
        Compress().compress([1, 2, 3])


def test_bernstein_compress_dont_learn_lims():
    import numpy as np

    cmp = Compress(min_val=0, max_val=np.pi)
    xc = cmp.compress(np.array([1, 2, 3]), learn_limits=False)
    assert tuple([int(v) for v in cmp.expand(xc)]) == (1, 2, 3)


def test_bernstein_compress_dont_learn_bad_limits():
    import numpy as np

    cmp = Compress()

    with pytest.raises(ValueError):
        cmp.compress(np.array([1, 2, 3]), learn_limits=False)


def test_bernstein_too_short():
    import numpy as np
    x = np.array([1, 2])

    with pytest.raises(ValueError):
        Bernstein(x=x, y=x, N=1000).predict(x)


def test_bernstein_not_numpy():
    x = [1, 2, 3]

    with pytest.raises(ValueError):
        Bernstein(x=x, y=x, N=1000).predict(x)
