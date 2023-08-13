# from pyrsistent import T
from ..bernstein import Bernstein, Compress, BernsteinFitter
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


def test_bernstein_fitter():
    import numpy as np

    # Make a test function that goes negative, has negative slope and a weird right end
    x = np.linspace(0, 7 * np.pi / 3, 300)
    y = -1 + np.sin(x) + 0.8 * x
    y[-1] = y[-1] + 1

    # Check completely free fitt
    blob = (
        BernsteinFitter(
            non_negative=False, monotonic=False, match_left=False, match_right=False
        )
        .fit(x, y, 150)
        .to_blob()
    )
    b = BernsteinFitter().from_blob(blob)
    yf = b.predict(x)
    assert np.abs(np.max(y - yf)) < 0.1

    # Test that matching works
    yf = BernsteinFitter(
        non_negative=False, monotonic=False, match_left=False, match_right=False
    ).fit_predict(x, y, 5)
    assert np.abs(yf[0] - y[0]) > 1e-6
    assert np.abs(yf[-1] - y[-1]) > 1e-6
    assert np.min(yf) < 0
    assert np.min(np.diff(yf)) < 0

    yf = BernsteinFitter(
        non_negative=False, monotonic=False, match_left=True, match_right=True
    ).fit_predict(x, y, 5)
    assert np.abs(yf[0] - y[0]) < 1e-6
    assert np.abs(yf[-1] - y[-1]) < 1e-6

    # Test non-neg works
    yf = BernsteinFitter(
        non_negative=True, monotonic=False, match_left=False, match_right=False
    ).fit_predict(x, y, 5)
    assert np.min(y) < 0
    assert np.min(yf) > 0

    # Test monotonic works
    yf = BernsteinFitter(
        non_negative=False, monotonic=True, match_left=False, match_right=False
    ).fit_predict(x, y, 15)
    assert np.min(np.diff(y)) < 0
    assert np.min(np.diff(yf)) >= -1e-5

    # Test derivative
    t = np.linspace(0, 2 * np.pi, 1000)
    ys = np.exp(np.sin(2 * t))
    dydt = np.diff(ys) / np.diff(t)
    fitter = BernsteinFitter(non_negative=False, monotonic=False).fit(t, ys, 55)
    dydtf = fitter.predict_derivative(t)[1:]
    assert np.max(np.abs(dydtf - dydt)) < 0.05
