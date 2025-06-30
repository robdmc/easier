from ..bernstein import Bernstein, Compress, BernsteinFitter
import numpy as np
import pytest

from ..bernstein import Bernstein, Compress, BernsteinFitter
import pytest

def get_xy():
    x = np.linspace(0, np.pi / 4, 300)
    y = np.sin(x)
    dydx = np.cos(x)
    return (x, y, dydx)

def test_bernstein_simple():
    x, y, dydx = get_xy()
    yf = Bernstein(x=x, y=y, N=1000).predict(x)
    dydxf = Bernstein(x=x, y=y, N=1000).predict_derivative(x)
    max_delta = np.max(np.abs(yf - y))
    max_delta_deriv = np.max(np.abs(dydxf - dydx))
    assert max_delta < 0.0001
    assert max_delta_deriv < 0.001

def test_bernstein_limits():
    x, y, dydx = get_xy()
    yf = Bernstein(x=x, y=y, N=1000, xlim=(0, np.pi / 4)).predict(x)
    max_delta = np.max(np.abs(yf - y))
    assert max_delta < 0.0001

def test_bernstein_bad_compress_type():
    with pytest.raises(ValueError):
        Compress().compress([1, 2, 3])

def test_bernstein_compress_dont_learn_lims():
    cmp = Compress(min_val=0, max_val=np.pi)
    xc = cmp.compress(np.array([1, 2, 3]), learn_limits=False)
    assert tuple([int(v) for v in cmp.expand(xc)]) == (1, 2, 3)

def test_bernstein_compress_dont_learn_bad_limits():
    cmp = Compress()
    with pytest.raises(ValueError):
        cmp.compress(np.array([1, 2, 3]), learn_limits=False)

def test_bernstein_too_short():
    x = np.array([1, 2])
    with pytest.raises(ValueError):
        Bernstein(x=x, y=x, N=1000).predict(x)

def test_bernstein_not_numpy():
    x = [1, 2, 3]
    with pytest.raises(ValueError):
        Bernstein(x=x, y=x, N=1000).predict(x)

def test_bernstein_fitter():
    x = np.linspace(0, 7 * np.pi / 3, 300)
    y = -1 + np.sin(x) + 0.8 * x
    y[-1] = y[-1] + 1
    blob = BernsteinFitter(non_negative=False, monotonic=False, match_left=False, match_right=False).fit(x, y, 150).to_blob()
    b = BernsteinFitter().from_blob(blob)
    yf = b.predict(x)
    assert np.abs(np.max(y - yf)) < 0.1
    yf = BernsteinFitter(non_negative=False, monotonic=False, match_left=False, match_right=False).fit_predict(x, y, 5)
    assert np.abs(yf[0] - y[0]) > 1e-06
    assert np.abs(yf[-1] - y[-1]) > 1e-06
    assert np.min(yf) < 0
    assert np.min(np.diff(yf)) < 0
    yf = BernsteinFitter(non_negative=False, monotonic=False, match_left=True, match_right=True).fit_predict(x, y, 5)
    assert np.abs(yf[0] - y[0]) < 1e-06
    assert np.abs(yf[-1] - y[-1]) < 1e-06
    yf = BernsteinFitter(non_negative=True, monotonic=False, match_left=False, match_right=False).fit_predict(x, y, 5)
    assert np.min(y) < 0
    assert np.min(yf) > 0
    yf = BernsteinFitter(non_negative=False, monotonic=True, match_left=False, match_right=False).fit_predict(x, y, 15)
    assert np.min(np.diff(y)) < 0
    assert np.min(np.diff(yf)) >= -1e-05
    t = np.linspace(0, 2 * np.pi, 1000)
    ys = np.exp(np.sin(2 * t))
    dydt = np.diff(ys) / np.diff(t)
    fitter = BernsteinFitter(non_negative=False, monotonic=False).fit(t, ys, 55)
    dydtf = fitter.predict_derivative(t)[1:]
    assert np.max(np.abs(dydtf - dydt)) < 0.05