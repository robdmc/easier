import numpy as np
import pytest
from easier.ecdf_lib import ecdf


def test_basic_ecdf():
    x = np.array([1, 2, 3, 4, 5])
    x_out, y_out = ecdf(x, N=5)
    # ECDF should be monotonic and go from 0 to 1
    assert np.all(np.diff(y_out) >= 0)
    assert np.isclose(y_out[0], 1 / 5, atol=1e-8) or y_out[0] < y_out[1]  # first step
    assert np.isclose(y_out[-1], 1.0, atol=1e-8)


def test_ecdf_as_percent():
    x = np.array([10, 20, 30, 40, 50])
    x_out, y_out = ecdf(x, N=5, as_percent=True)
    assert np.all((0 <= y_out) & (y_out <= 100))
    assert np.isclose(y_out[-1], 100.0, atol=1e-8)


def test_ecdf_inverse():
    x = np.array([1, 2, 3, 4, 5])
    x_out, y_out = ecdf(x, N=5, inverse=True)
    # Inverse ECDF should be decreasing
    assert np.all(np.diff(y_out) <= 0)
    assert np.isclose(y_out[0], 1 - 1 / 5, atol=1e-8) or y_out[0] > y_out[1]
    assert np.isclose(y_out[-1], 0.0, atol=1e-8)


def test_ecdf_centered():
    x = np.array([1, 2, 3, 4, 5])
    x_out, y_out = ecdf(x, N=5, centered=True)
    # Centered ECDF should be shifted by -0.5
    assert np.isclose(y_out[0], 1 / 5 - 0.5, atol=1e-8)
    assert np.isclose(y_out[-1], 1.0 - 0.5, atol=1e-8)


def test_ecdf_folded():
    x = np.array([1, 2, 3, 4, 5])
    x_out, y_out = ecdf(x, N=5, folded=True)
    # Folded ECDF should not exceed 0.5
    assert np.all(np.abs(y_out) <= 0.5)
    assert np.isclose(y_out[0], 0.5 - abs(1 / 5 - 0.5), atol=1e-8)


def test_ecdf_plot():
    x = np.array([1, 2, 3, 4, 5])
    curve = ecdf(x, N=5, plot=True)
    # Should return a holoviews Curve object
    try:
        import holoviews as hv

        assert isinstance(curve, hv.Curve)
    except ImportError:
        pytest.skip("holoviews not installed")
