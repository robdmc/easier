from scipy import stats
import numpy as np


def ecdf(
    x,
    N=100,
    inverse=False,
    as_percent=False,
    centered=False,
    folded=False,
    plot=False,
    curve_args=None,
    curve_kwargs=None,
):
    """
    Compute the empirical cumulative distribution function (ECDF) for a given dataset.

    This is a thin wrapper around scipy ECDF implementation with additional
    functionality for centering, folding, and plotting.

    Parameters
    ----------
    x : array-like
        Array of points for which to compute the ECDF.
    N : int, optional
        Number of output points in the ECDF. Default is 100.
    inverse : bool, optional
        If True, returns 1 - ECDF. Default is False.
    as_percent : bool, optional
        If True, returns y values as percentages (0-100). Default is False.
    centered : bool, optional
        If True, centers the percentiles at 50%. Default is False.
    folded : bool, optional
        If True, folds the centered percentiles to not exceed 50%.
        The result makes the y value be the probability that an observation
        exceeds the x value in that direction. Default is False.
    plot : bool, optional
        If True, returns a holoviews Curve plot instead of arrays. Default is False.
    curve_args : tuple, optional
        Additional positional arguments for the holoviews Curve plot.
    curve_kwargs : dict, optional
        Additional keyword arguments for the holoviews Curve plot.

    Returns
    -------
    tuple or holoviews.Curve
        If plot is False, returns a tuple of (x_out, y_out) where:
            - x_out : array
                Array of x values
            - y_out : array
                Array of corresponding ECDF values
        If plot is True, returns a holoviews.Curve object.
    """
    x_out = np.linspace(min(x), max(x), N)
    res = stats.ecdf(x)
    y_out = res.cdf.evaluate(x_out)
    if curve_args is None:
        curve_args = tuple()
    if curve_kwargs is None:
        curve_kwargs = {}
    if centered or folded:
        y_out = y_out - 0.5
    if folded:
        y_out = 0.5 - np.abs(y_out)
    if inverse:
        y_out = 1 - y_out
    if as_percent:
        y_out = 100 * y_out
    if plot:
        import holoviews as hv

        return hv.Curve((x_out, y_out), *curve_args, **curve_kwargs)
    else:
        return (x_out, y_out)
