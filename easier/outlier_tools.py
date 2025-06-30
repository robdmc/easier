from .shaper import Shaper
from collections import Counter
import numpy as np

from collections import Counter
from .shaper import Shaper

def sigma_edit_series(in_series, sigma_thresh, iter_counter=None, max_iter=20):
    """
    Recursively remove outliers from a series using sigma editing method.

    This function iteratively identifies and removes outliers that fall outside
    a specified number of standard deviations from the mean. The process continues
    until no more outliers are found or the maximum number of iterations is reached.

    Args:
        in_series (pandas.Series): Input series containing numeric values
        sigma_thresh (float): Number of standard deviations to use as threshold
        iter_counter (Counter, optional): Counter to track iterations. Defaults to None
        max_iter (int, optional): Maximum number of iterations allowed. Defaults to 20

    Returns:
        pandas.Series: Series with outliers replaced by NaN values

    Raises:
        ValueError: If input series has no non-NaN values
        ValueError: If maximum number of iterations is exceeded
    """
    iter_counter = Counter() if iter_counter is None else iter_counter
    if in_series.count() == 0:
        msg = 'Error:  No non-NaN values from which to remove outliers'
        raise ValueError(msg)
    iter_counter.update('n')
    if iter_counter['n'] > max_iter:
        msg = 'Error:  Max Number of iterations exceeded in sigma-editing'
        raise ValueError(msg)
    resid = in_series - in_series.mean()
    std = resid.std()
    sigma_t = sigma_thresh * std
    outside = resid.abs() >= sigma_t
    if any(outside):
        in_series.loc[outside] = np.nan
        in_series = sigma_edit_series(in_series, sigma_thresh, iter_counter, max_iter)
    return in_series

def kill_outliers_sigma_edit(data, sigma_thresh=3, max_iter=20):
    """
    Remove outliers from data using recursive sigma editing method.

    This function identifies and removes outliers by recursively applying sigma editing,
    where values that fall outside a specified number of standard deviations from the mean
    are set to NaN. For 2D arrays and dataframes, all elements are included in the sample.

    Args:
        data (ndarray, Series, or DataFrame): Input data containing numeric values
        sigma_thresh (float, optional): Number of standard deviations to use as threshold
            for outlier detection. Defaults to 3.
        max_iter (int, optional): Maximum number of iterations allowed for the recursive
            sigma editing process. Defaults to 20.

    Returns:
        ndarray, Series, or DataFrame: Data with outliers replaced by NaN values,
            maintaining the original shape of the input.

    Note:
        The function uses a recursive approach where each iteration recalculates the mean
        and standard deviation after removing outliers from the previous iteration.
    """
    shaper = Shaper()
    x = shaper.flatten(data)
    x = sigma_edit_series(x, sigma_thresh=sigma_thresh, max_iter=max_iter)
    return shaper.expand(x)

def kill_outliers_iqr(data, multiple=1.5):
    """
    Remove outliers from data using the Interquartile Range (IQR) method.

    This function identifies and removes outliers by setting values that fall outside
    a specified multiple of the IQR to NaN. For 2D arrays and dataframes, all elements
    are included in the sample.

    Args:
        data (ndarray, Series, or DataFrame): Input data containing numeric values
        multiple (float, optional): The IQR multiple to use for outlier detection.
            Values outside the range [Q1 - multiple*IQR, Q3 + multiple*IQR] are
            considered outliers. Defaults to 1.5.

    Returns:
        ndarray, Series, or DataFrame: Data with outliers replaced by NaN values,
            maintaining the original shape of the input.

    Note:
        The IQR method is robust to outliers and works well for non-normal distributions.
        The default multiple of 1.5 corresponds to approximately ±2.7 standard deviations
        for normally distributed data.
    """
    shaper = Shaper()
    x = shaper.flatten(data)
    q1, q2 = tuple(np.percentile(x, [25, 75]))
    iqr = q2 - q1
    lower = q1 - multiple * iqr
    upper = q2 + multiple * iqr
    out = x.where((lower <= x) & (x <= upper))
    return shaper.expand(out)