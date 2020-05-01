from collections import Counter
from .shaper import Shaper


def sigma_edit_series(in_series, sigma_thresh, iter_counter=None, max_iter=20):
    """
    Workhorse recursive sigma edit function.
    """
    import numpy as np
    iter_counter = Counter() if iter_counter is None else iter_counter

    if in_series.count() == 0:
        msg = "Error:  No non-NaN values from which to remove outliers"
        raise ValueError(msg)

    iter_counter.update('n')
    if iter_counter['n'] > max_iter:
        msg = "Error:  Max Number of iterations exceeded in sigma-editing"
        raise ValueError(msg)

    resid = in_series - in_series.mean()
    std = resid.std()
    sigma_t = sigma_thresh * std
    outside = resid.abs() >= sigma_t
    if any(outside):
        in_series.loc[outside] = np.NaN
        in_series = sigma_edit_series(
            in_series, sigma_thresh, iter_counter, max_iter)

    return in_series


def kill_outliers_sigma_edit(data, sigma_thresh=3, max_iter=20):
    """
    Recursive sigma edit setting outliers to NaN.
    For 2-d arrays and dataframes, all elements will be
    included in the sampe.

    Args:
        data: ndarray, series or dataframe
        sigma_thresh: the threshold standard dev for editing
        max_iter: The maximum iterations allowed.
    """
    shaper = Shaper()
    x = shaper.flatten(data)
    x = sigma_edit_series(x, sigma_thresh=sigma_thresh, max_iter=max_iter)
    return shaper.expand(x)


def kill_outliers_iqr(data, multiple=1.5):
    """
    Identify outliers using the IQR method and null outliers
    to NaN.  For 2-d nd-arrays and dataframes all elements
    are used in the sample.

    Args:
        data: dataframe, series or ndarray
        multiple: the iqr multiple to use in outlier removal
    """
    import numpy as np
    shaper = Shaper()
    x = shaper.flatten(data)

    # Run outlier detection
    q1, q2 = tuple(np.percentile(x, [25, 75]))
    iqr = q2 - q1
    lower = q1 - multiple * iqr
    upper = q2 + multiple * iqr
    out = x.where((lower <= x) & (x <= upper))
    return shaper.expand(out)
