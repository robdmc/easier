def ecdf(x, N=100, inverse=False, as_percent=False, centered=False, folded=False):
    """
    Thin wrapper around statsmodels ecdf.
    Arguments:
        x: array of points for which to find ecdf
        N: The number of output points you want in your ecdf
        inverse: Return 1 - ecdf
        centered: Centers the percentiles at 50%
        folded: Folds the centered percentiles to not exceed 50%
                The result makes the y value be the probability that
                an observsation exceeds the x value in that direction.
    Returns:
        x_out: an array of values
        y_out: an array of percentiles
    """
    import numpy as np
    from statsmodels.distributions.empirical_distribution import ECDF
    x_out = np.linspace(min(x), max(x), N)
    y_out = ECDF(x)(x_out)

    if centered or folded:
        y_out = y_out - .5

    if folded:
        y_out = .5 - np.abs(y_out)

    if inverse:
        y_out = 1 - y_out

    if as_percent:
        y_out = 100 * y_out
    return x_out, y_out
