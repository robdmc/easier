def ecdf(x, N=100):
    """
    Thin wrapper around statsmodels ecdf.
    Arguments:
        x: array of points for which to find ecdf
        N: The number of output points you want in your ecdf
    Returns:
        x_out: an array of values
        y_out: an array of percentiles
    """
    import numpy as np
    from statsmodels.distributions.empirical_distribution import ECDF
    x_out = np.linspace(min(x), max(x), N)
    y_out = ECDF(x)(x_out)
    return x_out, y_out
