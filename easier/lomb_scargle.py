#! /usr/bin/env python

try:
    import pandas as pd
    import numpy as np
# will catch import errors in module_checker_lib so won't test this branch
except ImportError:  # pragma: nocover
    pass


def _next_power_two(x):
    """given a number, returns the next power of two"""
    x = int(x)
    n = 1
    while n < x:
        n = n << 1
    return n


def _compute_pad(t, interp_exponent=0):
    """
    Given a sorted time series t, compute the zero padding.
    The final padded arrays are the next power of two in length multiplied
    by 2 ** interp_exponent.
    returns t_pad and y_pad
    """
    t_min, t_max, n = t[0], t[-1], len(t)
    dt = (t_max - t_min) / float(n - 1)
    n_padded = _next_power_two(len(t)) << interp_exponent
    n_pad = n_padded - n
    t_pad = np.linspace(t_max + dt, t_max + dt + (n_pad - 1) * dt, n_pad)
    y_pad = np.zeros(len(t_pad))
    return t_pad, y_pad


def _compute_params(t):
    """
    Takes a timeseries and computes the parameters needed for the fast
    lomb scargle algorithm in gatspy
    """
    t_min, t_max, n = t[0], t[-1], len(t)
    dt = (t_max - t_min) / float(n - 1)
    min_freq = 1.0 / (t_max - t_min)
    d_freq = 1.0 / (2 * dt * len(t))
    return min_freq, d_freq, len(t)


def lomb_scargle(time, value, interp_exponent=0, freq_order=False):
    """Compute the Lomb-Scargle periodogram for a time series.

    This function performs a Lomb-Scargle periodogram analysis on a time series,
    which is useful for finding periodic signals in unevenly sampled data.

    Args:
        time (pandas.Series or array-like): Time values
        value (pandas.Series or array-like): Observed values
        interp_exponent (int, optional): Power of two for spectrum interpolation.
            Defaults to 0.
        freq_order (bool, optional): If True, return results in frequency order
            instead of period order. Defaults to False.

    Returns:
        pandas.DataFrame: A dataframe containing the periodogram results with columns:
            - period: The period of each frequency component
            - freq: The frequency of each component
            - power: The power spectrum
            - amp: The amplitude spectrum

    Note:
        The input time series is automatically:
        - Sorted by time
        - Mean-centered
        - Zero-padded to the next power of two
    """
    # do imports here to avoid loading plot libraries when this
    # module is loaded in __init__.py
    # which then doesn't allow for doing matplotlib.use() later
    import gatspy

    # Convert inputs to Series if they aren't already
    if not isinstance(time, pd.Series):
        time = pd.Series(time)
    if not isinstance(value, pd.Series):
        value = pd.Series(value)

    # Check that time and value have compatible shapes
    if len(time) != len(value):
        raise ValueError(
            f"Time and value arrays must have the same length. "
            f"Got time length {len(time)} and value length {len(value)}"
        )

    # Create a new dataframe with just the time and value columns
    df = pd.DataFrame({"t": time, "y": value}).dropna()

    # standardize column names, remove mean from values, and sort by time
    df = df.sort_values(by=["t"])
    df["y"] = df["y"] - df.y.mean()

    #  compute total energy in the time series
    E_in = np.sum((df.y * df.y))

    # appropriately zero-pad the timeseries before taking spectrum
    pre_pad_length = len(df)
    t_pad, y_pad = _compute_pad(df.t.values, interp_exponent=interp_exponent)
    if len(t_pad) > 0:
        df = df.append(pd.DataFrame({"t": t_pad, "y": y_pad}), ignore_index=True)

    # fit the lombs scargle model to the time series
    model = gatspy.periodic.LombScargleFast()
    model.fit(df.t.values, df.y.values, 1)

    # compute params for getting results out of lomb scargle fit
    f0, df, N = _compute_params(df.t.values)
    f = f0 + df * np.arange(N)
    p = 1.0 / f

    # retrieve the lomb scarge fit and normalize for power / amplitude
    yf = model.score_frequency_grid(f0, df, N)
    yf_power = 2 * yf * E_in * len(yf) / float(pre_pad_length) ** 2
    yf_amp = np.sqrt(yf_power)

    # generate the output dataframe
    df = pd.DataFrame({"freq": f, "period": p, "power": yf_power, "amp": yf_amp})[
        ["period", "freq", "power", "amp"]
    ]

    # order by period if desired
    if not freq_order:
        df = df.sort_values(by="period")
    return df
