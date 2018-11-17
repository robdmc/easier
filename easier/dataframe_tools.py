from typing import Union, List, Iterable
import datetime
import re

from dateutil.parser import parse
import numpy as np
import pandas as pd
from scipy.stats import scoreatpercentile


def mute_warnings():
    """
    Mute all Python warnings
    """
    import warnings
    warnings.filterwarnings("ignore")


def iqr_outlier_killer(
        x: Iterable,
        multiple: float=1.5,
        dropna: bool=False) -> Union[List, np.ndarray, pd.Series]:
    """
    Identify outliers using the IQR method and null outliers
    to NaN.  Input types of list, np.array, or pd.Series will
    have matching output types.  Other iterables will default
    to an array output type.

    Args:
        x: the iterable from which to remove outliers
        multiple: the iqr multiple to use in outlier removal
        dropna: a flag indicating whether to drop all NaNs after
        outlier nulling.
    """
    # Set output type based on input type
    if isinstance(x, list):
        transformer = list
    elif isinstance(x, pd.Series):
        transformer = pd.Series
    else:
        transformer = np.array

    # Transform input to series
    x = pd.Series(x)

    # Run outlier detection
    q1, q2 = tuple(scoreatpercentile(x, [25, 75]))
    iqr = q2 - q1
    lower = q1 - multiple * iqr
    upper = q2 + multiple * iqr
    out = x.where((lower <= x) & (x <= upper))
    if dropna:
        out = out.dropna()

    return transformer(out)


def date_diff(
        series: pd.Series,
        epoch: Union[str, datetime.datetime, np.datetime64],
        amount: int =1,
        unit: str = 'D',
        dtype: str = 'float'
):
    """
    Compute the number of intervals between an epoch and elements of a
    Pandas series.

    Allowed units are at:
    https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.datetime.html

    Entering bad units will tell you what is allowed

    """
    # Make sure the units look right
    allowed_units = ['D', 'W', 'h', 'm', 's', 'ms', 'us']
    if unit not in allowed_units:
        raise ValueError(f'unit {unit} is not in {allowed_units}')

    # Transform epoch into a datetime64
    if isinstance(epoch, str):
        epoch = parse(epoch)
    epoch = np.datetime64(epoch)

    # Return a series of numerical differences
    return ((series - epoch) / np.timedelta64(amount, unit)).astype(dtype)


def slugify(vals: Union[str, List[str]], sep: str = '_'):
    """
    Creates slugs out of string inputs.
    """
    if isinstance(vals, str):
        str_input = True
        vals = [vals]
    else:
        str_input = False
    out = [re.sub(r'[^A-Za-z0-9]+', sep, v).lower() for v in vals]
    out = [re.sub(r'_{2:}', sep, v) for v in out]

    if str_input:
        return out[0]
    else:
        return out
