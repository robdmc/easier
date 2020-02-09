from typing import Union, List
import datetime
import re

from dateutil.parser import parse
import numpy as np
import pandas as pd


def date_diff(
        series: pd.Series,
        epoch: Union[str, datetime.datetime, np.datetime64, pd.Series],
        amount: int = 1,
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
    elif isinstance(epoch, pd.Series):
        epoch = epoch.astype(np.datetime64)
    else:
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
    out = [re.sub(r'[^A-Za-z0-9]+', sep, v.strip()).lower() for v in vals]
    out = [re.sub(r'_{2:}', sep, v) for v in out]
    out = [re.sub(r'^_', '', v) for v in out]
    out = [re.sub(r'_$', '', v) for v in out]

    if str_input:
        return out[0]
    else:
        return out
