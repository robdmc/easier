import pandas as pd
import numpy as np
import datetime
from dateutil.parser import parse
from typing import Union


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





