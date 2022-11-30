from typing import Union, Iterable
import re


def slugify(vals: Union[str, Iterable[str]], sep: str = '_', kill_camel: bool = False, as_dict: bool = False):
    """
    Creates slugs out of string inputs.
    """
    if isinstance(vals, str):
        str_input = True
        vals = [vals]
    else:
        str_input = False

    in_vals = list(vals)
    if kill_camel:
        vals = [re.sub(r'([0-9]|[a-z]|_)([A-Z])', r'\1_\2', v) for v in vals]

    out = [re.sub(r'[^A-Za-z0-9]+', sep, v.strip()).lower() for v in vals]
    out = [re.sub(r'_{2:}', sep, v) for v in out]
    out = [re.sub(r'^_', '', v) for v in out]
    out = [re.sub(r'_$', '', v) for v in out]

    if as_dict:
        return dict(zip(in_vals, out))

    if str_input:
        return out[0]
    else:
        return out


def _pandas_time_integer_converter(series_converter, type_str, df_or_ser, columns=None):
    import pandas as pd
    # You don't want to mutate input object
    df_or_ser = df_or_ser.copy()

    if isinstance(columns, str):
        raise ValueError('You must supply a list of columns')

    # The logic to convert each desired timestamp column of a dataframe
    if isinstance(df_or_ser, pd.DataFrame):
        if columns is None:
            # time_cols = [c for (c, v) in df_or_ser.dtypes.items() if v.name == 'datetime64[ns]']
            time_cols = [c for (c, v) in df_or_ser.dtypes.items() if v.name == type_str]
        else:
            time_cols = columns
        for col in time_cols:
            df_or_ser.loc[:, col] = series_converter(df_or_ser.loc[:, col])
        return df_or_ser

    # The logic to convert a series
    elif isinstance(df_or_ser, pd.Series):
        return series_converter(df_or_ser)

    else:
        raise ValueError('You can only pass dataframes or series objects to this function')


def pandas_time_to_utc_seconds(df_or_ser, columns=None):
    """
    Convert pandas Timestamp records to integer unix seconds
    since epoch.

    If columns is specified, only those columns get converted.
    Otherwise, all timestamp columns get converted
    """

    # A function for converting a series
    def series_converter(ser):
        return (ser.view('int64') // 10 ** 9)

    return _pandas_time_integer_converter(series_converter, 'datetime64[ns]', df_or_ser, columns)


def pandas_utc_seconds_to_time(df_or_ser, columns=None):
    """
    Convert pandas integer fields to timestamps.  The integers must
    contain the number of seconds since unix epoch.

    You must specify the columns you want to transform
    """

    # A function to convert a series
    def series_converter(ser):
        return (ser.astype('int64') * 10**9).view('datetime64[ns]')

    return _pandas_time_integer_converter(series_converter, 'int64', df_or_ser, columns)
