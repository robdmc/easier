from typing import Union, Iterable
import re


def slugify(
    vals: Union[str, Iterable[str]],
    sep: str = "_",
    kill_camel: bool = False,
    as_dict: bool = False,
):
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
        vals = [re.sub(r"([0-9]|[a-z]|_)([A-Z])", r"\1_\2", v) for v in vals]

    out = [re.sub(r"[^A-Za-z0-9]+", sep, v.strip()).lower() for v in vals]
    out = [re.sub(r"_{2:}", sep, v) for v in out]
    out = [re.sub(r"^_", "", v) for v in out]
    out = [re.sub(r"_$", "", v) for v in out]

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
        raise ValueError("You must supply a list of columns")

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
        raise ValueError(
            "You can only pass dataframes or series objects to this function"
        )


def pandas_time_to_utc_seconds(df_or_ser, columns=None):
    """
    Convert pandas Timestamp records to integer unix seconds
    since epoch.

    If columns is specified, only those columns get converted.
    Otherwise, all timestamp columns get converted
    """

    # A function for converting a series
    def series_converter(ser):
        return ser.view("int64") // 10**9

    return _pandas_time_integer_converter(
        series_converter, "datetime64[ns]", df_or_ser, columns
    )


def pandas_utc_seconds_to_time(df_or_ser, columns=None):
    """
    Convert pandas integer fields to timestamps.  The integers must
    contain the number of seconds since unix epoch.

    You must specify the columns you want to transform
    """

    # A function to convert a series
    def series_converter(ser):
        return (ser.astype("int64") * 10**9).view("datetime64[ns]")

    return _pandas_time_integer_converter(series_converter, "int64", df_or_ser, columns)


def localize_utc_to_timezone(time_ser, timezone, return_naive=True):
    """
    Takes a series of utc timestamps and converts them
    to (naive) timestamps of the specified timezone
    """
    time_ser = time_ser.dt.tz_convert(timezone)
    if return_naive:
        time_ser = time_ser.dt.tz_localize(None)
    return time_ser


def events_from_starting_ending(
    *,
    df,
    start_time_col,
    end_time_col,
    delta_cols=None,
    non_delta_cols=None,
    new_time_col_name="time",
    non_numerics_are_index=True,
):
    """
    Converts a dataframe with start and end times into a dataframe of events.
    Numeric delta columns are assumed to be deltas, and are added to the start time
    and substracted at end time.  Non-delta columns are included in the events
    unaltered and can optionally be used as the index.

    Args:
        df: The dataframe to convert

        start_time_col: The name of the column containing the start time

        end_time_col: The name of the column containing the end time

        delta_cols: The names of the columns to be treated as deltas

        non_delta_cols: The names of the columns to be treated as non-deltas

        new_time_col_name: The name of the new event time column

        non_numerics_are_index: If True, the non-numerics columns are combined
                                with the time column in a groupby statement and
                                the deltas are summed by those groups.
    """
    import pandas as pd

    # Turn defaults args into lists
    if delta_cols is None:
        delta_cols = []

    if non_delta_cols is None:
        non_delta_cols = []

    # If no delta columns are specified, assume all non time columns are deltas
    if delta_cols is None:
        cols_to_keep = set(df.columns) - {start_time_col, end_time_col}
        delta_cols = [c for c in df.columns if c in cols_to_keep]

    # Sainity check columns specifications
    if (set(delta_cols).union(non_delta_cols)).intersection(
        {start_time_col, end_time_col}
    ):
        raise ValueError("(Non)delta cols cant contain start or end time col")

    if set(delta_cols).intersection(non_delta_cols):
        raise ValueError("Same column(s) found in both delta and non_delta columns")

    # Create a starting frame with the start times as event times
    df_start = df[[start_time_col] + non_delta_cols + delta_cols].rename(
        columns={start_time_col: new_time_col_name}
    )

    # Create an ending frame with the end times as event times
    df_end = df[[end_time_col] + non_delta_cols + delta_cols].rename(
        columns={end_time_col: new_time_col_name}
    )

    # Negate the delta columns in the ending frame
    df_end[delta_cols] = -df_end[delta_cols]

    # Create the event frame ordered by event time
    df = pd.concat([df_start, df_end], ignore_index=True, sort=False)
    df = df.sort_values(by=new_time_col_name)

    # Set the index if requested
    if non_numerics_are_index:
        index_cols = non_delta_cols + [new_time_col_name]
        df = df.groupby(by=index_cols).sum().sort_index()

    return df


def weekday_string(ser, kind="tag"):
    """
    Transform a pandas series of datetims into strings
    corresponding to their weekday.
    Args:
        ser: a pandas series object of timestamps
        kind: ['tag', 'slug', 'name']
              tag -> ['0_mon', '1_tue', ...]
              slug -> ['Mon', 'Tue', ...]
              name -> ['Monday', 'Tuesday', ...]
    """
    import pandas as pd
    import calendar

    if not isinstance(ser, pd.Series):
        raise ValueError("Input must be a pandas series")

    ser = ser.dt.weekday

    allowed_kinds = ["slug", "name", "tag"]
    if kind not in allowed_kinds:
        raise ValueError(f"kind must be on of {allowed_kinds}")
    if kind == "tag":
        out = [f"{d}_{calendar.day_abbr[d].lower()}" for d in ser]
    elif kind == "slug":
        out = [calendar.day_abbr[d] for d in ser]
    elif kind == "name":
        out = [calendar.day_name[d] for d in ser]

    out = pd.Series(out, index=ser.index)
    return out


def month_string(ser, kind="tag"):
    """
    Transform a pandas series of datetims into strings
    corresponding to their months.
    Args:
        ser: a pandas series object of timestamps
        kind: ['tag', 'slug', 'name']
              tag -> [a_jan', 'b_feb', ...]
              slug -> ['Jan', 'Feb', ...]
              name -> ['January', 'February', ...]
    """
    import calendar
    from string import ascii_lowercase
    import pandas as pd

    if not isinstance(ser, pd.Series):
        raise ValueError("Input must be a pandas series")

    ser = ser.dt.month

    allowed_kinds = ["slug", "name", "tag"]
    if kind not in allowed_kinds:
        raise ValueError(f"kind must be on of {allowed_kinds}")
    if kind == "tag":
        out = [f"{ascii_lowercase[d]}_{calendar.month_abbr[d].lower()}" for d in ser]
    elif kind == "slug":
        out = [calendar.month_abbr[d] for d in ser]
    elif kind == "name":
        out = [calendar.month_name[d] for d in ser]

    out = pd.Series(out, index=ser.index)
    return out
