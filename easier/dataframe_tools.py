import os
from typing import Union, Iterable
import re


def heatmap(df, axis=None, cmap="magma", format="{:.1f}"):
    # This will crash if not run in a jupyter notebook.  That's ok.
    display(df.style.background_gradient(axis=axis, cmap=cmap).format(format))


def column_level_flattener(df, level=1, kill_index_names=False):
    """
    Takes a multi-level column dataframe and returns a flattened version.
    Default is to use level=1, but you can use other levels as well.
    Args:
        level: The level of the index you want to use (defaults to 1)
               "smash" will join column levels with an underscore
        kill_index_names: If True, the column/index names will be set to None

    """
    df = df.copy()
    if level == "smash":
        df.columns = ["_".join([str(v) for v in t]) for t in df.columns]
    else:
        df.columns = df.columns.get_level_values(level)
    if kill_index_names:
        df.columns.name = None
        df.index.name = None

    return df


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


def weekday_string(ser, kind="slug"):
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


def month_string(ser, kind="slug"):
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


def get_quick_schema_class():
    import pandera as pa

    class QuickSchema:
        dt = pa.dtypes

        def __init__(self, columns, **kwargs):
            """
            A thin wrapper around Pandera DataframeSchema. The main
            use case is to allow more terse schema specification than
            the provided by the Pandera API.

            https://pandera.readthedocs.io/en/stable/reference/generated/pandera.api.pandas.container.DataFrameSchema.html

            The main shortcuts this class provides are that the output
            columns will exactly match the keys of the schema in the
            supplied order.  Additionally, columns can be specified with
            just their types instead of the more verbose pa.Column(...)
            syntax.

            There is a class attribute named dt that is a link to
            pandera datatypes.  This allows for easy tab-completion of types.

            All columns specified as types will result in nullable columns
            that have been coerced to the requested data type.  If you don't
            like this default behavior, pass a pa.Column() object that
            explicitely communicates what you want.

            Once constructed, a QuickSchema object will have a .schema
            attribute that contains the underlying pandera DataFrameSchema.
            This allows you the full power of pandera's api on that schema.

            Apply the schema to a dataframe using the .apply() method, which
            returns a new dataframe

            Args:
                columns: A dict mapping a string columns name into either a
                        full on pa.Columns() instance, or any object that
                        pandera knows how to interpret as a type
                **kwargs: Passed directly to DataframeSchema constructor
            """

            # Save off the column specifications to enable proper column ordering later
            self.col_spec = columns.copy()

            # Apply default ordering
            if "ordered" not in kwargs:
                kwargs["ordered"] = True
            self.ordered = kwargs["ordered"]

            # Apply default coercion
            if "coerce" not in kwargs:
                kwargs["coerce"] = True

            # Initialize a columns dict that Panera will understand
            typed_cols = {}

            # Populate a pandera friendly columns dict
            for key, val in columns.items():
                # If the column type is already a pa.Column type, just use it
                if isinstance(val, type(pa.Column(self.dt.Int16))):
                    typed_cols[key] = val
                # Otherwise, assume it is a type and coerce it to a nullable column
                else:
                    typed_cols[key] = pa.Column(val, nullable=True)

            # Create the Pandera DataFrameSchema object and assign it to the schema attribute
            self.schema = pa.DataFrameSchema(typed_cols, **kwargs)

        @classmethod
        def infer_from_dataframe(cls, df):
            """
            Class method that creates a Quickschema object from a dataframe
            """
            pandera_schema = pa.infer_schema(df)
            spec = {}
            for name, col in pandera_schema.columns.items():
                spec[name] = col.dtype
            return cls(spec)

        @property
        def dtypes(self):
            return dict(self.schema.columns)

        @property
        def ibis_schema(self):
            from ibis.expr import schema as sch

            return sch.from_mapping(
                {
                    col: self.schema.dtypes[col].type
                    for col in self.schema.columns.keys()
                }
            )

        def __repr__(self):
            s = repr(self.schema).replace("DataFrameSchema", "QuickSchema")
            return s

        def __str__(self):
            s = str(self.schema).replace("DataFrameSchema", "QuickSchema")
            return s

        def apply(self, df):
            """
            Apply the schema to retrieve a new dataframe
            """
            if self.ordered:
                df = df[list(self.col_spec.keys())]
            df = df.copy()

            return self.schema(df)

    return QuickSchema


def get_pandas_sql_class():
    import duckdb

    class PandasSql:
        def __init__(self, file=":memory:", overwrite=False, **table_mappings):
            """
            If the file is specified and it contains a database with tables,
            you need not register dataframes. However, any dataframes you register
            will overwrite any existing table of that name.

            Uses duckdb sql dialect

            Args:
            file: str = an optional name for a file for the frame records
            overwrite: bool = When true, this will overwrite the existing database
            **table_mappings: dict = {table_name1: df1, table_name2: df2, ...}
            """
            if file == ":memory:":
                self.file = file
            else:
                self.file = os.path.realpath(os.path.expanduser(file))

            self.conn = self._get_db_connection(self.file, overwrite)
            self.register(**table_mappings)

        def _get_db_connection(self, file, overwrite):
            if overwrite and os.path.isfile(file):
                os.unlink(file)
            conn = duckdb.connect(file)
            return conn

        def register(self, **table_mappings):
            """
            Creates tables from dataframes.
            **table_mappings: dict = {{table_name1: df1, table_name2: df2, ...}}
            """
            if len(table_mappings) == 0:
                return self

            for table_name, df in table_mappings.items():
                self.conn.execute(
                    f"drop table if exists {table_name}; create table {table_name} as select * from df"
                )

        def query(self, sql):
            """
            Returns a dataframe from a sql query.
            (Use .execute() for queries that don't return a value)
            Args:
                sql: str = a string containing the query
            Returns: Pandas Dataframe = a dataframe with the results
            """
            return self.conn.query(sql).to_df()

        def execute(self, sql):
            """
            Runs a sql query (e.g. create, add, drop) that doesn't
            return a value.
            (Use .query() for queries that return a value)
            Args:
                sql: str = a string containing the query
            """
            self.conn.execute(sql)

        @property
        def tables(self):
            """
            A property that returns a frame showing all table names
            """
            df = self.conn.query("show tables").to_df()
            return df.sort_values(by="name")

    return PandasSql


def hex_from_dataframe(df):
    """
    Convert a dataframe to a hex string
    """
    import pickle
    import binascii
    import lzma

    serialized_df = pickle.dumps(df)
    compressed_data = lzma.compress(serialized_df, preset=9)
    hex_encoded_string = binascii.hexlify(compressed_data).decode("utf-8")
    return hex_encoded_string


def hex_to_dataframe(hex_encoded_string):
    """
    Convert a hex string to a dataframe
    """
    import pickle
    import binascii
    import lzma

    # Convert hex string back to binary
    compressed_data = binascii.unhexlify(hex_encoded_string)

    # Decompress using lzma
    serialized_df = lzma.decompress(compressed_data)

    # Unpickle to get the original DataFrame
    df = pickle.loads(serialized_df)

    return df
