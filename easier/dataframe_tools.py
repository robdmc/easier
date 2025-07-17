from string import ascii_lowercase
from typing import Union, Iterable
import binascii
import lzma
import os
import pickle
import re
import tempfile
import zipfile


def heatmap(df, axis=None, cmap="magma", format="{:.1f}"):
    """
    Create a heatmap visualization of a pandas DataFrame using background
    gradients.

    This function is designed to be used in Jupyter notebooks and will crash if
    run outside of a Jupyter environment.

    Args:
        df (pandas.DataFrame): The DataFrame to visualize
        axis (int, optional): The axis along which to apply the gradient.
            Defaults to None.
        cmap (str, optional): The colormap to use for the gradient.
            Defaults to "magma".
        format (str, optional): The format string for displaying values.
            Defaults to "{:.1f}".

    Returns:
        None: Displays the styled DataFrame directly in the notebook.
    """
    from IPython.display import display

    display(df.style.background_gradient(axis=axis, cmap=cmap).format(format))


def column_level_flattener(df, level=1, kill_index_names=False):
    """
    Flatten a pandas DataFrame with multi-level columns into a single level.

    This function takes a DataFrame with multi-level columns and returns a
    version with flattened column names. It can either use a specific level
    from the multi-index or join all levels with underscores.

    Args:
        df (pandas.DataFrame): The DataFrame to flatten
        level (int or str, optional): The level of the index to use.
            Defaults to 1.
            If set to "smash", joins all column levels with underscores.
        kill_index_names (bool, optional): If True, removes column and index
            names. Defaults to False.

    Returns:
        pandas.DataFrame: A new DataFrame with flattened column names.

    Examples:
        >>> df = pd.DataFrame({('A', 'X'): [1, 2], ('A', 'Y'): [3, 4]})
        >>> column_level_flattener(df)
           X  Y
        0  1  3
        1  2  4

        >>> column_level_flattener(df, level='smash')
           A_X  A_Y
        0    1    3
        1    2    4
    """
    df = df.copy()
    if level == "smash":
        df.columns = ["_".join([str(v) for v in t]) for t in df.columns]
    else:
        df.columns = df.columns.get_level_values(level)
    if kill_index_names:
        df.columns.name = None  # type: ignore
        df.index.name = None
    return df


def slugify(
    vals: Union[str, Iterable[str]],
    sep: str = "_",
    kill_camel: bool = False,
    as_dict: bool = False,
):
    """
    Convert strings into URL-friendly slugs by removing special characters and
    normalizing case.

    This function handles both single strings and iterables of strings,
    converting them into standardized slugs that are suitable for URLs,
    filenames, or database keys.

    Args:
        vals (Union[str, Iterable[str]]): Input string or iterable of strings
            to convert to slugs
        sep (str, optional): Separator character to use between words.
            Defaults to "_"
        kill_camel (bool, optional): If True, converts camelCase to snake_case.
            Defaults to False
        as_dict (bool, optional): If True, returns a dictionary mapping
            original values to slugs. Defaults to False

    Returns:
        Union[str, List[str], Dict[str, str]]:
            - If vals is a string and as_dict is False: returns a single slug
              string
            - If vals is an iterable and as_dict is False: returns a list of
              slug strings
            - If as_dict is True: returns a dictionary mapping original values
                to their slugs

    Examples:
        >>> slugify("Hello World!")
        'hello_world'
        >>> slugify([
        ...     "Hello World!",
        ...     "FooBar"
        ... ], kill_camel=True)
        ['hello_world', 'foo_bar']
        >>> slugify([
        ...     "Hello World!",
        ...     "FooBar"
        ... ], as_dict=True)
        {'Hello World!': 'hello_world', 'FooBar': 'foobar'}
    """
    if isinstance(vals, str):
        str_input = True
        vals = [vals]
    else:
        str_input = False
    in_vals = list(vals)
    if kill_camel:
        vals = [re.sub("([0-9]|[a-z]|_)([A-Z])", "\\1_\\2", v) for v in vals]
    out = [re.sub("[^A-Za-z0-9]+", sep, v.strip()).lower() for v in vals]
    out = [re.sub("_{2:}", sep, v) for v in out]
    out = [re.sub("^_", "", v) for v in out]
    out = [re.sub("_$", "", v) for v in out]
    if as_dict:
        return dict(zip(in_vals, out))
    if str_input:
        return out[0]
    else:
        return out


def _pandas_time_integer_converter(series_converter, type_str, df_or_ser, columns=None):
    """
    Helper to convert pandas time columns.
    """
    import pandas as pd

    df_or_ser = df_or_ser.copy()
    if isinstance(columns, str):
        raise ValueError("You must supply a list of columns")
    if isinstance(df_or_ser, pd.DataFrame):
        if columns is None:
            time_cols = [c for c, v in df_or_ser.dtypes.items() if (v.name == type_str)]
        else:
            time_cols = columns
        for col in time_cols:
            df_or_ser[col] = series_converter(df_or_ser[col])
        return df_or_ser
    elif isinstance(df_or_ser, pd.Series):
        return series_converter(df_or_ser)
    else:
        raise ValueError("You can only pass dataframes or series objects to this function")


def pandas_time_to_utc_seconds(df_or_ser, columns=None):
    """
    Convert pandas Timestamp records to integer unix seconds
    since epoch.

    If columns is specified, only those columns get converted
    Otherwise, all timestamp columns get converted
    """

    def series_converter(ser):
        return ser.astype("int64") // 10**9

    return _pandas_time_integer_converter(series_converter, "datetime64[ns]", df_or_ser, columns)


def pandas_utc_seconds_to_time(df_or_ser, columns=None):
    """
    Convert pandas integer fields to timestamps.  The integers must
    contain the number of seconds since unix epoch.

    You must specify the columns you want to transform
    """

    def series_converter(ser):
        return (ser.astype("int64") * 10**9).astype("datetime64[ns]")

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
    Convert a dataframe with start and end times into a dataframe of events.

    This function takes a dataframe with start and end time columns and
    converts it into a sequence of events. For each row in the input
    dataframe, it creates two events: one at the start time and one at the
    end time. Numeric delta columns are added at start time and subtracted at
    end time, while non-delta columns are preserved unchanged.

    Args:
        df (pandas.DataFrame): The input dataframe containing start and end
            times.
        start_time_col (str): Name of the column containing start times.
        end_time_col (str): Name of the column containing end times.
        delta_cols (list, optional): Names of columns to be treated as deltas.
            If None, all non-time columns are treated as deltas. Defaults to
            None.
        non_delta_cols (list, optional): Names of columns to be treated as
            non-deltas. These columns will be preserved unchanged in the
            output. Defaults to None.
        new_time_col_name (str, optional): Name for the new time column in the
            output. Defaults to "time".
        non_numerics_are_index (bool, optional): If True, non-numeric columns
            are combined with the time column in a groupby statement and
            deltas are summed by those groups. Defaults to True.

    Returns:
        pandas.DataFrame: A new dataframe containing the sequence of events,
            ordered by time. Each event has a time and associated delta values.

    Raises:
        ValueError: If delta_cols or non_delta_cols contain start or end time
            columns, or if the same column appears in both delta_cols and
            non_delta_cols.

    Examples:
        >>> df = pd.DataFrame({
        ...     'start': ['2023-01-01', '2023-01-02'],
        ...     'end': ['2023-01-03', '2023-01-04'],
        ...     'value': [10, 20]
        ... })
        >>> events = events_from_starting_ending(
        ...     df=df,
        ...     start_time_col='start',
        ...     end_time_col='end',
        ...     delta_cols=['value']
        ... )
    """
    import pandas as pd

    if delta_cols is None:
        delta_cols = []
    if non_delta_cols is None:
        non_delta_cols = []
    if delta_cols is None:
        cols_to_keep = set(df.columns) - {start_time_col, end_time_col}
        delta_cols = [c for c in df.columns if c in cols_to_keep]
    if set(delta_cols).union(non_delta_cols).intersection({start_time_col, end_time_col}):
        raise ValueError("(Non)delta cols cant contain start or end time col")
    if set(delta_cols).intersection(non_delta_cols):
        raise ValueError("Same column(s) found in both delta and non_delta columns")
    df_start = df[[start_time_col] + non_delta_cols + delta_cols].rename(columns={start_time_col: new_time_col_name})
    df_end = df[[end_time_col] + non_delta_cols + delta_cols].rename(columns={end_time_col: new_time_col_name})
    df_end[delta_cols] = -df_end[delta_cols]
    df = pd.concat([df_start, df_end], ignore_index=True, sort=False)
    df = df.sort_values(by=new_time_col_name)
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
        out = [f"{d}_{calendar.day_abbr[d].lower()}" for d in ser]  # type: ignore
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
    import pandas as pd
    import calendar

    if not isinstance(ser, pd.Series):
        raise ValueError("Input must be a pandas series")
    ser = ser.dt.month
    allowed_kinds = ["slug", "name", "tag"]
    if kind not in allowed_kinds:
        raise ValueError(f"kind must be on of {allowed_kinds}")
    if kind == "tag":
        out = [ascii_lowercase[d] + "_" + calendar.month_abbr[d].lower() for d in ser]  # type: ignore
    elif kind == "slug":
        out = [calendar.month_abbr[d] for d in ser]
    elif kind == "name":
        out = [calendar.month_name[d] for d in ser]
    out = pd.Series(out, index=ser.index)
    return out


def get_quick_schema_class():

    class QuickSchema:

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
            import pandera as pa

            self.dt = pa.dtypes

            self.col_spec = columns.copy()
            if "ordered" not in kwargs:
                kwargs["ordered"] = True
            self.ordered = kwargs["ordered"]
            if "coerce" not in kwargs:
                kwargs["coerce"] = True
            typed_cols = {}
            for key, val in columns.items():
                if isinstance(val, type(pa.Column(self.dt.Int16))):
                    typed_cols[key] = val
                else:
                    typed_cols[key] = pa.Column(val, nullable=True)
            self.schema = pa.DataFrameSchema(typed_cols, **kwargs)

        @classmethod
        def infer_from_dataframe(cls, df):
            """
            Class method that creates a Quickschema object from a dataframe
            """
            import pandera as pa

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

            return sch.from_mapping({col: self.schema.dtypes[col].type for col in self.schema.columns.keys()})

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

    class PandasSql:
        """
        A class for executing SQL queries and returning results as pandas
        DataFrames.

        Args:
            file (str, optional): Path to the database file. Defaults to
                ":memory:" for in-memory database.
            overwrite (bool, optional): Whether to overwrite existing tables.
                Defaults to False.
            **table_mappings: Additional keyword arguments will be treated as
                table mappings.
        """

        def __init__(self, file=":memory:", overwrite=False, **table_mappings):
            """
            If the file is specified and it contains a database with tables,
            you need not register dataframes. However, any dataframes you
            register will overwrite any existing table of that name.

            Uses duckdb sql dialect

            Args:
            file: str = an optional name for a file for the frame records
            overwrite: bool = When true, this will overwrite the existing
                database
            **table_mappings: dict = {table_name1: df1, table_name2: df2, ...}
            """
            if file == ":memory:":
                self.file = file
            else:
                self.file = os.path.realpath(os.path.expanduser(file))
            self.conn = self._get_db_connection(self.file, overwrite)
            self.register(**table_mappings)

        def _get_db_connection(self, file, overwrite):
            import duckdb

            if overwrite and os.path.isfile(file):
                os.unlink(file)
            conn = duckdb.connect(file)
            return conn

        def register(self, **table_mappings):
            """
            Creates tables from dataframes.
            **table_mappings: dict = {
                {table_name1: df1, table_name2: df2, ...}
            }
            """
            if len(table_mappings) == 0:
                return self
            for table_name, _ in table_mappings.items():
                self.conn.execute(
                    "drop table if exists " + table_name + "; create table " + table_name + " as select * from df"
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
    Convert a pandas DataFrame to a compressed hex-encoded string.

    This function serializes the DataFrame using pickle, compresses it using
    LZMA, and then converts the binary data to a hex string. This is useful
    for storing DataFrames in text-based formats or transmitting them over
    text-only channels.

    Args:
        df (pandas.DataFrame): The DataFrame to convert to a hex string.

    Returns:
        str: A hex-encoded string representing the compressed DataFrame.

    Example:
        >>> df = pd.DataFrame({'A': [1, 2, 3]})
        >>> hex_str = hex_from_dataframe(df)
        >>> isinstance(hex_str, str)
        True
    """
    serialized_df = pickle.dumps(df)
    compressed_data = lzma.compress(serialized_df, preset=9)
    hex_encoded_string = binascii.hexlify(compressed_data).decode("utf-8")
    return hex_encoded_string


def hex_to_dataframe(hex_encoded_string):
    """
    Convert a hex-encoded string back to a pandas DataFrame.

    This function reverses the process of hex_from_dataframe by converting the
    hex string back to binary data, decompressing it, and unpickling it to
    reconstruct the original DataFrame.

    Args:
        hex_encoded_string (str): The hex-encoded string to convert back to a
            DataFrame.

    Returns:
        pandas.DataFrame: The reconstructed DataFrame.

    Example:
        >>> hex_str = "..."  # A hex string from hex_from_dataframe
        >>> df = hex_to_dataframe(hex_str)
        >>> isinstance(df, pd.DataFrame)
        True
    """
    compressed_data = binascii.unhexlify(hex_encoded_string)
    serialized_df = lzma.decompress(compressed_data)
    df = pickle.loads(serialized_df)
    return df


def hex_from_duckdb(conn) -> str:
    """
    Exports a DuckDB database to a compressed hex-encoded string.

    This function exports all the tables in the provided database connection
    into a hex string.  The string is maximally compressed.

    Args:
        conn: A DuckDB connection object to export data from

    Returns:
        str: Hex-encoded string representation of the zipped database export

    Example:
        >>> import duckdb
        >>> conn = duckdb.connect(':memory:')
        >>> conn.execute("CREATE TABLE test (id INTEGER, name VARCHAR)")
        >>> conn.execute("INSERT INTO test VALUES (1, 'test')")
        >>> hex_data = hex_from_duckdb(conn)
    """
    with tempfile.TemporaryDirectory() as d:
        conn.execute(f"export database '{d}/ddb_dump'")
        zip_path = f"{d}/ddb_dump.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
            for root, _, files in os.walk(f"{d}/ddb_dump"):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, d)
                    zipf.write(file_path, arcname)
        with open(zip_path, "rb") as f:
            ddb_dump_bytes = f.read()
        ddb_dump_hex = ddb_dump_bytes.hex()
    return ddb_dump_hex


def hex_to_duckdb(hex_dump: str) -> "duckdb.DuckDBPyConnection":  # type: ignore
    """
    Converts a compressed hex-dump of a duckdb database into a new in-memory db.

    This function takes a hex string created by hex_from_duckdb.  It imports
    the compressed data back into a new duckdb memory connection and returns
    that connection.

    Args:
        hex_dump: A hexadecimal string representation of a zipped DuckDB
            database
            dump

    Returns:
        A DuckDB connection with the imported database loaded in memory

    Example:
        >>> db_hex = hex_from_duckdb(conn)
        >>> new_conn = hex_to_duckdb(db_hex)
        >>> new_conn.execute("SELECT * FROM my_table").fetchall()
    """
    import duckdb

    dump_bytes = bytes.fromhex(hex_dump)
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, "ddb_dump.zip")
        with open(zip_path, "wb") as f:
            f.write(dump_bytes)
        extract_dir = os.path.join(temp_dir, "ddb_dump")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)
        new_conn = duckdb.connect(":memory:")
        new_conn.execute(f"IMPORT DATABASE '{extract_dir}'")
        return new_conn
