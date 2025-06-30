from .postgres import pg_creds_from_env
from ibis.expr import schema as sch
from io import StringIO
import ibis
import ibis.expr.datatypes as dtypes
import os
import pandas as pd
import weakref
import contextlib
import warnings


def _get_duck_connection(file_name, overwrite=False, read_only=False):
    """
    A function to get a connection to the local database
    Args:
        reset: if set to True, will blow away any existing database file
    """
    ibis.options.sql.default_limit = None
    if os.path.isfile(file_name) and overwrite:
        os.unlink(file_name)
    conn = ibis.duckdb.connect(file_name, read_only=read_only)
    return conn


def _get_pg_connection(url=None):
    ibis.options.sql.default_limit = None
    if url is None:
        url = pg_creds_from_env()
    conn = ibis.postgres.connect(url=url)
    return conn


@contextlib.contextmanager
def ibis_conn_to_sqlalchemy_conn(ibis_conn):
    engine = ibis_conn.con
    with engine.begin() as conn:
        yield conn


@contextlib.contextmanager
def ibis_postgres_connection(url=None):
    connection = _get_pg_connection(url)
    try:
        proxy_conn = weakref.proxy(connection)
        yield proxy_conn
    finally:
        connection.con.dispose()
        del connection


@contextlib.contextmanager
def ibis_duck_connection(file_name, overwrite=False, read_only=False):
    connection = _get_duck_connection(
        file_name, overwrite=overwrite, read_only=read_only
    )
    try:
        proxy_conn = weakref.proxy(connection)
        yield proxy_conn
    finally:
        connection.con.dispose()
        del connection


def get_sql(expr):
    """
    Returns the SQL for an ibis expression as a string
    """
    buff = StringIO()
    ibis.show_sql(expr, file=buff)
    return buff.getvalue()


def sql_to_frame(conn, sql):
    """
    Returns a pandas dataframe from a SQL query on an ibis connection
    """
    res = conn.raw_sql(sql)
    df = pd.DataFrame(res.fetchall(), columns=res._metadata.keys)
    return df


def get_order_schema_class():

    class OrderedSchema(ibis.Schema):
        dt = dtypes

        def ordered_apply_to(self, df, strict=True):
            """
            Ensures the dataframe will have columns listed
            in the same order as the schema definition.
            Args:
                strict:  If set to True (the default) ensures that the entire
                        schema of the resulting dataframe has been properly typed.
            """
            if not isinstance(df, pd.DataFrame):
                raise ValueError("ordered_appy_to only defined for " "dataframes")
            cols, types = zip(*self.items())
            df = df[list(cols)].copy()

            # Apply type conversions
            for col, typ in zip(cols, types):
                ibis_type = str(typ)
                try:
                    if ibis_type in ("int32", "int64"):
                        if not pd.api.types.is_integer_dtype(df[col]):
                            if df[col].isnull().any():
                                raise TypeError(
                                    f"Column {col} has nulls, cannot convert to "
                                    f"{ibis_type}"
                                )
                            df[col] = pd.to_numeric(df[col], downcast="integer")
                        df[col] = df[col].astype(ibis_type)
                    elif ibis_type in ("float32", "float64"):
                        df[col] = pd.to_numeric(df[col], downcast="float")
                        df[col] = df[col].astype(ibis_type)
                    elif ibis_type == "string":
                        if not pd.api.types.is_string_dtype(df[col]):
                            if (
                                df[col]
                                .dropna()
                                .apply(lambda x: isinstance(x, str))
                                .all()
                            ):
                                df[col] = df[col].astype(str)
                            else:
                                raise TypeError(
                                    f"Column {col} cannot be safely "
                                    f"converted to string"
                                )
                    elif ibis_type == "timestamp":
                        if not pd.api.types.is_datetime64_any_dtype(df[col]):
                            with warnings.catch_warnings():
                                warnings.filterwarnings(
                                    "ignore",
                                    message="Could not infer format, so each element will be parsed individually*",
                                    category=UserWarning,
                                )
                                try:
                                    df[col] = pd.to_datetime(df[col], errors="raise")
                                except Exception:
                                    if strict:
                                        raise TypeError(
                                            f"Column {col} cannot be safely "
                                            f"converted to timestamp"
                                        )
                                    # If not strict, skip conversion and leave column as is
                                    continue
                    else:
                        df[col] = df[col].astype(ibis_type)
                except Exception as e:
                    if strict:
                        raise TypeError(
                            f"Failed to convert column {col} to " f"{ibis_type}: {e}"
                        )
                    # If not strict, skip conversion and leave column as is
                    continue

            if strict and (not df.empty):
                actual = sch.infer(df)
                expected = ibis.Schema(dict(self.items()))
                if actual != expected:
                    s = (
                        f"\n\nExpected Schema:\n{expected}\n--------\n"
                        f"Resulting Schema\n{actual}"
                    )
                    raise TypeError(s)
            return df

    return OrderedSchema
