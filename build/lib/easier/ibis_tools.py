import contextlib
import os
import weakref
from io import StringIO


def _get_duck_connection(file_name, overwrite=False, read_only=False):
    """
    A function to get a connection to the local database
    Args:
        reset: if set to True, will blow away any existing database file
    """
    import ibis

    ibis.options.sql.default_limit = None
    if os.path.isfile(file_name) and overwrite:
        os.unlink(file_name)
    conn = ibis.duckdb.connect(file_name, read_only=read_only)
    return conn


def _get_pg_connection(url=None):
    import ibis

    ibis.options.sql.default_limit = None
    from .postgres import pg_creds_from_env

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
    import ibis

    buff = StringIO()
    ibis.show_sql(expr, file=buff)
    return buff.getvalue()


def sql_to_frame(conn, sql):
    """
    Returns a pandas dataframe from a SQL query on an ibis connection
    """
    import pandas as pd

    res = conn.raw_sql(sql)
    df = pd.DataFrame(res.fetchall(), columns=res._metadata.keys)
    return df


def get_order_schema_class():
    import ibis
    import ibis.expr.datatypes as dtypes
    from ibis.expr import schema as sch
    import pandas as pd

    class OrderedSchema(ibis.Schema):
        dt = dtypes

        def ordered_apply_to(self, df, strict=True):
            """
            Ensures the dataframe will have columns listed
            in the same order as the schema definition.
            Args:
                strict:  If set to True (the default) ensures that the entire schema
                        of the resulting dataframe has been properly typed.
            """
            # Make sure the input is a dataframe
            if not isinstance(df, pd.DataFrame):
                raise ValueError("ordered_appy_to only defined for dataframes")

            # Get the columns and datatypes for the schema
            cols, types = zip(*self.items())

            # Limit the frame to have only the desired columns and
            # copy it.  This avoids set with copy as well as ensures
            # that the input frame is not mutated
            df = df[list(cols)].copy()

            # Apply the schema to the frame
            self.apply_to(df)

            # If strict comparison is requested
            if strict and not df.empty:
                # Infer the schema of the dataframe
                actual = sch.infer(df)

                # Created a new ibis schema that mirrors this one
                expected = ibis.Schema(dict(self.items()))

                # If the two schemas are not equal, raise an error
                if actual != expected:
                    s = f"\n\nExpected Schema:\n{expected}\n--------\nResulting Schema\n{actual}"
                    raise TypeError(s)

            return df

    return OrderedSchema
