import contextlib
import os
import weakref
import StringIO


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
