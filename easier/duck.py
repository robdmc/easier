import contextlib
import os
import re
import easier as ezr


@contextlib.contextmanager
def duck_connection(duck_file_name):
    import duckdb
    con = duckdb.connect(duck_file_name, read_only=False)
    try:
        yield con
    finally:
        con.close()

def run_query(connection, sql, fetch=True):
        connection.execute(sql)
        if fetch:
            return connection.fetchdf()

class Table:
    def __init__(self, connection, table_name):
        self.table_name = table_name
        self.connection = connection
        self._exists_dict = {}

    @property
    def exists(self):
        if self._exists_dict.get('exists', None) is None:
            df = self.query('PRAGMA show_tables')
            self._exists_dict['exists'] = self.table_name in list(df.name)
        return self._exists_dict['exists']

    def _ensure_exists(self):
        if not self.exists:
            raise ValueError(f'Table {self.table_name} has not yet been populated in the database')

    # def query_to_frame(self, query):
    #     return self.connection.execute(query).fetchdf()

    # def query_no_frame(self, query):
    #     self.connection.execute(query)
    #     return self

    def query(self, sql, fetch=True):
        return run_query(self.connection, sql, fetch=fetch)

    @property
    def df(self):
        self._ensure_exists()
        df = self.query(f'SELECT * FROM {self.table_name}')
        return df

    def head(self, n=5):
        self._ensure_exists()
        return self.query(f'SELECT * FROM {self.table_name} LIMIT {n}')

    def create(self, df):
        self.drop()
        self.connection.register('__df_in__', df)
        self.query(f'CREATE TABLE {self.table_name} AS SELECT * FROM __df_in__', fetch=False)
        self.exists_dict = {}

    def insert(self, df):
        self._ensure_exists()
        self.connection.register('__df_in__', df)
        self.query(f"INSERT INTO {self.table_name} SELECT * FROM __df_in__", fetch=False)

    def drop(self):
        self.query(f'DROP TABLE IF EXISTS {self.table_name}', fetch=False)


# class Tables:
#     def __init__(self, duck_obj):
#         self._table_names = []
#         for k, v, in kwargs.items():
#             self._table_names.append(k)
#             setattr(self, k, v)

#     def __str__(self):  # pragma: no cover
#         return f'Tables({self._table_names})'

#     def __repr__(self):  # pragma: no cover
#         return self.__str__()

class Duck:
    def __init__(self, file_name='./duck.ddb', overwrite=False, read_only=False):
        self.file_name = file_name
        self._read_only = read_only

        if overwrite and os.path.isfile(self.file_name):
            os.unlink(self.file_name)

    def query(self, sql, fetch=True):
        with duck_connection(self.file_name) as conn:
            return run_query(conn, sql, fetch=fetch)


    # @property
    # def table_names(self):
    #     with duck_connection(self.file_name) as conn
    #     df = self.query('PRAGMA show_tables')
    #     return list(df.name)

    # def __setattr__(self, name, value):
    #     import pandas as pd
    #     if (isinstance(value, pd.DataFrame) or value is None) and name.startswith('df_'):
    #         setattr(self.__class__, name, Table(name))
    #     super().__setattr__(name, value)

    # def query(self, sql, fetch=True):
    #     with duck_connection(self.file_name) as con:
    #         con.execute(sql)
    #         if fetch:
    #             df = con.fetchdf()
    #         else:
    #             df = None
    #     return df

    # def export_db(self, directory):
    #     #TODO: Duck db hardcodes the export path.  Try to find a way to make this portable
    #     directory = os.path.realpath(os.path.expanduser(directory))
    #     if os.path.isfile(directory) or os.path.isdir(directory):
    #         raise ValueError(f'\n\n {directory!r} already exists.  Cannot overwrite. Nothing done.')

    #     self.query(
    #         f'EXPORT DATABASE {directory!r};',
    #         fetch=False
    #     )

    # def import_db(self, directory):
    #     directory = os.path.realpath(os.path.expanduser(directory))
    #     if not os.path.isdir(directory):
    #         raise ValueError(f'\n\n {directory!r} Not found.  Nothing done')

    #     self.query(
    #         f'import DATABASE {directory!r};',
    #         fetch=False
    #     )
