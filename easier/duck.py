import contextlib
import os
import re


@contextlib.contextmanager
def duck_connection(duck_file_name):
    import duckdb
    con = duckdb.connect(duck_file_name, read_only=False)
    try:
        yield con
    finally:
        con.close()


class Table:
    def __init__(self, name):
        self.name = name

    @property
    def table_name(self):
        return re.sub(r'^df_*', '', self.name)

    def __get__(self, obj, cls):
        if cls is None:  # pragma: no cover
            return

        return obj.query(f'SELECT * FROM {self.table_name}')

    def __set__(self, obj, df=None):

        if df is None:
            return

        with duck_connection(obj.file_name) as con:
            if self.table_name in obj.table_names:
                con.execute(f'DROP TABLE IF EXISTS {self.table_name}')

            con.register('df_in', df)
            con.execute(f'CREATE TABLE {self.table_name} AS SELECT * FROM df_in')


class Duck:
    def __init__(self, file_name='./duck.ddb', overwrite=False):
        self.file_name = file_name

        if overwrite and os.path.isfile(self.file_name):
            os.unlink(self.file_name)

        if os.path.isfile(self.file_name):
            for table in self.table_names:
                setattr(self, f'df_{table}', None)

    @property
    def table_names(self):
        df = self.query('PRAGMA show_tables')
        return list(df.name)

    def __setattr__(self, name, value):
        import pandas as pd
        if (isinstance(value, pd.DataFrame) or value is None) and name.startswith('df_'):
            setattr(self.__class__, name, Table(name))
        super().__setattr__(name, value)

    def query(self, sql):
        with duck_connection(self.file_name) as con:
            con.execute(sql)
            df = con.fetchdf()
        return df
