import contextlib
import os


@contextlib.contextmanager
def duck_connection(duck_file_name):
    import duckdb
    con = duckdb.connect(duck_file_name, read_only=False)
    try:
        yield con
    finally:
        con.close()


def run_query(connection, sql, fetch=True, **kwargs):
    for key, value in kwargs.items():
        connection.register(key, value)
    connection.execute(sql)
    if fetch:
        return connection.fetchdf()


class Table:
    def __init__(self, duck_obj, table_name):
        self.duck = duck_obj
        self.table_name = table_name
        self._exists_dict = {}

    def __dir__(self):  # pragma: no cover
        return [
            'df',
            'head',
            'insert',
        ]

    def ensure_writeable(self):
        if self.duck._read_only:
            raise ValueError('Trying to modify read-only database')

    def query(self, sql, fetch=True, **kwargs):
        return self.duck.query(sql, fetch=fetch, **kwargs)

    @property
    def df(self):
        df = self.query(f'SELECT * FROM {self.table_name}')
        return df

    def head(self, n=5):
        return self.query(f'SELECT * FROM {self.table_name} LIMIT {n}')

    def create(self, df):
        self.drop()
        # with duck_connection(self.duck.file_name) as connection:
        #     connection.register('__df_in__', df)

        self.query(
            f'CREATE TABLE {self.table_name} AS SELECT * FROM __df_in__',
            fetch=False,
            __df_in__=df
        )
        self.exists_dict = {}

    def insert(self, df):
        self.ensure_writeable()
        self.query(
            f"INSERT INTO {self.table_name} SELECT * FROM __df_in__",
            fetch=False,
            __df_in__=df
        )

    def drop(self):
        self.ensure_writeable()
        self.query(f'DROP TABLE IF EXISTS {self.table_name}', fetch=False)


class Tables:
    def __init__(self, duck_obj):
        self.duck = duck_obj

    def __dir__(self):  # pragma: no cover
        return ['create', 'drop', 'drop_all'] + self.duck.table_names

    def create(self, name, df):
        if name == 'create':
            raise ValueError("You cannot name a table 'create'.  It is a reserved word.")

        table = Table(self.duck, name)
        table.create(df)

    def drop(self, name):
        table = getattr(self, name)
        table.drop()

    def drop_all(self):
        table_names = list(self.duck.table_names)
        for name in table_names:
            self.drop(name)

    def __getattr__(self, name):
        if name in self.duck.table_names:
            table = Table(self.duck, name)
            setattr(self, name, table)
        else:
            raise AttributeError(f'No attribute {name}')
        return table

    def __str__(self):  # pragma: no cover
        return repr(self.duck.table_names)

    def __repr__(self):  # pragma: no cover
        return self.__str__()


class DuckModel:
    example = '''
    # Use default file_name
    duck = Duck()

    # Add a table to the database
    duck.tables.create(table_name, df)

    # Insert data to existing table
    duck.tables.<table_name>.insert(df)

    # Drop a table
    duck.tables.drop(table_name)

    # Get a list of all table-names
    duck.table_names
    or
    duck.tables.<tab>

    # Run sql on database
    duck.query(sql, **kwargs) . # (see docstring for kwargs)

    # Drop all tables
    duck.tables.drop_all()


    '''

    def __init__(self, file_name='./duck.ddb', overwrite=False, read_only=False):
        self.file_name = file_name
        self._read_only = read_only

        if overwrite and os.path.isfile(self.file_name):
            os.unlink(self.file_name)

        if overwrite and read_only:
            raise ValueError("It doesn't make sense to set read_only and overwrite at the same time")

        self.tables = Tables(self)

    def query(self, sql, fetch=True, **kwargs):
        """
        Run a sql query across the database.

        Args:
            sql: The sql you want to run
            fetch: If set to True (default) a dataframe will be returned.
                   Otherwise, the query will be executed without returning anything
            **kwargs: This is a tricky one.
                       DuckDB allows you to register dataframes as tables within a database
                       The kwargs to the query allow you to inject fake tables into the database
                       that hold the contents of as specified dataframe.

                       So, for example
                       duck = Duck()
                       duck.tables.create('one', df1)
                       duck.query("select * from one join two on my_id" two=df2)

        """
        with duck_connection(self.file_name) as conn:
            return run_query(conn, sql, fetch=fetch, **kwargs)

    @property
    def table_names(self):
        df = self.query('PRAGMA show_tables')
        return sorted(df.name)
