import os
import textwrap
import contextlib
import warnings


@contextlib.contextmanager
def sqlite_connection(file_name):
    import dataset
    conn = dataset.connect(f'sqlite:///{file_name}')
    try:
        yield conn
    finally:
        conn.close()


@contextlib.contextmanager
def pg_connection(url):
    import dataset
    conn = dataset.connect(url)
    try:
        yield conn
    finally:
        conn.close()


class ConnectorSqlite:
    def __init__(self, file_name):
        self.file_name = file_name

    @property
    def connection(self):
        return sqlite_connection(self.file_name)


class ConnectorPG:
    def __init__(self, url):
        self.url = url

    @property
    def connection(self):
        return pg_connection(self.url)


class Tables:
    """
    This descriptor provides attribute acces to the tables in the db.
    Think of it like the .objects attribute on a Django model
    """
    def __init__(self, **kwargs):
        self._table_names = []
        for k, v, in kwargs.items():
            self._table_names.append(k)
            setattr(self, k, v)

    def __str__(self):  # pragma: no cover
        return f'Tables({self._table_names})'

    def __repr__(self):  # pragma: no cover
        return self.__str__()


class MiniTable:
    """
    A wrapper around a dataset table that knows how to dump to
    a pandas dataframe
    """
    def __init__(self, connector, table_name):
        self._table_name = table_name
        self.connector = connector

    def __str__(self):  # pragma: no cover
        return f'MiniTable({self._table_name})'

    def __repr__(self):  # pragma: no cover
        return self.__str__()

    @property
    def df(self):
        import pandas as pd
        with self.connector.connection as connection:
            table = connection[self._table_name]
            df = pd.DataFrame(table.all())
        return df


class table_getter:
    def __get__(self, obj, cls):
        content = {}
        for table_name in obj.table_names:
            content[table_name] = MiniTable(obj.connector, table_name)
        return Tables(**content)


class MiniModelBase:
    """
    A class for storing pandas dataframes in sqlite database
    """
    # This is a descriptor that provides attribute-style lookup for tables
    # Think of it like the .objects attribute on django models
    tables = table_getter()

    example = textwrap.dedent("""
        import pandas as pd
        import easier as ezr

        # Creat dataframe to store
        df_one = pd.DataFrame(
            [
                {'a': 1, 'b': pd.Timestamp('1/1/2022')},
                {'a': 2, 'b': pd.Timestamp('1/2/2022')}
            ]
        )

        # Create second dataframe to store
        df_two = df_one.copy()
        df_two.loc[0, 'a'] = 7
        df_two.loc[0, 'b'] = pd.Timestamp('11/18/2022')


        # Create a sqlite file managed by minimodel
        mm = ezr.MiniModel('rob.sqlite', overwrite=True)

        # Create a table from the first dataframe
        mm.create('table_one', df_one)

        # Insert additional records to the first table
        mm.insert('table_one', df_two)

        # Create a second table from the second dataframe
        # (Note I'm creating the table using insert, which
        # is also allowed)
        mm.insert('table_two', df_two)

        # Get the saved table as a dataframe
        df1 = mm.tables.table_one.df
        df2 = mm.tables.table_two.df
        print()
        print(df1.to_string())
        print()
        print(df2.to_string())

        # Create a third dataframe to demonstrate upserting
        dfup = df1.copy().drop('id', axis=1)
        dfup.loc[:, 'b'] = pd.Timestamp('11/18/2050')

        # Upsert into table two
        df3_pre = mm.tables.table_two.df
        mm.upsert('table_two', ['a'], dfup)
        df3_post = mm.tables.table_two.df
        print()
        print(df3_pre.to_string())
        print()
        print(df3_post.to_string())
    """)

    def __init__(self, file_name_or_url='', overwrite=False, read_only=False):  # pragma: no cover
        raise NotImplementedError(
            'You must write your own constructor that creates a connector '
            'and allows for overwrite=True/False and read_only=True/False')

    def __str__(self):  # pragma: no cover
        return f'MiniModel({os.path.basename(self.file_name)})'

    def __repr__(self):  # pragma: no cover
        return self.__str__()

    @property
    def table_names(self):
        """
        A utility property that returns all the table names in the db
        """
        names = []
        with self.connector.connection as connection:
            names = connection.inspect.get_table_names()
        return names

    def _framify(self, data):
        """
        A utility method to turn inputs into dataframe.
        Inputs can be dicts, series or dataframes.

        If dicts or series, they are transformed into rows of a dataframe
        """
        import pandas as pd
        if isinstance(data, dict):
            data = pd.Series(data)
        if isinstance(data, pd.Series):
            data = pd.DataFrame(data).T

        if not isinstance(data, pd.DataFrame):
            raise ValueError('data must be a dict, pandas series or pandas dataframe')
        return data

    def _listify(self, df):
        """
        Utility method to turn a dataframe into a list of dicts that dataset
        knows how to insert into the db
        """
        import pandas as pd
        datetime_cols = [c for c in df.columns if isinstance(df[c].iloc[0], pd.Timestamp)]
        recs = df.to_dict('records')
        for col in datetime_cols:
            for rec in recs:
                rec[col] = rec[col].to_pydatetime()
        return recs

    def _ensure_writeable(self):
        if self._read_only:
            raise ValueError('Trying to write to read-only db')

    def insert(self, table_name, data):
        """
        Method for inserting data (series, dict or dataframe) into db
        """
        self._ensure_writeable()

        # Turn the input into a dataframe
        df = self._framify(data)

        # Turn the dataframe into a record list
        recs = self._listify(df)

        # Insert the record list
        with self.connector.connection as connection:
            connection[table_name].insert_many(recs)
        return self

    def create(self, table_name, data):
        """
        Creates (i.e. overwrites) a table in the database with the supplied data
        """
        self._ensure_writeable()

        # Drop the table if it exists
        if table_name in self.table_names:
            with self.connector.connection as connection:
                connection[table_name].drop()

        # Insert the new records
        self.insert(table_name, data)
        return self

    def upsert(self, table_name, keys, data):
        """
        Upsert records into specified table.
        Args:
            table_name: the name of the table to upsert
                  keys: a list of column names that the input data must match
                        in the db for a record to be updated in stead of inserted
                  data: The data to upsert.  Single records can be series or dicts.
                        Multiple records defined using dataframes

        """
        self._ensure_writeable()

        # Turn the data into a dataframe
        df = self._framify(data)

        # Turn a dataframe into lists
        recs = self._listify(df)

        # Do the upsert
        with self.connector.connection as connection:
            connection[table_name].upsert_many(recs, keys)
        return self

    def drop_all_tables(self):
        for table_name in self.table_names:
            self.drop(table_name)

    def drop(self, table_name):
        if self._read_only:
            raise ValueError("Can't drop when in read-only mode")
        if table_name in self.table_names:
            with self.connector.connection as connection:
                connection[table_name].drop()
        else:
            raise ValueError(f'{table_name} not in {self.table_names}')

    def query(self, sql, fetch=True):
        """
        Run a SQL query against the database
        """
        import pandas as pd
        with self.connector.connection as connection:
            res = connection.query(sql)
            if fetch:
                return pd.DataFrame(connection.query(sql))


class MiniModelSqlite(MiniModelBase):
    def __init__(self, file_name='./mini_model.sqlite', overwrite=False, read_only=False):
        """
        Args:
            file_name: The name of the sqlite file
            overwrite: Blow away any existing file with that name and ovewrite
            read_only: Prevents unintentional overwriting of a database
        """
        # Make sure inputs make sense
        if overwrite and read_only:
            raise ValueError("It doesn't make sense to overwrite a database in read-only mode")

        # Store the absolute path to the file and handle overwriting
        self.file_name = os.path.realpath(os.path.expanduser(file_name))
        self._read_only = read_only
        if overwrite and os.path.isfile(file_name):
            os.unlink(file_name)

        self.connector = ConnectorSqlite(self.file_name)

    def __str__(self):  # pragma: no cover
        return f'MiniModel({os.path.basename(self.file_name)})'

    def __repr__(self):  # pragma: no cover
        return self.__str__()


# This is just an alias to the sqlite minimodel for backwards compatibility
class MiniModel(MiniModelSqlite):  # pragma: no cover
    def __init__(self, *args, **kwargs):
        warnings.warn('MiniModel will soon be deprecated.  Use MiniModelSqlite or MiniModelPG')
        super().__init__(*args, **kwargs)


class MiniModelPG(MiniModelBase):
    def __init__(self, overwrite=False, read_only=False):
        """
        Args:
            file_name: The name of the sqlite file
            overwrite: Blow away any existing file with that name and ovewrite
            read_only: Prevents unintentional overwriting of a database
        """
        # Make sure inputs make sense
        if overwrite and read_only:
            raise ValueError("It doesn't make sense to overwrite a database in read-only mode")

        # Store the absolute path to the file and handle overwriting
        self._read_only = read_only

        self.connector = ConnectorPG(self.url)

        if overwrite:
            self.drop_all_tables()

    def __getattr__(self, name):  # pragma: no cover
        # This will be used to get postres url components
        mapper = {
            'host': 'PGHOST',
            'user': 'PGUSER',
            'password': 'PGPASSWORD',
            'database': 'PGDATABASE',
            'port': 'PGPORT',
        }

        if name in mapper:
            try:
                return os.environ[mapper[name]]
            except KeyError:
                raise RuntimeError(f'{mapper[name]} must be in your environment')
        else:
            raise AttributeError(f'{name} is not an attribute')

    @property
    def url(self):
        url = f'postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}'
        return url

    def __str__(self):  # pragma: no cover
        return f'MiniModel(postgres:{self.database})'

    def __repr__(self):  # pragma: no cover
        return self.__str__()
