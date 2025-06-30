import contextlib
import dataset
import os
import pandas as pd
import textwrap

import os
import textwrap
import contextlib

@contextlib.contextmanager
def dataset_connection(file_name):
    conn = dataset.connect(f'sqlite:///{file_name}')
    try:
        yield conn
    finally:
        conn.close()

class Tables:
    """
    This descriptor provides attribute acces to the tables in the db.
    Think of it like the .objects attribute on a Django model
    """

    def __init__(self, **kwargs):
        self._table_names = []
        for k, v in kwargs.items():
            self._table_names.append(k)
            setattr(self, k, v)

    def __str__(self):
        return f'Tables({self._table_names})'

    def __repr__(self):
        return self.__str__()

class MiniTable:
    """
    A wrapper around a dataset table that knows how to dump to
    a pandas dataframe
    """

    def __init__(self, file_name, table_name):
        self._table_name = table_name
        self._file_name = file_name

    def __str__(self):
        return f'MiniTable({self._table_name})'

    def __repr__(self):
        return self.__str__()

    @property
    def df(self):
        with dataset_connection(self._file_name) as connection:
            table = connection[self._table_name]
            df = pd.DataFrame(table.all())
        return df

class table_getter:

    def __get__(self, obj, cls):
        content = {}
        for table_name in obj.table_names:
            content[table_name] = MiniTable(obj.file_name, table_name)
        return Tables(**content)

class MiniModel:
    """
    A class for storing pandas dataframes in sqlite database
    """
    tables = table_getter()
    example = textwrap.dedent("\n        import pandas as pd\n        import easier as ezr\n\n        # Creat dataframe to store\n        df_one = pd.DataFrame(\n            [\n                {'a': 1, 'b': pd.Timestamp('1/1/2022')},\n                {'a': 2, 'b': pd.Timestamp('1/2/2022')}\n            ]\n        )\n\n        # Create second dataframe to store\n        df_two = df_one.copy()\n        df_two.loc[0, 'a'] = 7\n        df_two.loc[0, 'b'] = pd.Timestamp('11/18/2022')\n\n\n        # Create a sqlite file managed by minimodel\n        mm = ezr.MiniModel('rob.sqlite', overwrite=True)\n\n        # Create a table from the first dataframe\n        mm.create('table_one', df_one)\n\n        # Insert additional records to the first table\n        mm.insert('table_one', df_two)\n\n        # Create a second table from the second dataframe\n        # (Note I'm creating the table using insert, which\n        # is also allowed)\n        mm.insert('table_two', df_two)\n\n        # Get the saved table as a dataframe\n        df1 = mm.tables.table_one.df\n        df2 = mm.tables.table_two.df\n        print()\n        print(df1.to_string())\n        print()\n        print(df2.to_string())\n\n        # Create a third dataframe to demonstrate upserting\n        dfup = df1.copy().drop('id', axis=1)\n        dfup.loc[:, 'b'] = pd.Timestamp('11/18/2050')\n\n        # Upsert into table two\n        df3_pre = mm.tables.table_two.df\n        mm.upsert('table_two', ['a'], dfup)\n        df3_post = mm.tables.table_two.df\n        print()\n        print(df3_pre.to_string())\n        print()\n        print(df3_post.to_string())\n    ")

    def __init__(self, file_name='./mini_model.sqlite', overwrite=False, read_only=False):
        """
        Args:
            file_name: The name of the sqlite file
            overwrite: Blow away any existing file with that name and ovewrite
            read_only: Prevents unintentional overwriting of a database
        """
        self.file_name = os.path.realpath(os.path.expanduser(file_name))
        self._read_only = read_only
        if overwrite and os.path.isfile(file_name):
            os.unlink(file_name)

    def __str__(self):
        return f'MiniModel({os.path.basename(self.file_name)})'

    def __repr__(self):
        return self.__str__()

    @property
    def table_names(self):
        """
        A utility property that returns all the table names in the db
        """
        names = []
        if os.path.isfile(self.file_name):
            with dataset_connection(self.file_name) as connection:
                names = connection.inspect.get_table_names()
        return names

    def _framify(self, data):
        """
        A utility method to turn inputs into dataframe.
        Inputs can be dicts, series or dataframes.

        If dicts or series, they are transformed into rows of a dataframe
        """
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
        df = self._framify(data)
        recs = self._listify(df)
        with dataset_connection(self.file_name) as connection:
            connection[table_name].insert_many(recs)
        return self

    def create(self, table_name, data):
        """
        Creates (i.e. overwrites) a table in the database with the supplied data
        """
        self._ensure_writeable()
        if table_name in self.table_names:
            with dataset_connection(self.file_name) as connection:
                connection[table_name].drop()
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
        df = self._framify(data)
        recs = self._listify(df)
        with dataset_connection(self.file_name) as connection:
            connection[table_name].upsert_many(recs, keys)
        return self

    def query(self, sql):
        """
        Run a SQL query against the database
        """
        with dataset_connection(self.file_name) as connection:
            df = pd.DataFrame(connection.query(sql))
        return df