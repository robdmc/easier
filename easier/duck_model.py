import os
from textwrap import dedent


class Table:
    def __init__(self, duck_obj, table_name, force_index_join=False):
        self.duck = duck_obj
        self.table_name = table_name
        self._exists_dict = {}
        self.force_index_join = force_index_join

    def __dir__(self):  # pragma: no cover
        return [
            "df",
            "head",
            "insert",
        ]

    def ensure_writeable(self):
        if self.duck._read_only:
            raise ValueError("Trying to modify read-only database")

    def query(self, sql, fetch=True, **kwargs):
        return self.duck.query(sql, fetch=fetch, **kwargs)

    @property
    def df(self):
        df = self.query(f"SELECT * FROM {self.table_name}")
        return df

    def head(self, n=5):
        return self.query(f"SELECT * FROM {self.table_name} LIMIT {n}")

    def create(self, df):
        self.drop()
        self.query(
            f"BEGIN TRANSACTION; CREATE TABLE {self.table_name} AS SELECT * FROM __df_in__; COMMIT",
            fetch=False,
            __df_in__=df,
        )
        self.exists_dict = {}
        self.duck.connection.unregister("__df_in__")

    def insert(self, df):
        self.ensure_writeable()
        self.query(
            f"BEGIN TRANSACTION INSERT INTO {self.table_name} SELECT * FROM __df_in__ COMMIT",
            fetch=False,
            __df_in__=df,
        )
        self.duck.connection.unregister("__df_in__")

    def drop(self):
        self.ensure_writeable()
        self.query(f"DROP TABLE IF EXISTS {self.table_name}", fetch=False)


class Tables:
    def __init__(self, duck_obj):
        self.duck = duck_obj

    def __dir__(self):  # pragma: no cover
        return ["create", "drop", "drop_all"] + [
            t for t in self.duck.table_names if not t.startswith("__idx")
        ]

    def create(self, name, df):
        if not isinstance(name, str):
            raise ValueError("The first argument must be the name")

        if name == "create":
            raise ValueError(
                "You cannot name a table 'create'.  It is a reserved word."
            )

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
            raise AttributeError(f"No attribute {name}")
        return table

    def __str__(self):  # pragma: no cover
        return repr(self.duck.table_names)

    def __repr__(self):  # pragma: no cover
        return self.__str__()


class DuckModel:
    """
    A class for working with duckdb

    print(ezr.DuckModel.example)

    ...
    """

    example = dedent(
        """
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
    """
    )

    def __init__(
        self,
        file_name="./duck.ddb",
        overwrite=False,
        read_only=False,
        force_index_join=False,
    ):
        self.file_name = file_name
        self._read_only = read_only
        self._force_index_join = force_index_join
        self._connection = None

        if overwrite and os.path.isfile(self.file_name):
            os.unlink(self.file_name)
        if overwrite and os.path.isfile(f"{self.file_name}.wal"):
            os.unlink(f"{self.file_name}.wal")

        if overwrite and read_only:
            raise ValueError(
                "It doesn't make sense to set read_only and overwrite at the same time"
            )

        self.reset_connection()
        self.tables = Tables(self)

    def __dir__(self):  # pragma: no cover
        return [
            "tables",
            "table_names",
            "query",
            "explain",
            "reset_connection",
            "set_index",
            "list_indexes",
            "file_name",
        ]

    def reset_connection(self):
        """
        This resets the connection to the database eliminating registered dataframes
        and defined indexes.
        """
        import duckdb

        self._connection = duckdb.connect(self.file_name, read_only=self._read_only)
        if self._force_index_join:
            self.query("PRAGMA force_index_join", fetch=False)

    @property
    def connection(self):
        if self._connection is None:
            self.reset_connection()
        return self._connection

    def drop_connection(self):
        """
        Drops the connection to the database.  If you are using
        ibis sometimes the ibis and DuckModel connections can fight.
        """
        if self._connection is not None:
            self._connection.close()
        self._connection = None

    def set_index(self, table_name, col_name):
        index_name = f"__idx_{table_name}_{col_name}__"
        self.query(
            f"create unique index {index_name} on {table_name} ({col_name})",
            fetch=False,
        )

    def list_indexes(self):
        return self.query("select * from pg_indexes")

    def explain(
        self, sql, **kwargs
    ):  # pragma: no cover because this is just used in debug
        sql = f"explain {sql}"
        res = self.query(sql, **kwargs)
        print(res.explain_value.iloc[0])

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
        for key, value in kwargs.items():
            self.connection.register(key, value)
        self.connection.execute(sql)
        if fetch:
            return self.connection.fetchdf()

    @property
    def table_names(self):
        df = self.query("PRAGMA show_tables")
        return sorted(df.name)

    def __del__(self):
        self.connection.close()
