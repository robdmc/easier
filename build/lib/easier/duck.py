import contextlib
import os
import re
import warnings


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
        return re.sub(r"^df_*", "", self.name)

    def __get__(self, obj, cls):
        if cls is None:  # pragma: no cover
            return

        return obj.query(f"SELECT * FROM {self.table_name}")

    def __set__(self, obj, df=None):
        if df is None:
            return

        if obj._read_only:
            raise ValueError("You can't assign dataframes to a read-only duck")

        with duck_connection(obj.file_name) as con:
            if self.table_name in obj.table_names:
                con.execute(f"DROP TABLE IF EXISTS {self.table_name}")

            con.register("df_in", df)
            con.execute(f"CREATE TABLE {self.table_name} AS SELECT * FROM df_in")


class Duck:
    example = '''
    # Use default file_name
    duck = Duck()

    # Attribute names MUST start with df_ in order to get persisted.
    # The database tablename will be everything following df_
    duck.df_data = df

    # Print a list of all table names in the database
    # You can get frames from these tables by accessing an attribute named "df_<table_name>"
    duck.table_names

    # Accessing the attribute will query all rows from the corresponding
    # table name.
    df = duck.df_data

    # Overwrite any existing specified file
    duck = Duck('./my_database.ddb', overwrite=True)
    duck.df_data = df


    # Run a custom sql query against the database
    df = duck.query("""
        SELECT * FROM table1 JOIN table2 ON field1 LIMIT 10;
    """)
    '''

    def __init__(self, file_name="./duck.ddb", overwrite=False, read_only=False):
        warnings.warn(
            "You are using a deprecated version of Duck.  Use DuckModel instead"
        )
        self.file_name = file_name
        self._read_only = read_only

        if overwrite and os.path.isfile(self.file_name):
            os.unlink(self.file_name)

        if os.path.isfile(self.file_name):
            for table in self.table_names:
                setattr(self, f"df_{table}", None)

    @property
    def table_names(self):
        df = self.query("PRAGMA show_tables")
        return list(df.name)

    def __setattr__(self, name, value):
        import pandas as pd

        if (isinstance(value, pd.DataFrame) or value is None) and name.startswith(
            "df_"
        ):
            setattr(self.__class__, name, Table(name))
        super().__setattr__(name, value)

    def query(self, sql, fetch=True):
        with duck_connection(self.file_name) as con:
            con.execute(sql)
            if fetch:
                df = con.fetchdf()
            else:
                df = None
        return df

    def export_db(self, directory):
        # TODO: Duck db hardcodes the export path.  Try to find a way to make this portable
        directory = os.path.realpath(os.path.expanduser(directory))
        if os.path.isfile(directory) or os.path.isdir(directory):
            raise ValueError(
                f"\n\n {directory!r} already exists.  Cannot overwrite. Nothing done."
            )

        self.query(f"EXPORT DATABASE {directory!r};", fetch=False)

    def import_db(self, directory):
        directory = os.path.realpath(os.path.expanduser(directory))
        if not os.path.isdir(directory):
            raise ValueError(f"\n\n {directory!r} Not found.  Nothing done")

        self.query(f"import DATABASE {directory!r};", fetch=False)
