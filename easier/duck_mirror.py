import os
import dataclasses
import sys
import re
import time


class DuckMirror:
    """
    Mirrors a postgres database to a duckdb file.
    Authentication to the postgres database is read
    from the psql-friendly environment variables.
    """

    @dataclasses.dataclass
    class TableRec:
        source: str
        target: str

    def __init__(self, duckdb_file, overwrite=True):
        """
        Create a DuckMirror object
        Args:
            duckdb_file: The file to write the duckdb database to
            overwrite: If True, the file will be deleted if it exists
        """

        file = os.path.realpath(os.path.expanduser(duckdb_file))
        self.file_name = file
        self.overwrite = overwrite

    def clean(self):
        """
        A utility method to delete the duckdb file
        :return:
        """
        if os.path.isfile(self.file_name):
            os.unlink(self.file_name)

    def get_table_names(self):
        """
        Get the table names from the postgres database
        Returns: a list of TableRec objects
        """
        import duckdb

        conn = duckdb.connect()
        conn.execute("install postgres; load postgres;")
        conn.execute("ATTACH '' AS pg (TYPE POSTGRES, READ_ONLY);")

        df = conn.execute("use pg; show all tables").df()
        records = []
        for tup in df.itertuples():
            target = tup.name
            source = ".".join([tup.database, tup.schema, tup.name])
            records.append(self.TableRec(source=source, target=target))
        del conn
        return records

    def clone_tables(
        self,
        including=None,
        excluding=None,
        including_regex=None,
        excluding_regex=None,
        dry_run=False,
    ):
        """
        Clone the tables from the postgres database to the duckdb file
        Args:
            including: List[str] limit cloning to this  list of target table names
            excluding: List[str] exclude these target table names
        """
        import duckdb
        import pandas as pd

        if including is None:
            including = []
        if excluding is None:
            excluding = []

        if self.overwrite and not dry_run:
            self.clean()
        conn = duckdb.connect(self.file_name)
        try:
            conn.execute("install postgres; load postgres;")
            conn.execute("ATTACH '' AS pg (TYPE POSTGRES, READ_ONLY);")
        except duckdb.BinderException:
            pass

        recs = self.get_table_names()
        if including:
            recs = [r for r in recs if r.target in including]
        if excluding:
            recs = [r for r in recs if r.target not in excluding]

        if including_regex is not None:
            rex = re.compile(including_regex)
            recs = [r for r in recs if rex.match(r.target)]

        if excluding_regex is not None:
            rex = re.compile(excluding_regex)
            recs = [r for r in recs if not rex.match(r.target)]

        nrecs = len(recs)
        then = time.time()
        log_recs = []
        total_time = 0
        for nn, rec in enumerate(recs):
            print(f"{nn}/{nrecs}  {rec.target} : sec=", end="")
            sys.stdout.flush()
            sql = f"DROP TABLE IF EXISTS {rec.target}; CREATE TABLE {rec.target} AS FROM {rec.source};"
            if not dry_run:
                conn.execute(sql)
            now = time.time()
            delta = now - then
            then = now
            print(f"{delta:0.3f}")
            log_recs.append({"target": rec.target, "seconds": delta})
            total_time += delta
        print(f"\n----\nTotal time = {total_time:0.3} seconds")
        df = pd.DataFrame(log_recs)
        df = df.sort_values(by="seconds", ascending=False)
        return df
