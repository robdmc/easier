from collections import namedtuple
from typing import List, Dict, Any, Tuple
import importlib
import os


class PG:
    _sql = None
    _context = None
    _conn_kwargs = None
    _raw_results = None
    _raw_columns = None

    def __init__(self, **kwargs):
        """
        kwargs = dict(host=None, user=None, password=None, dbname=None)

        Any kwargs that are not supplied will be loaded from the environment.
        Environment variables should be named according to the psql convention:
            kwarg:  environment_var_name
            'host': 'PGHOST',
            'user': 'PGUSER',
            'password': 'PGPASSWORD',
            'dbname': 'PGDATABASE'

        """
        env_translator = {
            'host': 'PGHOST',
            'user': 'PGUSER',
            'password': 'PGPASSWORD',
            'dbname': 'PGDATABASE'
        }
        conn_kwargs = {
            key: kwargs.get(key, os.environ.get(env_translator[key], None))
            for key in env_translator.keys()
        }

        bad_keys = []
        for key in conn_kwargs.keys():
            if conn_kwargs[key] is None:
                bad_keys.append(key)
        if bad_keys:
            raise ValueError(f'The following connections params not specified {bad_keys}')

        self._conn_kwargs = conn_kwargs

    def query(self, sql: str, **context) -> 'PG':
        '''
        sql: SQL query
        **context: Jinja2-style params to interpolate into query

        The sql and context will be processed through the jinjasql formatter.

        An example query showing most of the syntax is show here.

        query = """
            select * from {{ table_name | sqlsafe }}
            where {{field_name}} in {{my_values | inclause}} limit 2; "
        """

        table_name='my_table',
        field_name='my_field',
        my_values=[1, 2, 3]

        See the following link for more documentation
        https://github.com/hashedin/jinjasql
        '''
        self._sql = sql
        self._context = context
        return self

    def _get_prepared_query(self) -> tuple:
        """
        This is just a thin wrapper to around the translation of
        jinja-style templating to database-style parameters.
        """
        jinjasql = self.safe_import('jinjasql')
        if not self._context:
            return self._sql, None
        else:
            query, params = jinjasql.JinjaSql().prepare_query(self._sql, self._context)
            return query, tuple(params)

    def run(self) -> 'PG':
        """
        Runs the query on the database populating instance variables with results
        """
        psycopg2 = self.safe_import('psycopg2')
        with psycopg2.connect(**self._conn_kwargs) as connection:
            with connection.cursor() as cursor:
                sql, params = self._get_prepared_query()
                cursor.execute(sql, vars=params)
                try:
                    self._raw_results = list(cursor.fetchall())
                    self._raw_columns = [col[0] for col in cursor.description]
                except psycopg2.ProgrammingError as e:
                    if str(e) == 'no results to fetch':
                        self._raw_results = []
                        self._raw_columns = []
                    else:  # pragma: no cover  No expected to hit this, but raise just in case
                        raise
        return self

    def reset(self):
        self._raw_columns = None
        self._raw_results = None

    @property
    def _results(self) -> List[Tuple]:
        """
        A list of raw result tuples
        """
        if self._raw_results is None:
            self.run()
        return self._raw_results

    @property
    def columns(self) -> List[str]:
        if self._raw_columns is None:
            self.run()
        return self._raw_columns

    def as_tuples(self) -> List[Tuple]:
        """
        :return: Results as a list of tuples
        """
        self.reset()
        return self._results

    def as_dicts(self) -> List[Dict[str, Any]]:
        """
        :return: Results as a list of dicts
        """
        self.reset()
        return [dict(zip(self.columns, row)) for row in self._results]

    def as_named_tuples(self, named_tuple_name='Result') -> List[Any]:
        """
        :return: Results as a list of named tuples
        """
        self.reset()
        # Ignore typing in here because of unconventional namedtuple usage
        nt_result = namedtuple(named_tuple_name, self.columns)  # type: ignore
        return [nt_result(*row) for row in self._results]  # type: ignore

    def safe_import(self, module_name, package=None):
        try:
            imported = importlib.import_module(module_name, package=package)
        except ImportError:  # pragma: no cover.  Not going to uninstall pandas to test this
            raise ImportError(f'\n\nNope! This method requires that {module_name} be installed.  You know what to do.')

        return imported

    def as_dataframe(self) -> Any:
        """
        :return: Results as a pandas dataframe
        """
        self.reset()
        pd = self.safe_import('pandas')

        return pd.DataFrame(self._results, columns=self.columns)

    def to_tuples(self) -> List[Tuple]:
        """
        alias
        """
        return self.as_tuples()

    def to_dicts(self) -> List[Dict[str, Any]]:
        """
        alias
        """
        return self.as_dicts()

    def to_named_tuples(self) -> List[Any]:
        """
        alias
        """
        return self.as_named_tuples()

    def to_dataframe(self) -> Any:
        """
        alias
        """
        return self.as_dataframe()

    @property
    def df(self):
        return self.to_dataframe()

    @property
    def sql(self) -> str:
        sqlparse = self.safe_import('sqlparse')
        psycopg2 = self.safe_import('psycopg2')
        with psycopg2.connect(**self._conn_kwargs) as connection:
            with connection.cursor() as cursor:
                sql, params = self._get_prepared_query()
                query = cursor.mogrify(sql, params)
                query = sqlparse.format(query, reindent=True, keyword_case='upper')
        return query
