from collections import namedtuple
from django.db import connection
from pathlib import Path
from typing import List, Dict, Any, Tuple
import copy
import easier as ezr
import functools
import importlib
import jinja2
import os
import urllib.parse

from collections import namedtuple
from typing import List, Dict, Any, Tuple
import copy
import functools
import importlib
import os
import urllib.parse

def pg_creds_from_env(kind='dict', force_docker=False):
    """
    Pulls PostgreSQL credentials from environment variables or defaults to Docker settings.

    This function retrieves database connection credentials from environment variables.
    If environment variables are not set, it defaults to standard Docker PostgreSQL
    credentials. The credentials can be returned either as a dictionary or as a
    connection URL.

    Args:
        kind (str, optional): The format to return credentials in. Must be either
            "dict" or "url". Defaults to "dict".
        force_docker (bool, optional): If True, ignores environment variables and
            returns Docker default credentials. Defaults to False.

    Returns:
        Union[dict, str]: If kind="dict", returns a dictionary with connection
            parameters. If kind="url", returns a PostgreSQL connection URL string.

    Raises:
        ValueError: If kind is not one of the allowed values ("dict" or "url").
    """
    allowed_kinds = ['url', 'dict']
    if kind not in allowed_kinds:
        raise ValueError(f'Allowed kinds are {allowed_kinds}')
    if force_docker:
        env = {}
    else:
        env = os.environ.copy()
    creds = {'host': env.get('PGHOST', 'db'), 'port': env.get('PGPORT', '5432'), 'database': env.get('PGDATABASE', 'postgres'), 'user': env.get('PGUSER', 'postgres'), 'password': env.get('PGPASSWORD', 'postgres')}
    encoded_user = urllib.parse.quote(creds['user'])
    encoded_password = urllib.parse.quote(creds['password'])
    url = f"postgresql://{encoded_user}:{encoded_password}@{creds['host']}:{creds['port']}/{creds['database']}"
    if kind == 'dict':
        return creds
    else:
        return url

class PG:
    """
    A PostgreSQL database connection and query execution class.

    This class provides a convenient interface for connecting to PostgreSQL databases,
    executing queries, and retrieving results in various formats (DataFrame, tuples,
    dictionaries, etc.). It supports both direct connection parameters and Django
    database configurations.

    Args:
        host (str, optional): Database host. Defaults to PGHOST environment variable.
        user (str, optional): Database user. Defaults to PGUSER environment variable.
        password (str, optional): Database password. Defaults to PGPASSWORD environment variable.
        dbname (str, optional): Database name. Defaults to PGDATABASE environment variable.
        use_django (bool, optional): If True, use Django database settings. Defaults to False.

    Attributes:
        _sql (str): The SQL query to be executed
        _context (dict): Context for query execution
        _conn_kwargs (dict): Database connection parameters
        _raw_results (list): Raw query results
        _raw_columns (list): Column names from query results
    """
    _sql = None
    _context = None
    _conn_kwargs = None
    _raw_results = None
    _raw_columns = None

    def __init__(self, **kwargs):
        """
        Initialize a PostgreSQL database connection.

        Args:
            **kwargs: Connection parameters that can include:
                - host: Database host (default from PGHOST env var)
                - user: Database user (default from PGUSER env var)
                - password: Database password (default from PGPASSWORD env var)
                - dbname: Database name (default from PGDATABASE env var)
                - use_django: If True, use Django database settings (default: False)

        Using with Django
            After django is loaded, simply construct with

                pg = PG(use_django=True)

            Tables can be shown with
                pg.table_names()

            Django queries can be run with one line
            df = PG(use_django=True).query('SELECT * FROM my_table').df
        """
        use_django = kwargs.get('use_django', False)
        if use_django:
            self.safe_import('django')
            from django.conf import settings
            db = settings.DATABASES['default']
            kwargs = {'host': db['HOST'], 'user': db['USER'], 'password': db['PASSWORD'], 'dbname': db['NAME']}
        env_translator = {'host': 'PGHOST', 'user': 'PGUSER', 'password': 'PGPASSWORD', 'dbname': 'PGDATABASE'}
        conn_kwargs = {key: kwargs.get(key, os.environ.get(env_translator[key], None)) for key in env_translator.keys()}
        bad_keys = []
        for key in conn_kwargs.keys():
            if conn_kwargs[key] is None:
                bad_keys.append(key)
        if bad_keys:
            raise ValueError(f'The following connections params not specified {bad_keys}')
        self._conn_kwargs = conn_kwargs

    @classmethod
    def queryset_to_sql(cls, queryset):
        """
        Transform a Django queryset into formatted SQL that can be used in pg-admin.

        Args:
            queryset: A Django QuerySet object

        Returns:
            str: Formatted SQL query string
        """
        sqlparse = cls.safe_import('sqlparse')
        cls.safe_import('django')
        sql, sql_params = queryset.query.get_compiler(using=queryset.db).as_sql()
        with connection.cursor() as cur:
            query = cur.mogrify(sql, sql_params)
        query = sqlparse.format(query, reindent=True, keyword_case='upper')
        return query

    def schema_names(self):
        """
        Get a list of all schema names in the database.

        Returns:
            pandas.DataFrame: DataFrame containing schema names
        """
        pg = copy.deepcopy(self)
        return pg.query('SELECT nspname FROM pg_catalog.pg_namespace').to_dataframe()

    @functools.lru_cache()
    def table_names(self, schema_name='public'):
        """
        Get a list of table names in the specified schema.

        Args:
            schema_name (str): Name of the schema to query (default: "public")

        Returns:
            pandas.DataFrame: DataFrame containing table names, sorted alphabetically
        """
        pg = copy.deepcopy(self)
        df = pg.query(f"SELECT table_name FROM information_schema.tables WHERE table_schema='{schema_name}'").to_dataframe()
        df = df.sort_values(by='table_name')
        return df

    def query(self, sql) -> 'PG':
        """
        Set the SQL query to be executed.

        Args:
            sql (str): SQL query string

        Returns:
            PG: Self instance for method chaining
        """
        self._sql = sql
        return self

    def run(self) -> 'PG':
        """
        Execute the current SQL query and store results.

        Returns:
            PG: Self instance for method chaining

        Raises:
            psycopg2.ProgrammingError: If there's an error executing the query
        """
        psycopg2 = self.safe_import('psycopg2')
        with psycopg2.connect(**self._conn_kwargs) as connection:
            with connection.cursor() as cursor:
                cursor.execute(self._sql)
                try:
                    self._raw_results = list(cursor.fetchall())
                    self._raw_columns = [col[0] for col in cursor.description]
                except psycopg2.ProgrammingError as e:
                    if str(e) == 'no results to fetch':
                        self._raw_results = []
                        self._raw_columns = []
                    else:
                        raise
        return self

    def reset(self):
        """
        Reset the query results and columns to None.
        """
        self._raw_columns = None
        self._raw_results = None

    @property
    def _results(self) -> List[Tuple]:
        """
        Get the raw query results, executing the query if necessary.

        Returns:
            List[Tuple]: List of result tuples
        """
        if self._raw_results is None:
            self.run()
        return self._raw_results

    @property
    def columns(self) -> List[str]:
        """
        Get the column names from the query results, executing the query if necessary.

        Returns:
            List[str]: List of column names
        """
        if self._raw_columns is None:
            self.run()
        return self._raw_columns

    def as_tuples(self) -> List[Tuple]:
        """
        Get query results as a list of tuples.

        Returns:
            List[Tuple]: List of result tuples
        """
        self.reset()
        return self._results

    def as_dicts(self) -> List[Dict[str, Any]]:
        """
        Get query results as a list of dictionaries.

        Returns:
            List[Dict[str, Any]]: List of dictionaries with column names as keys
        """
        self.reset()
        return [dict(zip(self.columns, row)) for row in self._results]

    def as_named_tuples(self, named_tuple_name='Result') -> List[Any]:
        """
        Get query results as a list of named tuples.

        Args:
            named_tuple_name (str): Name for the named tuple class (default: "Result")

        Returns:
            List[Any]: List of named tuples
        """
        self.reset()
        nt_result = namedtuple(named_tuple_name, self.columns)
        return [nt_result(*row) for row in self._results]

    @classmethod
    def safe_import(cls, module_name, package=None):
        """
        Safely import a module, raising a helpful error if import fails.

        Args:
            module_name (str): Name of the module to import
            package (str, optional): Package name for relative imports

        Returns:
            module: The imported module

        Raises:
            ImportError: If the module cannot be imported
        """
        try:
            imported = importlib.import_module(module_name, package=package)
        except ImportError:
            raise ImportError(f'\n\nNope! This method requires that {module_name} be installed.  You know what to do.')
        return imported

    def as_dataframe(self) -> Any:
        """
        Get query results as a pandas DataFrame.

        Returns:
            pandas.DataFrame: DataFrame containing query results
        """
        self.reset()
        pd = self.safe_import('pandas')
        return pd.DataFrame(self._results, columns=self.columns)

    def to_tuples(self) -> List[Tuple]:
        """
        Alias for as_tuples().

        Returns:
            List[Tuple]: List of result tuples
        """
        return self.as_tuples()

    def to_dicts(self) -> List[Dict[str, Any]]:
        """
        Alias for as_dicts().

        Returns:
            List[Dict[str, Any]]: List of dictionaries with column names as keys
        """
        return self.as_dicts()

    def to_named_tuples(self) -> List[Any]:
        """
        Alias for as_named_tuples().

        Returns:
            List[Any]: List of named tuples
        """
        return self.as_named_tuples()

    def to_dataframe(self) -> Any:
        """
        Alias for as_dataframe().

        Returns:
            pandas.DataFrame: DataFrame containing query results
        """
        return self.as_dataframe()

    @property
    def df(self):
        """
        Property alias for to_dataframe().

        Returns:
            pandas.DataFrame: DataFrame containing query results
        """
        return self.to_dataframe()

    @property
    def sql(self) -> str:
        """
        Get the formatted SQL query string.

        Returns:
            str: Formatted SQL query string
        """
        sqlparse = self.safe_import('sqlparse')
        psycopg2 = self.safe_import('psycopg2')
        with psycopg2.connect(**self._conn_kwargs) as connection:
            with connection.cursor() as cursor:
                sql, params = self._get_prepared_query()
                query = cursor.mogrify(sql, params)
                query = sqlparse.format(query, reindent=True, keyword_case='upper')
        return query

def sql_file_to_df(file_name='sql_query.sql', context_dict=None):
    """
    Load and execute a SQL query from a file, returning the results as a DataFrame.
    Always connects to the postgres database with credentials from the environment.

    Args:
        file_name (str): Path to the SQL file. Defaults to 'sql_query.sql'.
        context_dict (dict, optional): Dictionary of variables to use for template rendering.
            Defaults to None.

    Returns:
        pandas.DataFrame: Results of the SQL query.
    """
    if context_dict is None:
        context_dict = {}
    file_path = Path(file_name).expanduser().resolve()
    dir_path = file_path.parent
    jinja_env = jinja2.Environment(loader=jinja2.FileSystemLoader(dir_path.as_posix()), autoescape=jinja2.select_autoescape())
    template = jinja_env.get_template(file_path.name)
    text = template.render(**context_dict)
    pg = ezr.PG()
    pg.query(text)
    df = pg.to_dataframe()
    return df

def sql_string_to_df(query, context_dict=None):
    """
    Execute a SQL query string with Jinja templating, returning the results as a DataFrame.
    Always connects to the postgres database with credentials from the environment.

    Args:
        query (str): SQL query string with optional Jinja template variables.
        context_dict (dict, optional): Dictionary of variables to use for template rendering.
            Defaults to None.

    Returns:
        pandas.DataFrame: Results of the SQL query.
    """
    if context_dict is None:
        context_dict = {}
    jinja_env = jinja2.Environment(loader=jinja2.BaseLoader(), autoescape=jinja2.select_autoescape())
    template = jinja_env.from_string(query)
    rendered_query = template.render(**context_dict)
    pg = ezr.PG()
    pg.query(rendered_query)
    df = pg.to_dataframe()
    return df