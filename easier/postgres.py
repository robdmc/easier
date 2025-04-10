from collections import namedtuple
from typing import List, Dict, Any, Tuple
import copy
import functools
import importlib
import os
import urllib.parse


def pg_creds_from_env(kind="dict", force_docker=False):
    """
    Pulls postgres credentials from the environment.  If env vars don't exist,
    it will default to the default docker creds.  You can force this behaviour
    by adding force_docker=True
    """
    allowed_kinds = ["url", "dict"]

    if kind not in allowed_kinds:
        raise ValueError(f"Allowed kinds are {allowed_kinds}")

    if force_docker:
        env = {}
    else:
        env = os.environ.copy()

    creds = {
        "host": env.get("PGHOST", "db"),
        "port": env.get("PGPORT", "5432"),
        "database": env.get("PGDATABASE", "postgres"),
        "user": env.get("PGUSER", "postgres"),
        "password": env.get("PGPASSWORD", "postgres"),
    }
    # URL encode username and password to handle special characters
    encoded_user = urllib.parse.quote(creds["user"])
    encoded_password = urllib.parse.quote(creds["password"])
    url = f"postgresql://{encoded_user}:{encoded_password}@{creds['host']}:{creds['port']}/{creds['database']}"

    if kind == "dict":
        return creds
    else:
        return url


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

        Using with Django
        After django is loaded, simply construct with

            pg = PG(use_django=True)

        Tables can be shown with
            pg.table_names()

        Django queries can be run with one line
        df = PG(use_django=True).query('SELECT * FROM my_table').df
        """
        # See if db info should be loaded from django
        use_django = kwargs.get("use_django", False)

        # If so, replace kwargs with values from django settings
        if use_django:
            self.safe_import("django")
            from django.conf import settings

            db = settings.DATABASES["default"]
            kwargs = {
                "host": db["HOST"],
                "user": db["USER"],
                "password": db["PASSWORD"],
                "dbname": db["NAME"],
            }

        env_translator = {
            "host": "PGHOST",
            "user": "PGUSER",
            "password": "PGPASSWORD",
            "dbname": "PGDATABASE",
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
            raise ValueError(
                f"The following connections params not specified {bad_keys}"
            )

        self._conn_kwargs = conn_kwargs

    @classmethod
    def queryset_to_sql(cls, queryset):
        """
        Transform a queryset into pretty sql that can be copy-pasted directly
        into pg-admin
        """
        # Do imports here to avoid dependencies
        sqlparse = cls.safe_import("sqlparse")
        cls.safe_import("django")
        from django.db import connection

        # Compile the query to python db api
        sql, sql_params = queryset.query.get_compiler(using=queryset.db).as_sql()

        # Translate the python query spec into a postgres query
        with connection.cursor() as cur:
            query = cur.mogrify(sql, sql_params)

        # Make the query pretty and return it
        query = sqlparse.format(query, reindent=True, keyword_case="upper")
        return query

    def schema_names(self):
        # Run schema query in a copied version of self so as not to mess with current query
        # on this object
        pg = copy.deepcopy(self)
        return pg.query("SELECT nspname FROM pg_catalog.pg_namespace").to_dataframe()

    @functools.lru_cache()
    def table_names(self, schema_name="public"):
        # Run table query in a copied version of self so as not to mess with current query
        # on this object
        pg = copy.deepcopy(self)
        df = pg.query(
            f"SELECT table_name FROM information_schema.tables WHERE table_schema='{schema_name}'"
        ).to_dataframe()
        df = df.sort_values(by="table_name")
        return df

    def query(self, sql) -> "PG":
        """
        sql: SQL query
        """
        self._sql = sql
        return self

    def run(self) -> "PG":
        """
        Runs the query on the database populating instance variables with results
        """
        psycopg2 = self.safe_import("psycopg2")
        with psycopg2.connect(**self._conn_kwargs) as connection:
            with connection.cursor() as cursor:
                cursor.execute(self._sql)
                try:
                    self._raw_results = list(cursor.fetchall())
                    self._raw_columns = [col[0] for col in cursor.description]
                except psycopg2.ProgrammingError as e:
                    if str(e) == "no results to fetch":
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

    def as_named_tuples(self, named_tuple_name="Result") -> List[Any]:
        """
        :return: Results as a list of named tuples
        """
        self.reset()
        # Ignore typing in here because of unconventional namedtuple usage
        nt_result = namedtuple(named_tuple_name, self.columns)  # type: ignore
        return [nt_result(*row) for row in self._results]  # type: ignore

    @classmethod
    def safe_import(cls, module_name, package=None):
        try:
            imported = importlib.import_module(module_name, package=package)
        except (
            ImportError
        ):  # pragma: no cover.  Not going to uninstall pandas to test this
            raise ImportError(
                f"\n\nNope! This method requires that {module_name} be installed.  You know what to do."
            )

        return imported

    def as_dataframe(self) -> Any:
        """
        :return: Results as a pandas dataframe
        """
        self.reset()
        pd = self.safe_import("pandas")

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
        sqlparse = self.safe_import("sqlparse")
        psycopg2 = self.safe_import("psycopg2")
        with psycopg2.connect(**self._conn_kwargs) as connection:
            with connection.cursor() as cursor:
                sql, params = self._get_prepared_query()
                query = cursor.mogrify(sql, params)
                query = sqlparse.format(query, reindent=True, keyword_case="upper")
        return query


def sql_file_to_df(file_name="sql_query.sql", context_dict=None):
    """
    Load and execute a SQL query from a file, returning the results as a DataFrame.
    Always connects to the postgres database with credentials from the environment.
    Parameters:
    -----------
    file_name : str
        Path to the SQL file (default: 'sql_query.sql')
    context_dict : dict, optional
        Dictionary of variables to use for template rendering.

    Returns:
    --------
    pandas.DataFrame
        Results of the SQL query
    """
    from pathlib import Path
    import jinja2
    import easier as ezr

    if context_dict is None:
        context_dict = {}

    # Expand user directory and get absolute path
    file_path = Path(file_name).expanduser().resolve()

    # Extract the directory the file lives in
    dir_path = file_path.parent

    jinja_env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(dir_path.as_posix()),
        autoescape=jinja2.select_autoescape(),
    )
    template = jinja_env.get_template(file_path.name)
    text = template.render(**context_dict)

    pg = ezr.PG()
    pg.query(text)
    df = pg.to_dataframe()
    return df


def sql_string_to_df(query, context_dict=None):
    """
    Execute a SQL query string with Jinja templating, returning the results as a DataFrame.
    Alwyas connects to the postgres database with credentials from the environment.

    Parameters:
    -----------
    query : str
        SQL query string with optional Jinja template variables
    context_dict : dict, optional
        Dictionary of variables to use for template rendering.
        Defaults to globals().

    Returns:
    --------
    pandas.DataFrame
        Results of the SQL query
    """
    import jinja2
    import easier as ezr

    if context_dict is None:
        context_dict = {}

    # Create a Jinja environment with a string loader
    jinja_env = jinja2.Environment(
        loader=jinja2.BaseLoader(), autoescape=jinja2.select_autoescape()
    )

    # Create a template from the query string
    template = jinja_env.from_string(query)

    # Render the template with the provided context
    rendered_query = template.render(**context_dict)

    # Execute the query and return results as DataFrame
    pg = ezr.PG()
    pg.query(rendered_query)
    df = pg.to_dataframe()
    return df
