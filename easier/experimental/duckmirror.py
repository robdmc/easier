import pandas as pd
import duckdb
import hashlib
import inspect
from typing import Any, Callable, Dict
import os


class _ManagedConnection:
    """A class that manages DuckDB connections with different behaviors for file and memory databases.

    For file-based databases, it creates new connections on each context entry.
    For in-memory databases, it maintains a single persistent connection.
    """

    def __init__(self, file_name: str):
        self.file_name = file_name
        self._memory_conn = (
            None if file_name != ":memory:" else duckdb.connect(file_name)
        )

    def __enter__(self):
        if self.file_name == ":memory:":
            return self._memory_conn
        return duckdb.connect(self.file_name)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file_name != ":memory:":
            # Only close file-based connections
            return True  # Suppress any exceptions

    @property
    def connection(self):
        """Get the current connection. For memory databases, returns the persistent connection.
        For file databases, creates a new connection."""
        if self.file_name == ":memory:":
            return self._memory_conn
        return duckdb.connect(self.file_name)


class Duckmirror:
    """A class that provides automatic synchronization between Python functions and DuckDB tables.

    Duckmirror automatically tracks changes to decorated functions and their corresponding
    DuckDB tables. It maintains a cache of DataFrames in memory (optional) and provides
    automatic synchronization between function outputs and cached database tables.

    The class supports both standalone functions and class methods as data sources.
    When used with class methods, it creates a schema named after the class and stores
    tables within that schema.

    Attributes:
        file_name (str): Path to the DuckDB database file or ":memory:" for in-memory database
        cache_in_memory (bool): Whether to maintain an in-memory cache of DataFrames
        _cache (Dict[str, pd.DataFrame]): Internal cache of DataFrames
        _registrations (Dict[str, Callable]): Internal registry of decorated functions
        _conn_manager (_ManagedConnection): Connection manager instance
    """

    def __init__(
        self,
        file_name: str = ":memory:",
        overwrite: bool = False,
        cache_in_memory: bool = True,
    ):
        """Initialize a new Duckmirror instance.

        Args:
            file_name (str, optional): Path to the DuckDB database file. Use ":memory:" for
                an in-memory database. Defaults to ":memory:".
            overwrite (bool, optional): Whether to overwrite existing database file.
                If True and file exists, it will be deleted. Defaults to False.
            cache_in_memory (bool, optional): Whether to maintain an in-memory cache of
                DataFrames for faster access. Defaults to True.
        """
        self.file_name = file_name
        self.cache_in_memory = cache_in_memory
        self._cache: Dict[str, pd.DataFrame] = {}
        self._registrations: Dict[str, Callable] = {}

        # Overwrite logic: delete file if overwrite is True and not in-memory
        if overwrite and file_name != ":memory:":
            if os.path.exists(file_name):
                os.remove(file_name)

        # Initialize connection manager
        self._conn_manager = _ManagedConnection(file_name)

        # Create duckmirror schema and state table if they don't exist
        with self._conn_manager as conn:
            conn.execute("CREATE SCHEMA IF NOT EXISTS duckmirror")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS duckmirror.duckmirror_state (
                    schema_name VARCHAR,
                    function_name VARCHAR,
                    code_hash VARCHAR,
                    PRIMARY KEY (schema_name, function_name)
                )
            """
            )

    @property
    def conn(self):
        """Get the current database connection."""
        return self._conn_manager.connection

    def run_query_df(self, query: str, *args, **kwargs) -> pd.DataFrame:
        """Execute a SELECT query and return the result as a DataFrame."""
        with self._conn_manager as conn:
            result = conn.execute(query, parameters=args if args else None)
            return result.df()

    def run_execute(self, query: str, *args, **kwargs):
        """Execute a query that does not return a DataFrame (DDL/DML)."""
        with self._conn_manager as conn:
            conn.execute(query, parameters=args if args else None)

    def _get_code_hash(self, func: Callable) -> str:
        """Get the hash of a function's source code."""
        source = inspect.getsource(func)
        return hashlib.sha256(source.encode()).hexdigest()

    def _needs_sync(self, schema: str, func_name: str, current_hash: str) -> bool:
        """Check if a function needs to be synced based on its hash."""
        df = self.run_query_df(
            """
            SELECT code_hash 
            FROM duckmirror.duckmirror_state 
            WHERE schema_name = ? AND function_name = ?
        """,
            schema,
            func_name,
        )
        if df.empty:
            return True
        return df.iloc[0]["code_hash"] != current_hash

    def _ensure_schema(self, schema: str) -> None:
        """Ensure a schema exists in the database."""
        self.run_execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")

    def _store_dataframe(self, schema: str, table: str, df: pd.DataFrame) -> None:
        """Store a dataframe in the database and optionally in memory."""
        self._ensure_schema(schema)
        with self._conn_manager as conn:
            conn.execute(f"DROP TABLE IF EXISTS {schema}.{table}")
            conn.register("df", df)
            conn.execute(f"CREATE TABLE {schema}.{table} AS SELECT * FROM df")
        cache_key = f"{schema}.{table}"
        if self.cache_in_memory:
            self._cache[cache_key] = df

    def _get_cached_dataframe(self, schema: str, table: str) -> pd.DataFrame:
        """Get a dataframe from cache or database."""
        cache_key = f"{schema}.{table}"
        if self.cache_in_memory and cache_key in self._cache:
            return self._cache[cache_key].copy()
        return self.run_query_df(f"SELECT * FROM {schema}.{table}")

    def register(self, func: Callable) -> Callable:
        """Register a function or method to be synchronized with a DuckDB table."""
        if "." in func.__qualname__:
            # It's a method: register for both class property and Duckmirror lookup
            key = func.__qualname__.replace(".", "_")
            self._registrations[key] = func
            # Also register with just the function name for class property access
            self._registrations[func.__name__] = func
            return property(func)
        else:
            # Standalone function
            key = func.__name__
            self._registrations[key] = func
            return func

    def __getattr__(self, name: str) -> Any:
        """Handle dynamic attribute access for df_<name>."""
        if name.startswith("df_"):
            func_name = name[3:]  # Remove 'df_' prefix
            # Try to find the function by its key
            if func_name in self._registrations:
                func = self._registrations[func_name]
                # Use qualname to distinguish method vs standalone
                if "." in func.__qualname__ and func.__qualname__.count(".") == 1:
                    schema = func.__qualname__.split(".")[0]
                    table_name = func.__name__
                elif "." in func.__qualname__:
                    # If function is nested, treat as standalone
                    schema = "main"
                    table_name = func.__name__
                else:
                    schema = "main"
                    table_name = func.__name__
                instance = None
            else:
                raise AttributeError(
                    f"'{self.__class__.__name__}' has no attribute '{name}'"
                )

            # Get current hash
            current_hash = self._get_code_hash(func)

            # Check if sync is needed
            if self._needs_sync(schema, table_name, current_hash):
                # Run the function
                if instance is not None:
                    df = func(instance)
                else:
                    df = func()

                # Store the result
                self._store_dataframe(schema, table_name, df)

                # Update hash
                self.run_execute(
                    """
                    INSERT OR REPLACE INTO duckmirror.duckmirror_state 
                    (schema_name, function_name, code_hash)
                    VALUES (?, ?, ?)
                """,
                    schema,
                    table_name,
                    current_hash,
                )

            return self._get_cached_dataframe(schema, table_name)
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
