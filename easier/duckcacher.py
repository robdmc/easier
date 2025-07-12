import hashlib
from typing import Any, Callable, Dict, Optional, Union
import os
import inspect
import re
from .utils import cached_container

# To run tests:
# python -m pytest -v --cov=duckcacher --cov-report=term-missing


class DuckCacher:
    """A class that provides automatic synchronization between Python functions and DuckDB tables.

    DuckCacher automatically tracks changes to decorated functions and their corresponding
    DuckDB tables. It maintains a cache of DataFrames in memory (optional) and provides
    automatic synchronization between function outputs and cached database tables.

    ---

    Attributes:
        file_name (str): Path to the DuckDB database file
        cache_in_memory (bool): Whether to maintain an in-memory cache of DataFrames
        _cache (Dict[str, pd.DataFrame]): Internal cache of DataFrames
        _registrations (Dict[str, Callable]): Internal registry of decorated functions
        mirrored_object (Optional[Any]): Optional object to mirror cached containers from

    Examples:
        Using DuckCacher as a function decorator:
            # Create a cache instance
            cache = DuckCacher("my_cache.duckdb")

            # Register a function that returns a DataFrame
            @cache.register
            def my_table() -> pd.DataFrame:
                return pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})

            # Access the table via attribute (automatically syncs)
            df = cache.df_my_table
            print(df)
            # Output:
            #    id  value
            # 0   1     10
            # 1   2     20
            # 2   3     30

            # Or call the function directly (also automatically syncs)
            df = my_table()  # This will populate the cache and database

        Using DuckCacher to mirror cached containers from an object:
            # Create a class with cached container methods
            class MyData:
                @cached_container
                def df_numbers(self):
                    return pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})

                @cached_container
                def df_text(self):
                    return pd.DataFrame({"name": ["Alice", "Bob", "Charlie"]})

            # Create an instance and cache it
            data = MyData()
            cache = DuckCacher(":memory:", mirrored_object=data)

            # Access cached tables
            numbers_df = cache.df_numbers  # Access cached table
            text_df = cache.df_text  # Access another cached table
    """

    def __init__(
        self,
        file_name: str,
        mirrored_object: Optional[Any] = None,
        overwrite: bool = False,
        cache_in_memory: bool = True,
        use_polars: bool = False,
    ):
        """Initialize a new DuckCacher instance.

        Args:
            file_name (str): Path to the DuckDB database file. Must be a file path, not ":memory:".
            mirrored_object (Any, optional): Object to mirror cached containers from.
                If provided, will automatically register functions for cached container attributes.
                Defaults to None.
            overwrite (bool, optional): Whether to overwrite existing database file.
                If True and file exists, it will be deleted. Defaults to False.
            cache_in_memory (bool, optional): Whether to maintain an in-memory cache of
                DataFrames for faster access. Defaults to True.
            use_polars (bool, optional): Whether to return polars instead of pandas
        """
        import pandas as pd
        import duckdb

        if file_name == ":memory:":
            raise ValueError("In-memory databases are not supported. Please provide a file path.")

        self.file_name = file_name
        self.cache_in_memory = cache_in_memory
        self._cache: Dict[str, pd.DataFrame] = {}
        self._registrations: Dict[str, Callable] = {}
        self.mirrored_object = mirrored_object
        self.use_polars = use_polars

        # Overwrite logic: delete file if overwrite is True
        if overwrite and os.path.exists(file_name):
            os.remove(file_name)

        # Initialize database with required schema and tables
        with duckdb.connect(self.file_name) as conn:
            conn.execute("CREATE SCHEMA IF NOT EXISTS duckcacher")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS duckcacher.duckcacher_state (
                    schema_name VARCHAR,
                    function_name VARCHAR,
                    code_hash VARCHAR,
                    PRIMARY KEY (schema_name, function_name)
                )
            """
            )

        # Register cached containers if an object was provided
        if mirrored_object is not None:
            self._register_containers(mirrored_object)

    def run_query_df(self, query: str, *args, **kwargs) -> Union["pd.DataFrame", "pl.DataFrame"]:
        """Execute a SELECT query and return the result as a DataFrame."""
        import duckdb
        import polars as pl

        with duckdb.connect(self.file_name) as conn:
            result = conn.execute(query, parameters=args if args else None)
            df = result.df()
            if self.use_polars:
                return pl.from_pandas(df)
            else:
                return df

    def run_execute(self, query: str, *args, **kwargs):
        """Execute a query that does not return a DataFrame (DDL/DML)."""
        import duckdb

        with duckdb.connect(self.file_name) as conn:
            conn.execute(query, parameters=args if args else None)

    def _get_code_hash(self, func: Callable) -> str:
        """Get a stable hash of a function's code object (bytecode and constants)."""
        code = func.__code__
        hash_input = code.co_code + b"|".join(str(const).encode("utf-8") for const in code.co_consts)
        return hashlib.sha256(hash_input).hexdigest()

    def _needs_sync(self, schema: str, func_name: str, current_hash: str) -> bool:
        """Check if a function needs to be synced based on its hash."""
        df = self.run_query_df(
            """
            SELECT code_hash
            FROM duckcacher.duckcacher_state
            WHERE schema_name = ? AND function_name = ?
        """,
            schema,
            func_name,
        )
        if hasattr(df, "empty"):
            if df.empty:
                return True
            code_hash = df.iloc[0]["code_hash"]
        elif hasattr(df, "height"):
            if df.height == 0:
                return True
            code_hash = df["code_hash"][0]  # polars
        return code_hash != current_hash

    def _ensure_schema(self, schema: str) -> None:
        """Ensure a schema exists in the database."""
        self.run_execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")

    def _store_dataframe(self, schema: str, table: str, df: "pd.DataFrame") -> None:
        """Store a dataframe in the database and optionally in memory."""
        import duckdb

        self._ensure_schema(schema)
        with duckdb.connect(self.file_name) as conn:
            conn.execute(f"DROP TABLE IF EXISTS {schema}.{table}")
            conn.register("df", df)
            conn.execute(f"CREATE TABLE {schema}.{table} AS SELECT * FROM df")
        cache_key = f"{schema}.{table}"
        if self.cache_in_memory:
            self._cache[cache_key] = df

    def _get_cached_dataframe(self, schema: str, table: str) -> Union["pd.DataFrame", "pl.DataFrame"]:
        """Get a dataframe from cache or database.

        Returns:
            pd.DataFrame | pl.DataFrame: The requested dataframe, either from cache or database.
            The type depends on self.use_polars.
        """
        import polars as pl

        cache_key = f"{schema}.{table}"

        # Try to get from cache first
        if self.cache_in_memory and cache_key in self._cache:
            cached_df = self._cache[cache_key]
            try:
                result = cached_df.clone() if hasattr(cached_df, "clone") else cached_df.copy()
            except AttributeError:  # pragma: no cover
                raise AttributeError("Bad attribute error")
        else:
            # Get from database if not in cache
            result = self.run_query_df(f"SELECT * FROM {schema}.{table}")
            # Store in cache if caching is enabled
            if self.cache_in_memory:
                self._cache[cache_key] = result

        # Convert to polars if requested
        return pl.from_pandas(result) if self.use_polars else result

    def sync(self, table: str = None) -> None:
        """Sync one or all tables with their registered functions.

        Args:
            table (str, optional): Name of the table to sync. If None, all tables are synced.
        """
        if table is not None:
            if table not in self._registrations:
                valid_tables = sorted(list(self._registrations.keys()))
                msg = (
                    f"No registered function found for table '{table}'. " f"Valid tables are: {', '.join(valid_tables)}"
                )
                raise ValueError(msg)
            tables_to_sync = [table]
        else:
            tables_to_sync = list(self._registrations.keys())

        schema = "main"
        for table_name in tables_to_sync:
            func = self._registrations[table_name]

            # Run the function to get fresh data
            df = func()

            # Store the result
            self._store_dataframe(schema, table_name, df)

            # Update hash
            current_hash = self._get_code_hash(func)
            self.run_execute(
                """
                INSERT OR REPLACE INTO duckcacher.duckcacher_state 
                (schema_name, function_name, code_hash)
                VALUES (?, ?, ?)
            """,
                schema,
                table_name,
                current_hash,
            )

    def _validate_no_args(self, func: Callable) -> None:
        """Validate that a function has no arguments.

        Args:
            func (Callable): The function to validate

        Raises:
            ValueError: If the function has any arguments
        """
        sig = inspect.signature(func)
        if len(sig.parameters) > 0:
            raise ValueError(
                f"Function '{func.__name__}' cannot have any arguments. "
                f"Found parameters: {list(sig.parameters.keys())}"
            )

    def register(self, func: Callable) -> Callable:
        """Register a function to be synchronized with a DuckDB table.

        Returns a wrapped function that automatically populates the cache when called.
        """
        import pandas as pd
        import polars as pl

        self._validate_no_args(func)
        key = func.__name__
        self._registrations[key] = func

        def wrapped_func() -> pd.DataFrame | pl.DataFrame:
            """Wrapped function that automatically syncs to cache when called."""
            # Simply access the attribute on the cacher object, which handles all the sync logic
            return getattr(self, f"df_{func.__name__}")

        # Preserve the original function's metadata
        wrapped_func.__name__ = func.__name__
        wrapped_func.__doc__ = func.__doc__
        wrapped_func.__module__ = func.__module__

        return wrapped_func

    def register_as(self, table_name: str) -> Callable:
        """Register a function to be synchronized with a DuckDB table using a custom table name.

        Args:
            table_name (str): The name to use for the table and attribute (without 'df_' prefix)

        Returns:
            Callable: A decorator that registers the function with the specified table name.
                     Returns a wrapped function that automatically populates the cache when called.

        Example:
            @cache.register_as('my_table')
            def anything() -> pd.DataFrame:
                return pd.DataFrame({'col': [1, 2, 3]})

            # Access via df_my_table
            df = cache.df_my_table

            # Or call the function directly to populate cache
            df = anything()  # This will automatically sync to the database
        """

        def is_valid_table_name(name: str) -> bool:
            # Only allow alphanumeric and underscores, must start with a letter or underscore
            return re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name) is not None

        def decorator(func: Callable) -> Callable:
            import pandas as pd
            import polars as pl

            self._validate_no_args(func)
            if not is_valid_table_name(table_name):
                raise ValueError(
                    f"Invalid table name '{table_name}'. Table names must start with a letter or underscore "
                    f"and contain only alphanumeric characters and underscores."
                )
            if table_name in self._registrations:
                raise ValueError(f"Table name '{table_name}' is already registered")
            self._registrations[table_name] = func

            def wrapped_func() -> Union["pd.DataFrame", "pl.DataFrame"]:
                """Wrapped function that automatically syncs to cache when called."""
                # Simply access the attribute on the cacher object, which handles all the sync logic
                return getattr(self, f"df_{table_name}")

            # Preserve the original function's metadata
            wrapped_func.__name__ = func.__name__
            wrapped_func.__doc__ = func.__doc__
            wrapped_func.__module__ = func.__module__

            return wrapped_func

        return decorator

    def __getattr__(self, name: str) -> Any:
        """Handle dynamic attribute access for registered functions.

        If the attribute starts with 'df_', it will be treated as a registered function
        by removing the prefix. For all other attributes, Python's default behavior is used.
        """
        if name.startswith("df_"):
            func_name = name[3:]  # Remove 'df_' prefix
            if func_name in self._registrations:
                func = self._registrations[func_name]
                schema = "main"
                table_name = func.__name__

                # Get current hash
                current_hash = self._get_code_hash(func)

                # Check if sync is needed
                if self._needs_sync(schema, table_name, current_hash):
                    # Run the function
                    df = func()

                    # Store the result
                    self._store_dataframe(schema, table_name, df)

                    # Update hash
                    self.run_execute(
                        """
                        INSERT OR REPLACE INTO duckcacher.duckcacher_state 
                        (schema_name, function_name, code_hash)
                        VALUES (?, ?, ?)
                    """,
                        schema,
                        table_name,
                        current_hash,
                    )

                return self._get_cached_dataframe(schema, table_name)
        # Let Python handle the attribute lookup
        return object.__getattribute__(self, name)

    def _register_containers(self, obj: Any) -> None:
        """Register functions for cached container attributes from the provided object.

        This method inspects the class of the provided object for attributes that are
        either cached_container or pickle_cached_container decorators. For each such
        attribute, it creates and registers a corresponding function that returns the
        cached value.

        Args:
            obj: An instance of a class with cached container attributes
        """
        # Get all attributes of the class
        for name, attr in inspect.getmembers(obj.__class__):
            # Skip private attributes
            if name.startswith("_"):
                continue

            # Check if the attribute is a cached container
            if hasattr(attr, "__class__") and attr.__class__.__name__ in (
                "cached_container",
                "pickle_cached_container",
            ):
                # Create a function that returns the cached value
                def make_getter(attr_name: str) -> Callable[[], "pd.DataFrame"]:
                    import pandas as pd

                    def getter() -> "pd.DataFrame":
                        return getattr(obj, attr_name)

                    return getter

                # Register the function, removing 'df_' prefix if present
                func_name = name[3:] if name.startswith("df_") else name
                getter_func = make_getter(name)
                getter_func.__name__ = func_name
                self.register(getter_func)


def duckloader_factory(file_name: str, schema: str = "main", use_polars: bool = False) -> "DuckLoader":
    """Factory function that creates a DuckLoader instance with dynamic table access.

    This factory function introspects a DuckDB database to create a DuckLoader class
    with cached container methods for each table in the specified schema. Each table
    becomes accessible as a cached property on the DuckLoader instance.

    Parameters
    ----------
    file_name : str
        Path to the DuckDB database file. Must be a file path, not ":memory:".
    schema : str, optional
        Schema to introspect for tables, by default "main"
    use_polars : bool, optional
        Whether to return polars DataFrames instead of pandas, by default False

    Returns
    -------
    DuckLoader
        An instance of DuckLoader with dynamic table access methods

    Examples
    --------
    ```python
    # Create a loader instance
    loader = duck_loader_factory("my_db.duckdb")

    # Access tables as properties
    df = loader.df_my_table

    # List available tables
    tables = loader.ls()
    ```
    """

    # Get list of tables in the schema
    import duckdb

    with duckdb.connect(file_name) as conn:
        tables_df = conn.execute(
            f"""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = '{schema}'
        """
        ).df()

    def ls() -> list[str]:
        """List all tables in the DuckDB schema.

        Returns:
            list[str]: List of table names, with 'df_' prefix added if not present.
        """
        return [t if t.startswith("df_") else f"df_{t}" for t in tables_df["table_name"].tolist()]

    def make_getter(tbl_name: str) -> Callable[[], "pd.DataFrame"]:
        """Create a cached container getter method for a specific table.

        Args:
            tbl_name (str): Name of the table to create a getter for.

        Returns:
            Callable[[], pd.DataFrame]: A cached container method that retrieves the table data.
        """

        @cached_container
        def getter(self) -> "pd.DataFrame":
            """Retrieve data from the specified table.

            Returns:
                pd.DataFrame: The table data, converted to polars DataFrame if use_polars is True.
            """
            import pandas as pd
            import polars as pl
            import duckdb

            with duckdb.connect(file_name) as conn:
                df = conn.execute(f"SELECT * FROM {schema}.{tbl_name}").df()
                if use_polars:
                    return pl.from_pandas(df)
                else:
                    return df

        return getter

    class DuckLoader:
        """A class that provides dynamic access to tables in a DuckDB database.

        This class is dynamically created with cached container methods for each table
        in the specified schema. Each table becomes accessible as a property on the
        instance, with the 'df_' prefix added if not present in the original table name.

        Attributes:
            ls (Callable[[], list[str]]): Method to list all available attributes.
            df_* (pd.DataFrame): Dynamic properties for each table in the database.
        """

        def ls(self) -> list[str]:
            """List all tables in the DuckDB schema.

            Returns:
                list[str]: List of table names, with 'df_' prefix added if not present.
            """
            return ls()

    for table_name in tables_df["table_name"]:
        attr_name = f"df_{table_name}" if not table_name.startswith("df_") else table_name
        setattr(DuckLoader, attr_name, make_getter(table_name))

    return DuckLoader()
