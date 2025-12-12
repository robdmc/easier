import os
import re
from typing import Optional
from .item import Item


def duck_frame_writer(file_name: str, **kwargs) -> None:
    """
    Write pandas DataFrames to a DuckDB file as named tables.

    Creates or overwrites a DuckDB file with tables in the 'main' schema.
    Each kwargs key becomes a table name, and each value must be a pandas DataFrame.
    If the file already exists, it will be deleted and recreated from scratch.

    Args:
        file_name (str): Path to the DuckDB file to create/overwrite.
            Can be a relative or absolute path.
        **kwargs: Table definitions where:
            - key (str): Table name (must be valid DuckDB identifier)
            - value (pd.DataFrame): DataFrame to store as table

    Returns:
        None

    Raises:
        TypeError: If any value is not a pandas DataFrame
        ValueError: If file_name is empty or table name is invalid
        OSError: If file cannot be written (permission denied, invalid path)

    Examples:
        >>> import pandas as pd
        >>> from easier.duck_frame import duck_frame_writer
        >>> df1 = pd.DataFrame({'id': [1, 2, 3], 'value': [10, 20, 30]})
        >>> df2 = pd.DataFrame({'name': ['Alice', 'Bob'], 'age': [25, 30]})
        >>> duck_frame_writer('data.duckdb', users=df1, people=df2)
    """
    import pandas as pd
    import duckdb

    # Validate file_name
    if not file_name or not isinstance(file_name, str):
        raise ValueError("file_name must be a non-empty string")

    # Validate all values are DataFrames and all keys are valid table names
    table_name_pattern = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')

    for table_name, df in kwargs.items():
        # Check DataFrame type
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"Value for table '{table_name}' must be a pandas DataFrame, "
                f"got {type(df).__name__}"
            )

        # Check table name validity
        if not table_name or not isinstance(table_name, str):
            raise ValueError(f"Table name must be a non-empty string, got: {table_name}")

        if not table_name_pattern.match(table_name):
            raise ValueError(
                f"Invalid table name '{table_name}'. Table names must start with a letter "
                "or underscore and contain only letters, numbers, and underscores."
            )

    # Delete existing file if it exists
    if os.path.exists(file_name):
        try:
            os.remove(file_name)
        except OSError as e:
            raise OSError(f"Failed to remove existing file '{file_name}': {e}")

    # Create new database and write tables
    try:
        with duckdb.connect(file_name) as conn:
            for table_name, df in kwargs.items():
                # Register DataFrame as a temporary table
                conn.register(table_name, df)
                # Create permanent table from registered DataFrame
                conn.execute(f"CREATE TABLE main.{table_name} AS SELECT * FROM {table_name}")
    except Exception as e:
        raise OSError(f"Failed to write tables to '{file_name}': {e}")


def duck_frame_reader(file_name: str, *args) -> Item:
    """
    Read tables from a DuckDB file into an Item object containing DataFrames.

    Reads specified tables from the 'main' schema of a DuckDB file.
    If no table names are specified, reads all tables from the main schema.
    Returns an Item object where attribute/key access maps to DataFrames.

    Args:
        file_name (str): Path to the DuckDB file to read from.
            Can be a relative or absolute path.
        *args: Optional table names to read. If not specified, reads all tables
               from the main schema. Table names should match exactly as stored.

    Returns:
        Item: Container object with tables as attributes/keys. Access via:
            - result.table_name (attribute access)
            - result['table_name'] (dictionary access)
            - result.keys() (iterate table names)
            - result.items() (iterate table_name/dataframe pairs)

    Raises:
        FileNotFoundError: If file_name doesn't exist
        ValueError: If file_name is empty, args contain non-strings,
                   or requested table doesn't exist in database
        OSError: If file cannot be read (permission denied, corrupt file)

    Examples:
        >>> from easier.duck_frame import duck_frame_reader
        >>> # Read all tables
        >>> result = duck_frame_reader('data.duckdb')
        >>> df1 = result.users
        >>> df2 = result['people']

        >>> # Read specific tables
        >>> result = duck_frame_reader('data.duckdb', 'users', 'people')
        >>> users_df = result.users
    """
    import duckdb

    # Validate file_name
    if not file_name or not isinstance(file_name, str):
        raise ValueError("file_name must be a non-empty string")

    # Check file exists
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"DuckDB file not found: {file_name}")

    # Validate all args are strings
    for arg in args:
        if not isinstance(arg, str):
            raise ValueError(f"Table names must be strings, got {type(arg).__name__}: {arg}")

    try:
        with duckdb.connect(file_name, read_only=True) as conn:
            # Discover all tables in main schema
            tables_df = conn.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'main'
            """).df()

            available_tables = tables_df['table_name'].tolist()

            # Determine which tables to read
            if args:
                # Use specified tables
                tables_to_read = list(args)

                # Check if all requested tables exist
                missing_tables = set(tables_to_read) - set(available_tables)
                if missing_tables:
                    raise ValueError(
                        f"Requested table(s) not found: {sorted(missing_tables)}. "
                        f"Available tables: {sorted(available_tables)}"
                    )
            else:
                # Read all tables
                tables_to_read = available_tables

            # Read tables into dictionary
            tables_dict = {}
            for table_name in tables_to_read:
                df = conn.execute(f"SELECT * FROM main.{table_name}").df()
                tables_dict[table_name] = df

            # Return as Item object
            return Item(**tables_dict)

    except (FileNotFoundError, ValueError):
        # Re-raise these as-is
        raise
    except Exception as e:
        raise OSError(f"Failed to read from '{file_name}': {e}")
