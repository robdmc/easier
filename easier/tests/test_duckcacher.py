import pandas as pd
from easier.duckcacher import DuckCacher, duckloader_factory
from unittest.mock import patch
import tempfile
import os
import shutil
import pytest
from easier.utils import cached_container
import polars as pl
import duckdb

################ To run tests ######################################
# python -m pytest -v --cov=duckcacher --cov-report=term-missing
######################################################################


def generate_test_df(df_type: str) -> pd.DataFrame:
    """Generate a test dataframe based on the specified type.

    Args:
        df_type (str): Type of dataframe to generate. Options:
            - 'numbers': Simple numeric dataframe
            - 'text': Simple text dataframe
            - 'mixed': Mixed numeric and text dataframe
            - 'empty': Empty dataframe with columns
            - 'dates': Dataframe with dates

    Returns:
        pd.DataFrame: Generated test dataframe
    """
    if df_type == "numbers":
        return pd.DataFrame({"id": [1, 2, 3], "value": [10.5, 20.5, 30.5]})
    elif df_type == "text":
        return pd.DataFrame(
            {"name": ["Alice", "Bob", "Charlie"], "category": ["A", "B", "A"]}
        )
    elif df_type == "mixed":
        return pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "score": [95.5, 88.0, 92.5],
            }
        )
    elif df_type == "empty":
        return pd.DataFrame(columns=["id", "name", "value"])
    elif df_type == "dates":
        return pd.DataFrame(
            {"date": pd.date_range("2024-01-01", periods=3), "value": [100, 200, 300]}
        )
    else:
        raise ValueError(f"Unknown dataframe type: {df_type}")


@pytest.fixture
def temp_db():
    """Create a temporary database file for testing."""
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test.db")
    yield db_path
    shutil.rmtree(temp_dir)


def test_register_decorator(temp_db):
    """Test that the register decorator correctly stores and returns dataframes."""
    # Create a new DuckCacher instance with temporary database
    cache = DuckCacher(temp_db)

    # Create a test function that returns a mixed dataframe
    @cache.register
    def test_table() -> pd.DataFrame:
        return generate_test_df("mixed")

    # Get the dataframe through the cache
    result_df = cache.df_test_table

    # Verify the dataframe contents
    expected_df = generate_test_df("mixed")
    pd.testing.assert_frame_equal(result_df, expected_df)

    # Verify we can access the same dataframe multiple times
    result_df2 = cache.df_test_table
    pd.testing.assert_frame_equal(result_df2, expected_df)


def test_register_as_basic(temp_db):
    """Test that register_as correctly stores and returns dataframes with custom table names."""
    cache = DuckCacher(temp_db)

    @cache.register_as("custom_table")
    def some_function() -> pd.DataFrame:
        return generate_test_df("mixed")

    # Get the dataframe through the cache using the custom name
    result_df = cache.df_custom_table

    # Verify the dataframe contents
    expected_df = generate_test_df("mixed")
    pd.testing.assert_frame_equal(result_df, expected_df)

    # Verify we can access the same dataframe multiple times
    result_df2 = cache.df_custom_table
    pd.testing.assert_frame_equal(result_df2, expected_df)


def test_register_as_function_name_independence(temp_db):
    """Test that the function name doesn't affect the table name when using register_as."""
    cache = DuckCacher(temp_db)

    @cache.register_as("my_table")
    def completely_different_name() -> pd.DataFrame:
        return generate_test_df("numbers")

    # Access should work with the custom table name, not the function name
    result_df = cache.df_my_table
    expected_df = generate_test_df("numbers")
    pd.testing.assert_frame_equal(result_df, expected_df)

    # Verify we can't access using the function name
    with pytest.raises(AttributeError):
        _ = cache.df_completely_different_name


def test_register_as_caching(temp_db):
    """Test that register_as has the same caching behavior as register."""
    cache = DuckCacher(temp_db)
    call_count = 0

    @cache.register_as("cached_table")
    def some_function() -> pd.DataFrame:
        nonlocal call_count
        call_count += 1
        return generate_test_df("numbers")

    # First call should execute the function
    df1 = cache.df_cached_table
    assert call_count == 1, "Function should be called on first access"

    # Second call should use cache
    df2 = cache.df_cached_table
    assert call_count == 1, "Function should not be called again"

    # Verify we get copies of the dataframe
    assert df1 is not df2, "Should get different dataframe objects"
    pd.testing.assert_frame_equal(df1, df2)


def test_register_as_duplicate_table(temp_db):
    """Test that we can't register two functions with the same table name."""
    cache = DuckCacher(temp_db)

    @cache.register_as("duplicate_table")
    def first_function() -> pd.DataFrame:
        return generate_test_df("numbers")

    # Registering another function with the same table name should raise an error
    with pytest.raises(ValueError):

        @cache.register_as("duplicate_table")
        def second_function() -> pd.DataFrame:
            return generate_test_df("text")


def test_caching_behavior(temp_db):
    """Test that the caching mechanism works correctly and returns copies."""
    # Create a new DuckCacher instance with temporary database
    cache = DuckCacher(temp_db)

    # Track function calls
    call_count = 0

    @cache.register
    def cached_table() -> pd.DataFrame:
        nonlocal call_count
        call_count += 1
        return generate_test_df("numbers")

    # First call should execute the function
    df1 = cache.df_cached_table
    assert call_count == 1, "Function should be called on first access"

    # Second call should use cache
    df2 = cache.df_cached_table
    assert call_count == 1, "Function should not be called again"

    # Verify we get copies of the dataframe
    assert df1 is not df2, "Should get different dataframe objects"

    # Verify the dataframes are equal
    pd.testing.assert_frame_equal(df1, df2)

    # Verify modifying one dataframe doesn't affect the other
    df1.loc[0, "value"] = 999.9
    assert (
        df1.loc[0, "value"] != df2.loc[0, "value"]
    ), "Modifying one dataframe should not affect the other"


def test_cache_contents(temp_db):
    """Test that the _cache attribute contains a copy of the dataframe only when cache_in_memory=True."""
    # Test with cache_in_memory=True
    cache = DuckCacher(temp_db, cache_in_memory=True)

    @cache.register
    def test_table() -> pd.DataFrame:
        return generate_test_df("numbers")

    # First access should populate cache
    df1 = cache.df_test_table
    expected_df = generate_test_df("numbers")

    # Verify cache contains a copy of the dataframe
    cache_key = "main.test_table"
    assert cache_key in cache._cache, "Cache should contain the dataframe"
    pd.testing.assert_frame_equal(cache._cache[cache_key], expected_df)
    assert (
        cache._cache[cache_key] is not df1
    ), "Cache should contain a copy, not the same object"

    # Test with cache_in_memory=False
    cache = DuckCacher(temp_db, cache_in_memory=False)

    @cache.register
    def test_table2() -> pd.DataFrame:
        return generate_test_df("numbers")

    # Access should not populate cache
    df2 = cache.df_test_table2

    # Verify cache is empty
    assert len(cache._cache) == 0, "Cache should be empty when cache_in_memory=False"


def test_run_query_call_count(temp_db):
    # cache_in_memory=True: run_query should be called only once for multiple accesses
    cache = DuckCacher(temp_db, cache_in_memory=True)
    with patch.object(
        cache, "run_query_df", wraps=cache.run_query_df
    ) as mock_run_query:

        @cache.register
        def test_table() -> pd.DataFrame:
            return generate_test_df("numbers")

        # First access triggers run_query for CREATE TABLE and SELECT
        df1 = cache.df_test_table
        expected_df = generate_test_df("numbers")
        pd.testing.assert_frame_equal(df1, expected_df)

        # Verify no queries have been called yet
        queries = [call[0][0] for call in mock_run_query.call_args_list]
        select_queries = [q for q in queries if "SELECT * FROM main.test_table" in q]
        assert (
            len(select_queries) == 0
        ), "No SELECT query should have been executed for test_table"

        df2 = cache.df_test_table
        pd.testing.assert_frame_equal(df2, expected_df)
        assert df1 is not df2, "Dataframes should be different objects (copies)"
        assert (
            len(select_queries) == 0
        ), "No SELECT query should have been executed for test_table"

    cache = DuckCacher(temp_db, cache_in_memory=False)
    with patch.object(
        cache, "run_query_df", wraps=cache.run_query_df
    ) as mock_run_query:

        @cache.register
        def test_table() -> pd.DataFrame:
            return generate_test_df("numbers")

        queries = [call[0][0] for call in mock_run_query.call_args_list]
        select_queries = [q for q in queries if "SELECT * FROM main.test_table" in q]
        assert (
            len(select_queries) == 0
        ), "No SELECT query should have been executed for test_table"

        # First access triggers run_query for CREATE TABLE and SELECT
        df1 = cache.df_test_table
        expected_df = generate_test_df("numbers")
        pd.testing.assert_frame_equal(df1, expected_df)

        queries = [call[0][0] for call in mock_run_query.call_args_list]
        select_queries = [q for q in queries if "SELECT * FROM main.test_table" in q]
        assert (
            len(select_queries) == 1
        ), "One SELECT query should have been executed for test_table"

        df2 = cache.df_test_table
        pd.testing.assert_frame_equal(df2, expected_df)
        assert df1 is not df2, "Dataframes should be different objects (copies)"

        queries = [call[0][0] for call in mock_run_query.call_args_list]
        select_queries = [q for q in queries if "SELECT * FROM main.test_table" in q]
        assert (
            len(select_queries) == 2
        ), "Two SELECT query should have been executed for test_table"


def test_file_based_database():
    """Test DuckCacher functionality with a file-based database."""
    # Create a temporary directory for our test database
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test.db")

    try:
        # Cached file should not exist
        assert not os.path.exists(db_path), "Database file should not exist"

        # Create a new DuckCacher instance with file-based database
        cache = DuckCacher(db_path, cache_in_memory=False)

        # Track function calls
        call_count = 0

        # Register a test function
        @cache.register
        def test_table() -> pd.DataFrame:
            nonlocal call_count
            call_count += 1
            return generate_test_df("mixed")

        # Verify the database file exists
        assert os.path.exists(db_path), "Database file should be created"

        # First access should return expected data
        df1 = cache.df_test_table
        expected_df = generate_test_df("mixed")
        pd.testing.assert_frame_equal(df1, expected_df)

        # Function should be called once on first access
        assert call_count == 1, "Function should be called once on first access"

        _ = cache.df_test_table

        # Function should not be called on second call
        assert call_count == 1, "Function should be called only once"

        # Create a new cache instance pointing to the same database
        cache2 = DuckCacher(db_path, cache_in_memory=False)

        # Reset call count for second cache
        call_count = 0

        # Register the same function
        @cache2.register
        def test_table() -> pd.DataFrame:
            nonlocal call_count
            call_count += 1
            return generate_test_df("mixed")

        # Access should return the same data
        df2 = cache2.df_test_table
        pd.testing.assert_frame_equal(df2, expected_df)

        # Second cache should not have called the function
        assert call_count == 0, "Function should not be called on second cache"

        # Modify the function to return different data
        call_count = 0  # Reset counter for modified function

        @cache.register
        def test_table() -> pd.DataFrame:
            nonlocal call_count
            call_count += 1
            return generate_test_df("numbers")

        # Access should return the new data
        df3 = cache.df_test_table
        new_expected_df = generate_test_df("numbers")
        pd.testing.assert_frame_equal(df3, new_expected_df)
        assert (
            call_count == 1
        ), "Modified function should be called once on first access"

        # Second access should use cache
        df3_2 = cache.df_test_table
        pd.testing.assert_frame_equal(df3_2, new_expected_df)
        assert (
            call_count == 1
        ), "Modified function should not be called again on second access"

        # Verify we get copies of the dataframes
        assert df1 is not df2, "Dataframes should be different objects"
        assert df2 is not df3, "Dataframes should be different objects"
        assert df3 is not df3_2, "Dataframes should be different objects (copies)"

        # Verify the dataframes are not equal after modification
        pd.testing.assert_frame_equal(df1, df2)  # Should be equal (both mixed)
        with pytest.raises(AssertionError):
            pd.testing.assert_frame_equal(
                df1, df3
            )  # Should not be equal (mixed vs numbers)

    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)


def test_overwrite_removes_existing_file():
    """Test that the overwrite parameter correctly handles existing database files."""
    # Create a temporary directory for our test
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "overwrite_test.db")

    try:
        # Phase 1: Initial database creation
        assert not os.path.exists(db_path), "Database file should not exist initially"

        cache = DuckCacher(db_path, overwrite=False, cache_in_memory=False)

        @cache.register
        def test_table() -> pd.DataFrame:
            return pd.DataFrame({"a": [1, 2, 3]})

        df1 = cache.df_test_table
        pd.testing.assert_frame_equal(df1, pd.DataFrame({"a": [1, 2, 3]}))
        assert os.path.exists(
            db_path
        ), "Database file should exist after first creation"

        # Get initial file creation time
        initial_file_stats = os.stat(db_path)
        initial_creation_time = initial_file_stats.st_ctime

        # Phase 2: Overwrite with new data
        # File should still exist before overwrite
        assert os.path.exists(db_path), "Database file should exist before overwrite"

        # Create a new cache with overwrite=True - this should delete the existing file
        cache2 = DuckCacher(db_path, overwrite=True, cache_in_memory=False)

        # Verify the file was deleted and recreated by checking creation time
        assert os.path.exists(db_path), "Database file should exist after overwrite"
        new_file_stats = os.stat(db_path)
        new_creation_time = new_file_stats.st_ctime
        assert (
            new_creation_time > initial_creation_time
        ), "File should have been deleted and recreated with a newer timestamp"

        # Register and access the table to ensure it's using the new database
        @cache2.register
        def test_table() -> pd.DataFrame:
            return pd.DataFrame({"a": [42, 43, 44]})

        df2 = cache2.df_test_table
        pd.testing.assert_frame_equal(df2, pd.DataFrame({"a": [42, 43, 44]}))

        # Phase 3: Verify data was overwritten
        assert os.path.exists(
            db_path
        ), "Database file should still exist after overwrite"
        assert not df2.equals(
            pd.DataFrame({"a": [1, 2, 3]})
        ), "Old data should not be present after overwrite"

    finally:
        # Clean up
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def test_getattr_errors(temp_db):
    """Test that __getattr__ raises appropriate errors for invalid attributes."""
    cache = DuckCacher(temp_db)
    with pytest.raises(AttributeError, match="has no attribute 'invalid'"):
        _ = cache.invalid
    with pytest.raises(AttributeError, match="has no attribute 'df_unregistered'"):
        _ = cache.df_unregistered


def test_multiple_cacher_instances_with_mirrored_objects(temp_db):
    """Test caching behavior when multiple DuckCacher instances point to the same database file
    and mirror different instances of the same class."""

    # Create a class with cached_container attributes
    class TestData:
        def __init__(self, df_type):
            self.df_type = df_type

        @cached_container
        def df_numbers(self):
            return generate_test_df("numbers")

        @cached_container
        def df_text(self):
            return generate_test_df("text")

        @cached_container
        def df_mixed(self):
            return generate_test_df("mixed")

    # Create two instances of the class
    data1 = TestData("numbers")
    data2 = TestData("text")

    # Create two DuckCacher instances pointing to the same database file
    cache1 = DuckCacher(temp_db, mirrored_object=data1)
    cache2 = DuckCacher(temp_db, mirrored_object=data2)

    # Access the dataframes through both caches
    df1_numbers = cache1.df_numbers
    df2_numbers = cache2.df_numbers

    # Verify the dataframes are equal since they use the same generator
    pd.testing.assert_frame_equal(df1_numbers, df2_numbers)

    # Verify the text dataframe is the same for both instances
    df1_text = cache1.df_text
    df2_text = cache2.df_text
    pd.testing.assert_frame_equal(df1_text, df2_text)

    # Verify caching works by accessing again
    df1_numbers_2 = cache1.df_numbers
    df2_numbers_2 = cache2.df_numbers

    # Should get copies of the dataframes
    assert df1_numbers is not df1_numbers_2
    assert df2_numbers is not df2_numbers_2

    # But the data should be the same
    pd.testing.assert_frame_equal(df1_numbers, df1_numbers_2)
    pd.testing.assert_frame_equal(df2_numbers, df2_numbers_2)


def test_duck_loader():
    """Test that DuckLoader correctly creates cached attributes for database tables."""
    # Create a temporary directory for our test database
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test.db")

    try:
        # Create a new DuckCacher instance to set up some test tables
        cache = DuckCacher(db_path)

        # Create test tables with and without df_ prefix
        @cache.register
        def test_table1() -> pd.DataFrame:
            return generate_test_df("numbers")

        @cache.register
        def test_table2() -> pd.DataFrame:
            return generate_test_df("text")

        @cache.register
        def df_test_table3() -> pd.DataFrame:
            return generate_test_df("text")

        # Access the tables to ensure they're created
        _ = cache.df_test_table1
        _ = cache.df_test_table2
        _ = cache.df_df_test_table3

        # Create a DuckLoader instance using the factory function
        loader = duckloader_factory(db_path)

        # Test ls() method
        loaded_tables = loader.ls()
        assert len(loaded_tables) == 3, "Should have 3 loaded tables"
        assert (
            "df_test_table1" in loaded_tables
        ), "Should include test_table1 with df_ prefix"
        assert (
            "df_test_table2" in loaded_tables
        ), "Should include test_table2 with df_ prefix"
        assert "df_test_table3" in loaded_tables, "Should include df_test_table3 as is"

        # Test accessing tables with and without df_ prefix
        df1 = loader.df_test_table1  # Should add df_ prefix
        df2 = loader.df_test_table2  # Should add df_ prefix
        df3 = loader.df_test_table3  # Should keep original name with df_ prefix

        # Verify the data is correct
        pd.testing.assert_frame_equal(df1, generate_test_df("numbers"))
        pd.testing.assert_frame_equal(df2, generate_test_df("text"))
        pd.testing.assert_frame_equal(df3, generate_test_df("text"))

        # Verify caching works by accessing again
        df1_2 = loader.df_test_table1
        df2_2 = loader.df_test_table2
        df3_2 = loader.df_test_table3

        # Verify we get copies
        assert df1 is not df1_2, "Should get a copy of the dataframe"
        assert df2 is not df2_2, "Should get a copy of the dataframe"
        assert df3 is not df3_2, "Should get a copy of the dataframe"

        # Verify the data is still correct
        pd.testing.assert_frame_equal(df1_2, generate_test_df("numbers"))
        pd.testing.assert_frame_equal(df2_2, generate_test_df("text"))
        pd.testing.assert_frame_equal(df3_2, generate_test_df("text"))

        # Test accessing with original table names (should fail)
        with pytest.raises(AttributeError):
            _ = loader.test_table1
        with pytest.raises(AttributeError):
            _ = loader._private_table

        # Test accessing non-existent table
        with pytest.raises(AttributeError, match="has no attribute 'df_nonexistent'"):
            _ = loader.df_nonexistent

    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def test_in_memory_database_error():
    """Test that using ':memory:' as file_name raises an error."""
    with pytest.raises(ValueError, match="In-memory databases are not supported"):
        DuckCacher(":memory:")


def test_get_cached_dataframe_error(temp_db):
    """Test error handling in _get_cached_dataframe."""
    cache = DuckCacher(temp_db)

    # Create a class that will raise AttributeError on clone/copy
    class BadDataFrame(pd.DataFrame):
        def clone(self):
            raise AttributeError("Bad attribute error")

        def copy(self, deep=True):  # Fixed: Added deep parameter
            raise AttributeError("Bad attribute error")

    # Register a function that returns our problematic DataFrame
    @cache.register
    def bad_table() -> pd.DataFrame:
        return BadDataFrame({"col": [1, 2, 3]})

    # Accessing the table should raise AttributeError
    with pytest.raises(AttributeError, match="Bad attribute error"):
        _ = cache.df_bad_table


def test_needs_sync_error_handling(temp_db):
    """Test error handling in _needs_sync."""
    cache = DuckCacher(temp_db)

    # Test with empty DataFrame but with columns
    @cache.register
    def empty_table() -> pd.DataFrame:
        return pd.DataFrame(columns=["col1", "col2"])

    # Force a sync to create the table
    cache.sync("empty_table")

    # The function should handle empty DataFrames gracefully
    assert cache._needs_sync("main", "empty_table", "some_hash")


def test_sync_error_handling(temp_db):
    """Test error handling in sync method."""
    cache = DuckCacher(temp_db)

    # Test with non-existent table
    with pytest.raises(ValueError, match="No registered function found for table"):
        cache.sync("non_existent_table")

    # Test with invalid table name
    with pytest.raises(ValueError, match="No registered function found for table"):
        cache.sync("invalid/table/name")


def test_register_as_error_handling(temp_db):
    """Test error handling in register_as decorator."""
    cache = DuckCacher(temp_db)

    # Test with invalid table name
    with pytest.raises(
        ValueError,
        match=(
            "Invalid table name 'invalid/name'. Table names must start with a letter or underscore "
            "and contain only alphanumeric characters and underscores."
        ),
    ):

        @cache.register_as("invalid/name")
        def invalid_name_func() -> pd.DataFrame:
            return pd.DataFrame({"col": [1, 2, 3]})

    # Register first function
    @cache.register_as("test_table")
    def first_func() -> pd.DataFrame:
        return pd.DataFrame({"col": [1, 2, 3]})

    # Try to register another function with same table name
    with pytest.raises(
        ValueError, match="Table name 'test_table' is already registered"
    ):

        @cache.register_as("test_table")
        def second_func() -> pd.DataFrame:
            return pd.DataFrame({"col": [4, 5, 6]})


def test_duck_loader_error_handling(temp_db):
    """Test error handling in DuckLoader."""
    # Create a DuckLoader instance
    loader = duckloader_factory(temp_db)

    # Test accessing non-existent table
    with pytest.raises(AttributeError):
        _ = loader.non_existent_table

    # Test accessing table with invalid name
    with pytest.raises(AttributeError):
        _ = loader.invalid_table_name


def test_run_query_df_polars(temp_db):
    """Test run_query_df returns polars DataFrame when use_polars=True."""
    cache = DuckCacher(temp_db, use_polars=True)

    @cache.register
    def test_table() -> pd.DataFrame:
        return pd.DataFrame({"a": [1, 2, 3]})

    # Trigger table creation
    _ = cache.df_test_table
    # Now run a query
    result = cache.run_query_df("SELECT * FROM main.test_table")
    assert isinstance(result, pl.DataFrame)
    assert result.shape == (3, 1)
    assert result.columns == ["a"]


def test_needs_sync_polars(temp_db):
    """Test _needs_sync with a polars DataFrame result."""
    cache = DuckCacher(temp_db)
    # Insert a row into duckcacher_state using polars
    df = pl.DataFrame({"code_hash": ["abc"]})
    # Patch run_query_df to return a polars DataFrame
    cache.run_query_df = lambda *a, **k: df
    # Should return True if hash doesn't match
    assert cache._needs_sync("main", "func", "not_abc")
    # Should return False if hash matches
    assert not cache._needs_sync("main", "func", "abc")


def test_duckloader_factory_polars(tmp_path):
    """Test duckloader_factory returns polars DataFrame when use_polars=True."""
    db_path = tmp_path / "test_polars.db"
    cache = DuckCacher(str(db_path))

    @cache.register
    def test_table() -> pd.DataFrame:
        return pd.DataFrame({"a": [1, 2, 3]})

    _ = cache.df_test_table
    loader = duckloader_factory(str(db_path), use_polars=True)
    df = loader.df_test_table
    assert isinstance(df, pl.DataFrame)
    assert df.shape == (3, 1)
    assert df.columns == ["a"]


def test_sync_nonexistent_table(temp_db):
    """Test sync raises ValueError when trying to sync a non-existent table."""
    cache = DuckCacher(temp_db)

    # Register a table first so we can see it in the error message
    @cache.register_as("test_table")
    def test_func() -> pd.DataFrame:
        return pd.DataFrame({"a": [1, 2, 3]})

    with pytest.raises(
        ValueError,
        match="No registered function found for table 'nonexistent_table'. Valid tables are: test_table",
    ):
        cache.sync("nonexistent_table")


def test_register_as_duplicate(temp_db):
    """Test register_as raises ValueError for a duplicate table name."""
    cache = DuckCacher(temp_db)

    @cache.register_as("test_table")
    def first_func() -> pd.DataFrame:
        return pd.DataFrame({"a": [1, 2, 3]})

    with pytest.raises(
        ValueError, match="Table name 'test_table' is already registered"
    ):

        @cache.register_as("test_table")
        def second_func() -> pd.DataFrame:
            return pd.DataFrame({"a": [4, 5, 6]})


def test_sync_nonexistent_attribute(temp_db):
    """Test sync raises ValueError when accessing a non-existing attribute."""
    cache = DuckCacher(temp_db)
    with pytest.raises(ValueError, match="No registered function found for table"):
        cache.sync("df_not_existing")


def test_sync_single_table(temp_db):
    """Test that sync correctly updates a single table and maintains data integrity."""
    cache = DuckCacher(temp_db)

    # Register a function that returns a simple dataframe
    @cache.register
    def foo() -> pd.DataFrame:
        return pd.DataFrame({"a": [1, 2, 3]})

    # First access should create the table
    df1 = cache.df_foo
    pd.testing.assert_frame_equal(df1, pd.DataFrame({"a": [1, 2, 3]}))

    # Modify the function to return different data
    @cache.register
    def foo() -> pd.DataFrame:
        return pd.DataFrame({"a": [4, 5, 6]})

    # Sync should update the table with new data
    cache.sync("foo")

    # Verify the data was updated
    df2 = cache.df_foo
    pd.testing.assert_frame_equal(df2, pd.DataFrame({"a": [4, 5, 6]}))

    # Verify we get copies of the dataframe
    assert df1 is not df2, "Should get different dataframe objects"

    # Verify the dataframes are not equal after modification
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(df1, df2)

    # Test sync without table name (should sync all tables)
    @cache.register
    def bar() -> pd.DataFrame:
        return pd.DataFrame({"b": [7, 8, 9]})

    # First access should create the table
    df3 = cache.df_bar
    pd.testing.assert_frame_equal(df3, pd.DataFrame({"b": [7, 8, 9]}))

    # Modify both functions
    @cache.register
    def foo() -> pd.DataFrame:
        return pd.DataFrame({"a": [10, 11, 12]})

    @cache.register
    def bar() -> pd.DataFrame:
        return pd.DataFrame({"b": [13, 14, 15]})

    # Sync all tables
    cache.sync()

    # Verify both tables were updated
    df4 = cache.df_foo
    df5 = cache.df_bar

    pd.testing.assert_frame_equal(df4, pd.DataFrame({"a": [10, 11, 12]}))
    pd.testing.assert_frame_equal(df5, pd.DataFrame({"b": [13, 14, 15]}))

    # Verify we get copies of the dataframes
    assert df3 is not df5, "Should get different dataframe objects"

    # Verify the dataframes are not equal after modification
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(df3, df5)


def test_register_with_args(temp_db):
    """Test that register raises ValueError for functions with arguments."""
    cache = DuckCacher(temp_db)

    # Test function with positional args
    with pytest.raises(
        ValueError, match="Function 'func_with_args' cannot have any arguments"
    ):

        @cache.register
        def func_with_args(x, y):
            return pd.DataFrame({"col": [x, y]})

    # Test function with kwargs
    with pytest.raises(
        ValueError, match="Function 'func_with_kwargs' cannot have any arguments"
    ):

        @cache.register
        def func_with_kwargs(x=1, y=2):
            return pd.DataFrame({"col": [x, y]})

    # Test function with *args
    with pytest.raises(
        ValueError, match="Function 'func_with_star_args' cannot have any arguments"
    ):

        @cache.register
        def func_with_star_args(*args):
            return pd.DataFrame({"col": args})

    # Test function with **kwargs
    with pytest.raises(
        ValueError, match="Function 'func_with_star_kwargs' cannot have any arguments"
    ):

        @cache.register
        def func_with_star_kwargs(**kwargs):
            return pd.DataFrame({"col": list(kwargs.values())})


def test_register_as_with_args(temp_db):
    """Test that register_as raises ValueError for functions with arguments."""
    cache = DuckCacher(temp_db)

    # Test function with positional args
    with pytest.raises(
        ValueError, match="Function 'func_with_args' cannot have any arguments"
    ):

        @cache.register_as("table1")
        def func_with_args(x, y):
            return pd.DataFrame({"col": [x, y]})

    # Test function with kwargs
    with pytest.raises(
        ValueError, match="Function 'func_with_kwargs' cannot have any arguments"
    ):

        @cache.register_as("table2")
        def func_with_kwargs(x=1, y=2):
            return pd.DataFrame({"col": [x, y]})

    # Test function with *args
    with pytest.raises(
        ValueError, match="Function 'func_with_star_args' cannot have any arguments"
    ):

        @cache.register_as("table3")
        def func_with_star_args(*args):
            return pd.DataFrame({"col": args})

    # Test function with **kwargs
    with pytest.raises(
        ValueError, match="Function 'func_with_star_kwargs' cannot have any arguments"
    ):

        @cache.register_as("table4")
        def func_with_star_kwargs(**kwargs):
            return pd.DataFrame({"col": list(kwargs.values())})


def test_duckloader_factory_ls_isolation(tmp_path):
    """Test that ls() returns different tables for different database files."""
    # Create two different database files
    db1_path = tmp_path / "test1.db"
    db2_path = tmp_path / "test2.db"

    # Create two different cachers with different tables
    cache1 = DuckCacher(str(db1_path))
    cache2 = DuckCacher(str(db2_path))

    # Register different tables in each cache
    @cache1.register
    def table1():
        return pd.DataFrame({"col1": [1, 2, 3]})

    @cache1.register
    def table2():
        return pd.DataFrame({"col2": [4, 5, 6]})

    @cache2.register
    def table1():
        return pd.DataFrame({"col3": [7, 8, 9]})

    @cache2.register
    def table3():
        return pd.DataFrame({"col4": [10, 11, 12]})

    # Sync both caches to ensure tables are created
    cache1.sync()
    cache2.sync()

    # Create loaders for each database
    loader1 = duckloader_factory(str(db1_path))
    loader2 = duckloader_factory(str(db2_path))

    # Get the list of tables from each loader
    tables1 = set(loader1.ls())
    tables2 = set(loader2.ls())

    # Verify the tables are different
    assert tables1 == {
        "df_table1",
        "df_table2",
    }, "First loader should only see its own tables"
    assert tables2 == {
        "df_table1",
        "df_table3",
    }, "Second loader should only see its own tables"

    # Verify there's no overlap in the actual data
    df1_table1 = loader1.df_table1
    df2_table1 = loader2.df_table1

    assert df1_table1["col1"].tolist() == [
        1,
        2,
        3,
    ], "First loader's table1 should have col1"
    assert df2_table1["col3"].tolist() == [
        7,
        8,
        9,
    ], "Second loader's table1 should have col3"


def test_direct_function_call_populates_cache_and_db(tmp_path):
    """Test that calling a registered function directly populates cache and database."""
    db_path = tmp_path / "test.db"
    cache = DuckCacher(str(db_path))

    @cache.register
    def my_table():
        return pd.DataFrame({"a": [1, 2, 3]})

    # Call the function directly
    df = my_table()

    # The in-memory cache should have the table
    assert "main.my_table" in cache._cache

    # The DuckDB table should exist
    with duckdb.connect(str(db_path)) as conn:
        tables = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
        table_names = {row[0] for row in tables}
        assert "my_table" in table_names

    # Verify the data is correct
    expected_df = pd.DataFrame({"a": [1, 2, 3]})
    pd.testing.assert_frame_equal(df, expected_df)


def test_duckcacher_attribute_isolation(tmp_path):
    """Test that different DuckCacher instances don't interfere with each other's attributes."""
    # Create two different database files
    db1_path = tmp_path / "test1.db"
    db2_path = tmp_path / "test2.db"

    # Create two different cachers
    cache1 = DuckCacher(str(db1_path))
    cache2 = DuckCacher(str(db2_path))

    # Register different functions with the same name in each cache
    @cache1.register
    def table1():
        return pd.DataFrame({"col1": [1, 2, 3]})

    @cache1.register
    def table2():
        return pd.DataFrame({"col2": [4, 5, 6]})

    @cache2.register
    def table1():
        return pd.DataFrame({"col3": [7, 8, 9]})

    @cache2.register
    def table3():
        return pd.DataFrame({"col4": [10, 11, 12]})

    # Access the tables via df_ attributes
    df1_table1 = cache1.df_table1
    df1_table2 = cache1.df_table2
    df2_table1 = cache2.df_table1
    df2_table3 = cache2.df_table3

    # Verify the data is different and correct
    assert df1_table1["col1"].tolist() == [
        1,
        2,
        3,
    ], "First cache's table1 should have col1"
    assert df1_table2["col2"].tolist() == [
        4,
        5,
        6,
    ], "First cache's table2 should have col2"
    assert df2_table1["col3"].tolist() == [
        7,
        8,
        9,
    ], "Second cache's table1 should have col3"
    assert df2_table3["col4"].tolist() == [
        10,
        11,
        12,
    ], "Second cache's table3 should have col4"

    # Verify that each cache only has its own registrations
    assert set(cache1._registrations.keys()) == {"table1", "table2"}
    assert set(cache2._registrations.keys()) == {"table1", "table3"}

    # Verify that accessing non-existent attributes raises AttributeError
    with pytest.raises(AttributeError):
        _ = cache1.df_table3

    with pytest.raises(AttributeError):
        _ = cache2.df_table2


def test_register_as_wrapped_function_call(tmp_path):
    """Test that calling the wrapped function from register_as directly works and covers line 347."""
    db_path = tmp_path / "test.db"
    cache = DuckCacher(str(db_path))

    @cache.register_as("my_table")
    def some_function():
        return pd.DataFrame({"a": [1, 2, 3]})

    # Call the wrapped function directly (this should trigger line 347)
    df = some_function()

    # Verify the data is correct
    expected_df = pd.DataFrame({"a": [1, 2, 3]})
    pd.testing.assert_frame_equal(df, expected_df)

    # Verify the table was created in the database
    with duckdb.connect(str(db_path)) as conn:
        tables = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
        table_names = {row[0] for row in tables}
        assert (
            "some_function" in table_names
        ), "Table should be created with the original function name"

    # Verify the dataframe is in the cache
    assert (
        "main.some_function" in cache._cache
    ), "Dataframe should be in the cache after calling wrapped function"

    # Verify we can also access it via the df_ attribute
    df2 = cache.df_my_table
    pd.testing.assert_frame_equal(df2, expected_df)
