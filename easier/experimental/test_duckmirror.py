import pandas as pd
from duckmirror import Duckmirror
from unittest.mock import patch
import tempfile
import os
import shutil
import pytest


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


def test_register_decorator():
    """Test that the register decorator correctly stores and returns dataframes."""
    # Create a new Duckmirror instance with in-memory database
    mirror = Duckmirror(":memory:")

    # Create a test function that returns a mixed dataframe
    @mirror.register
    def test_table() -> pd.DataFrame:
        return generate_test_df("mixed")

    # Get the dataframe through the mirror
    result_df = mirror.df_test_table

    # Verify the dataframe contents
    expected_df = generate_test_df("mixed")
    pd.testing.assert_frame_equal(result_df, expected_df)

    # Verify we can access the same dataframe multiple times
    result_df2 = mirror.df_test_table
    pd.testing.assert_frame_equal(result_df2, expected_df)


def test_caching_behavior():
    """Test that the caching mechanism works correctly and returns copies."""
    # Create a new Duckmirror instance with in-memory database
    mirror = Duckmirror(":memory:")

    # Track function calls
    call_count = 0

    @mirror.register
    def cached_table() -> pd.DataFrame:
        nonlocal call_count
        call_count += 1
        return generate_test_df("numbers")

    # First call should execute the function
    df1 = mirror.df_cached_table
    assert call_count == 1, "Function should be called on first access"

    # Second call should use cache
    df2 = mirror.df_cached_table
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


def test_cache_contents():
    """Test that the _cache attribute contains a copy of the dataframe only when cache_in_memory=True."""
    # Test with cache_in_memory=True
    mirror = Duckmirror(":memory:", cache_in_memory=True)

    @mirror.register
    def test_table() -> pd.DataFrame:
        return generate_test_df("numbers")

    # First access should populate cache
    df1 = mirror.df_test_table
    expected_df = generate_test_df("numbers")

    # Verify cache contains a copy of the dataframe
    cache_key = "main.test_table"
    assert cache_key in mirror._cache, "Cache should contain the dataframe"
    pd.testing.assert_frame_equal(mirror._cache[cache_key], expected_df)
    assert (
        mirror._cache[cache_key] is not df1
    ), "Cache should contain a copy, not the same object"

    # Test with cache_in_memory=False
    mirror = Duckmirror(":memory:", cache_in_memory=False)

    @mirror.register
    def test_table2() -> pd.DataFrame:
        return generate_test_df("numbers")

    # Access should not populate cache
    df2 = mirror.df_test_table2

    # Verify cache is empty
    assert len(mirror._cache) == 0, "Cache should be empty when cache_in_memory=False"


def test_run_query_call_count():
    # cache_in_memory=True: run_query should be called only once for multiple accesses
    mirror = Duckmirror(":memory:", cache_in_memory=True)
    with patch.object(
        mirror, "run_query_df", wraps=mirror.run_query_df
    ) as mock_run_query:

        @mirror.register
        def test_table() -> pd.DataFrame:
            return generate_test_df("numbers")

        # First access triggers run_query for CREATE TABLE and SELECT
        df1 = mirror.df_test_table
        expected_df = generate_test_df("numbers")
        pd.testing.assert_frame_equal(df1, expected_df)

        # Verify no queries have been called yet
        queries = [call[0][0] for call in mock_run_query.call_args_list]
        select_queries = [q for q in queries if "SELECT * FROM main.test_table" in q]
        assert (
            len(select_queries) == 0
        ), "No SELECT query should have been executed for test_table"

        df2 = mirror.df_test_table
        pd.testing.assert_frame_equal(df2, expected_df)
        assert df1 is not df2, "Dataframes should be different objects (copies)"
        assert (
            len(select_queries) == 0
        ), "No SELECT query should have been executed for test_table"

    mirror = Duckmirror(":memory:", cache_in_memory=False)
    with patch.object(
        mirror, "run_query_df", wraps=mirror.run_query_df
    ) as mock_run_query:

        @mirror.register
        def test_table() -> pd.DataFrame:
            return generate_test_df("numbers")

        queries = [call[0][0] for call in mock_run_query.call_args_list]
        select_queries = [q for q in queries if "SELECT * FROM main.test_table" in q]
        print(select_queries)
        assert (
            len(select_queries) == 0
        ), "No SELECT query should have been executed for test_table"

        # First access triggers run_query for CREATE TABLE and SELECT
        df1 = mirror.df_test_table
        expected_df = generate_test_df("numbers")
        pd.testing.assert_frame_equal(df1, expected_df)

        queries = [call[0][0] for call in mock_run_query.call_args_list]
        select_queries = [q for q in queries if "SELECT * FROM main.test_table" in q]
        assert (
            len(select_queries) == 1
        ), "One SELECT query should have been executed for test_table"

        df2 = mirror.df_test_table
        pd.testing.assert_frame_equal(df2, expected_df)
        assert df1 is not df2, "Dataframes should be different objects (copies)"

        queries = [call[0][0] for call in mock_run_query.call_args_list]
        select_queries = [q for q in queries if "SELECT * FROM main.test_table" in q]
        assert (
            len(select_queries) == 2
        ), "Two SELECT query should have been executed for test_table"


def test_file_based_database():
    """Test Duckmirror functionality with a file-based database."""
    # Create a temporary directory for our test database
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test.db")

    try:
        # Create a new Duckmirror instance with file-based database
        mirror = Duckmirror(db_path, cache_in_memory=False)

        # Track function calls
        call_count = 0

        # Register a test function
        @mirror.register
        def test_table() -> pd.DataFrame:
            nonlocal call_count
            call_count += 1
            return generate_test_df("mixed")

        # First access should create the table and return data
        df1 = mirror.df_test_table
        expected_df = generate_test_df("mixed")
        pd.testing.assert_frame_equal(df1, expected_df)
        assert call_count == 1, "Function should be called once on first access"

        # Verify the database file exists
        assert os.path.exists(db_path), "Database file should be created"

        # Create a new mirror instance pointing to the same database
        mirror2 = Duckmirror(db_path, cache_in_memory=False)

        # Track function calls for second instance
        call_count2 = 0

        # Register the same function
        @mirror2.register
        def test_table() -> pd.DataFrame:
            nonlocal call_count2
            call_count2 += 1
            return generate_test_df("mixed")

        # Access should return the same data
        df2 = mirror2.df_test_table
        pd.testing.assert_frame_equal(df2, expected_df)
        assert (
            call_count2 == 1
        ), "Function should be called once on first access in second instance"

        # Second access should use cache
        df2_2 = mirror2.df_test_table
        pd.testing.assert_frame_equal(df2_2, expected_df)
        assert call_count2 == 1, "Function should not be called again on second access"

        # Modify the function to return different data
        call_count = 0  # Reset counter for modified function

        @mirror.register
        def test_table() -> pd.DataFrame:
            nonlocal call_count
            call_count += 1
            return generate_test_df("numbers")

        # Access should return the new data
        df3 = mirror.df_test_table
        new_expected_df = generate_test_df("numbers")
        pd.testing.assert_frame_equal(df3, new_expected_df)
        assert (
            call_count == 1
        ), "Modified function should be called once on first access"

        # Second access should use cache
        df3_2 = mirror.df_test_table
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

        mirror = Duckmirror(db_path, overwrite=False, cache_in_memory=False)

        @mirror.register
        def test_table() -> pd.DataFrame:
            return pd.DataFrame({"a": [1, 2, 3]})

        df1 = mirror.df_test_table
        pd.testing.assert_frame_equal(df1, pd.DataFrame({"a": [1, 2, 3]}))
        assert os.path.exists(
            db_path
        ), "Database file should exist after first creation"

        # Get initial file creation time
        initial_file_stats = os.stat(db_path)
        initial_creation_time = initial_file_stats.st_ctime

        # Close the connection to ensure file is released
        mirror.conn.close()

        # Phase 2: Overwrite with new data
        # File should still exist before overwrite
        assert os.path.exists(db_path), "Database file should exist before overwrite"

        # Create a new mirror with overwrite=True - this should delete the existing file
        mirror2 = Duckmirror(db_path, overwrite=True, cache_in_memory=False)

        # Verify the file was deleted and recreated by checking creation time
        assert os.path.exists(db_path), "Database file should exist after overwrite"
        new_file_stats = os.stat(db_path)
        new_creation_time = new_file_stats.st_ctime
        assert (
            new_creation_time > initial_creation_time
        ), "File should have been deleted and recreated with a newer timestamp"

        # Register and access the table to ensure it's using the new database
        @mirror2.register
        def test_table() -> pd.DataFrame:
            return pd.DataFrame({"a": [42, 43, 44]})

        df2 = mirror2.df_test_table
        pd.testing.assert_frame_equal(df2, pd.DataFrame({"a": [42, 43, 44]}))

        # Phase 3: Verify data was overwritten
        assert os.path.exists(
            db_path
        ), "Database file should still exist after overwrite"
        assert not df2.equals(
            pd.DataFrame({"a": [1, 2, 3]})
        ), "Old data should not be present after overwrite"

        # Close the second connection
        mirror2.conn.close()

    finally:
        # Clean up
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def test_class_methods_memory_duckmirror():
    """Test Duckmirror with a class (TestData) caching two different methods (df1, df2) in memory."""
    mirror = Duckmirror(":memory:", cache_in_memory=True)

    class TestData:
        @mirror.register
        def df1(self):
            return pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        @mirror.register
        def df2(self):
            return pd.DataFrame({"x": [10, 20, 30], "y": [40, 50, 60]})

    obj = TestData()

    # Access both cached DataFrames via the decorated class
    df1_result = obj.df1
    df2_result = obj.df2

    expected_df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    expected_df2 = pd.DataFrame({"x": [10, 20, 30], "y": [40, 50, 60]})

    pd.testing.assert_frame_equal(df1_result, expected_df1)
    pd.testing.assert_frame_equal(df2_result, expected_df2)

    # Ensure cache returns copies, not the same object
    df1_result2 = obj.df1
    assert df1_result is not df1_result2
    pd.testing.assert_frame_equal(df1_result2, expected_df1)

    # Access both cached DataFrames via the mirror object
    mirror_df1 = mirror.TestData_df1
    mirror_df2 = mirror.TestData_df2

    pd.testing.assert_frame_equal(mirror_df1, expected_df1)
    pd.testing.assert_frame_equal(mirror_df2, expected_df2)

    # Ensure cache returns copies, not the same object
    mirror_df1_2 = mirror.TestData_df1
    assert mirror_df1 is not mirror_df1_2
    pd.testing.assert_frame_equal(mirror_df1_2, expected_df1)
