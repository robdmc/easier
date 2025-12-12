import pandas as pd
import pytest
import tempfile
import os
import shutil
import duckdb
from easier.duck_frame import duck_frame_writer, duck_frame_reader
from easier.item import Item


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def temp_db(temp_dir):
    """Create a temporary database file path for testing."""
    return os.path.join(temp_dir, "test.duckdb")


def test_duck_frame_writer_basic(temp_db):
    """Test writing a single table to DuckDB file."""
    # Create test DataFrame
    df = pd.DataFrame({'id': [1, 2, 3], 'value': [10, 20, 30]})

    # Write to DuckDB
    duck_frame_writer(temp_db, test_table=df)

    # Verify file was created
    assert os.path.exists(temp_db)

    # Verify table exists and has correct data
    with duckdb.connect(temp_db) as conn:
        result = conn.execute("SELECT * FROM main.test_table").df()
        pd.testing.assert_frame_equal(result, df)


def test_duck_frame_writer_multiple_tables(temp_db):
    """Test writing multiple tables in one call."""
    # Create test DataFrames
    df1 = pd.DataFrame({'id': [1, 2, 3], 'value': [10, 20, 30]})
    df2 = pd.DataFrame({'name': ['Alice', 'Bob'], 'age': [25, 30]})
    df3 = pd.DataFrame({'x': [1.1, 2.2], 'y': [3.3, 4.4]})

    # Write to DuckDB
    duck_frame_writer(temp_db, users=df1, people=df2, coords=df3)

    # Verify all tables exist
    with duckdb.connect(temp_db) as conn:
        tables_df = conn.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main'
        """).df()

        table_names = sorted(tables_df['table_name'].tolist())
        assert table_names == ['coords', 'people', 'users']

        # Verify data integrity for each table
        result1 = conn.execute("SELECT * FROM main.users").df()
        result2 = conn.execute("SELECT * FROM main.people").df()
        result3 = conn.execute("SELECT * FROM main.coords").df()

        pd.testing.assert_frame_equal(result1, df1)
        pd.testing.assert_frame_equal(result2, df2)
        pd.testing.assert_frame_equal(result3, df3)


def test_duck_frame_writer_overwrite(temp_db):
    """Test that writing to existing file deletes and recreates it."""
    # First write
    df1 = pd.DataFrame({'id': [1, 2, 3], 'value': [10, 20, 30]})
    df2 = pd.DataFrame({'name': ['Alice', 'Bob'], 'age': [25, 30]})
    duck_frame_writer(temp_db, table1=df1, table2=df2)

    # Second write with different tables and data
    df3 = pd.DataFrame({'x': [100, 200], 'y': [300, 400]})
    duck_frame_writer(temp_db, new_table=df3)

    # Verify only new table exists (old tables should be gone)
    with duckdb.connect(temp_db) as conn:
        tables_df = conn.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main'
        """).df()

        table_names = tables_df['table_name'].tolist()
        assert table_names == ['new_table']

        # Verify new data
        result = conn.execute("SELECT * FROM main.new_table").df()
        pd.testing.assert_frame_equal(result, df3)


def test_duck_frame_writer_validation(temp_db):
    """Test input validation for duck_frame_writer."""
    df = pd.DataFrame({'id': [1, 2, 3], 'value': [10, 20, 30]})

    # Test TypeError for non-DataFrame values
    with pytest.raises(TypeError, match="must be a pandas DataFrame"):
        duck_frame_writer(temp_db, table1=df, table2=[1, 2, 3])

    with pytest.raises(TypeError, match="must be a pandas DataFrame"):
        duck_frame_writer(temp_db, table1="not a dataframe")

    # Test ValueError for invalid table names
    with pytest.raises(ValueError, match="Invalid table name"):
        duck_frame_writer(temp_db, **{'123invalid': df})

    with pytest.raises(ValueError, match="Invalid table name"):
        duck_frame_writer(temp_db, **{'table-name': df})

    with pytest.raises(ValueError, match="Invalid table name"):
        duck_frame_writer(temp_db, **{'table name': df})

    # Test empty file_name
    with pytest.raises(ValueError, match="file_name must be a non-empty string"):
        duck_frame_writer("", table1=df)


def test_duck_frame_writer_empty_dataframe(temp_db):
    """Test writing an empty DataFrame (no rows but with columns)."""
    # Create empty DataFrame with columns
    df = pd.DataFrame(columns=['id', 'name', 'value'])

    # Should succeed
    duck_frame_writer(temp_db, empty_table=df)

    # Verify table structure
    with duckdb.connect(temp_db) as conn:
        result = conn.execute("SELECT * FROM main.empty_table").df()
        assert len(result) == 0
        assert list(result.columns) == ['id', 'name', 'value']


def test_duck_frame_reader_all_tables(temp_db):
    """Test reading all tables when no args provided."""
    # Write test data
    df1 = pd.DataFrame({'id': [1, 2, 3], 'value': [10, 20, 30]})
    df2 = pd.DataFrame({'name': ['Alice', 'Bob'], 'age': [25, 30]})
    duck_frame_writer(temp_db, users=df1, people=df2)

    # Read all tables
    result = duck_frame_reader(temp_db)

    # Verify result is an Item
    assert isinstance(result, Item)

    # Verify tables are accessible via attribute access
    pd.testing.assert_frame_equal(result.users, df1)
    pd.testing.assert_frame_equal(result.people, df2)

    # Verify tables are accessible via dict access
    pd.testing.assert_frame_equal(result['users'], df1)
    pd.testing.assert_frame_equal(result['people'], df2)

    # Verify keys() method works
    assert set(result.keys()) == {'users', 'people'}


def test_duck_frame_reader_specific_tables(temp_db):
    """Test reading specific tables using *args."""
    # Write test data
    df1 = pd.DataFrame({'id': [1, 2, 3], 'value': [10, 20, 30]})
    df2 = pd.DataFrame({'name': ['Alice', 'Bob'], 'age': [25, 30]})
    df3 = pd.DataFrame({'x': [1.1, 2.2], 'y': [3.3, 4.4]})
    duck_frame_writer(temp_db, users=df1, people=df2, coords=df3)

    # Read only specific tables
    result = duck_frame_reader(temp_db, 'users', 'coords')

    # Verify only requested tables are present
    assert set(result.keys()) == {'users', 'coords'}
    pd.testing.assert_frame_equal(result.users, df1)
    pd.testing.assert_frame_equal(result.coords, df3)

    # Verify 'people' is not present
    with pytest.raises(AttributeError):
        _ = result.people


def test_duck_frame_reader_empty_database(temp_db):
    """Test reading from an empty database."""
    # Create empty database
    with duckdb.connect(temp_db) as conn:
        pass  # Just create the file

    # Read all tables (should be empty)
    result = duck_frame_reader(temp_db)

    # Verify result is an Item with no attributes
    assert isinstance(result, Item)
    assert len(result.keys()) == 0


def test_duck_frame_reader_missing_table(temp_db):
    """Test error handling when requesting non-existent table."""
    # Write test data
    df1 = pd.DataFrame({'id': [1, 2, 3], 'value': [10, 20, 30]})
    duck_frame_writer(temp_db, users=df1)

    # Request non-existent table
    with pytest.raises(ValueError, match="Requested table.*not found"):
        duck_frame_reader(temp_db, 'nonexistent')

    # Error message should list available tables
    with pytest.raises(ValueError, match="Available tables.*users"):
        duck_frame_reader(temp_db, 'missing_table')


def test_duck_frame_reader_file_not_found():
    """Test error handling when file doesn't exist."""
    with pytest.raises(FileNotFoundError, match="DuckDB file not found"):
        duck_frame_reader('/nonexistent/path/file.duckdb')


def test_duck_frame_reader_validation(temp_db):
    """Test input validation for duck_frame_reader."""
    df = pd.DataFrame({'id': [1, 2, 3], 'value': [10, 20, 30]})
    duck_frame_writer(temp_db, users=df)

    # Test empty file_name
    with pytest.raises(ValueError, match="file_name must be a non-empty string"):
        duck_frame_reader("")

    # Test non-string table names in args
    with pytest.raises(ValueError, match="Table names must be strings"):
        duck_frame_reader(temp_db, 'users', 123)


def test_round_trip_data_integrity(temp_db):
    """Test data integrity for various data types in write-read cycle."""
    # Create DataFrame with various data types
    df = pd.DataFrame({
        'int_col': [1, 2, 3, 4],
        'float_col': [1.1, 2.2, 3.3, 4.4],
        'str_col': ['a', 'b', 'c', 'd'],
        'bool_col': [True, False, True, False],
        'datetime_col': pd.date_range('2024-01-01', periods=4),
        'nullable_col': [1.0, None, 3.0, None]
    })

    # Write and read
    duck_frame_writer(temp_db, data=df)
    result = duck_frame_reader(temp_db)

    # Verify data integrity
    pd.testing.assert_frame_equal(result.data, df)


def test_empty_dataframe_round_trip(temp_db):
    """Test writing and reading empty DataFrame preserves structure."""
    # Create empty DataFrame with columns
    df = pd.DataFrame(columns=['id', 'name', 'value', 'score'])

    # Write and read
    duck_frame_writer(temp_db, empty=df)
    result = duck_frame_reader(temp_db)

    # Verify structure is preserved
    assert len(result.empty) == 0
    assert list(result.empty.columns) == ['id', 'name', 'value', 'score']


def test_unicode_and_special_characters(temp_db):
    """Test handling of Unicode and special characters in data."""
    df = pd.DataFrame({
        'name': ['Alice', 'Bob', '中文', 'José'],
        'emoji': ['😀', '🚀', '❤️', '🎉'],
        'special': ['a"b', "c'd", 'e\\f', 'g\nh']
    })

    # Write and read
    duck_frame_writer(temp_db, unicode_data=df)
    result = duck_frame_reader(temp_db)

    # Verify data integrity
    pd.testing.assert_frame_equal(result.unicode_data, df)


def test_large_dataframe(temp_db):
    """Test with a larger DataFrame to verify performance."""
    # Create larger DataFrame
    df = pd.DataFrame({
        'id': range(10000),
        'value': range(10000, 20000),
        'label': [f'item_{i}' for i in range(10000)]
    })

    # Write and read
    duck_frame_writer(temp_db, large_table=df)
    result = duck_frame_reader(temp_db)

    # Verify data integrity
    pd.testing.assert_frame_equal(result.large_table, df)
