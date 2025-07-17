from easier.dataframe_tools import (
    pandas_time_to_utc_seconds,
    pandas_utc_seconds_to_time,
    slugify,
)
from unittest import TestCase
import datetime
import pandas as pd


class TestSlugify(TestCase):

    def test_strings(self):
        text = "hello:  world "
        camel_text = "helloWorld"
        out_text = slugify(text)
        out_text2 = slugify(text, sep=".")
        self.assertEqual(out_text, "hello_world")
        self.assertEqual(out_text2, "hello.world")
        self.assertEqual(slugify(camel_text, kill_camel=True), "hello_world")

    def test_iterable(self):
        values = [
            " these ",
            "are_things",
            "ThatNeedToBe",
            "*@#$%^fixed;#$ .   rightNow",
        ]
        fixed = slugify(values, kill_camel=True)
        self.assertListEqual(
            fixed, ["these", "are_things", "that_need_to_be", "fixed_right_now"]
        )


class TestUnixTimeConversion(TestCase):

    def setUp(self):
        dates = pd.date_range("1/1/2020", "1/10/2020")
        df = pd.DataFrame({"time1": dates.values})
        df["time2"] = df.time1
        self.df = df

    def test_conversion(self):
        dfin = self.df.copy()
        self.assertTrue(all(["datetime64" in str(val) for val in dfin.dtypes.values]))
        df = pandas_time_to_utc_seconds(dfin)
        self.assertTrue(all(["int64" in str(val) for val in df.dtypes.values]))
        df_back = pandas_utc_seconds_to_time(df)
        self.assertEqual((dfin - df_back).max().max().days, 0)
        df = pandas_time_to_utc_seconds(dfin, columns=["time1"])
        self.assertTrue(any(["int64" in str(val) for val in df.dtypes.values]))
        self.assertTrue(any(["datetime64" in str(val) for val in df.dtypes.values]))
        df["time1"] = pandas_utc_seconds_to_time(df.time1)
        self.assertEqual((dfin - df).max().max().days, 0)
        df = pandas_time_to_utc_seconds(dfin, ["time2"])
        epoch = datetime.datetime(1970, 1, 1)
        expected_seconds = (df.time1.iloc[0] - epoch).total_seconds()
        computed_seconds = df.time2.iloc[0]
        self.assertEqual(expected_seconds, computed_seconds)
        with self.assertRaises(ValueError):
            pandas_utc_seconds_to_time(df.time1, columns="time")
        with self.assertRaises(ValueError):
            pandas_utc_seconds_to_time([1, 2, 3])


def test_hex_dataframe_round_trip():
    import pandas as pd
    from easier.dataframe_tools import hex_from_dataframe, hex_to_dataframe

    df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"], "C": [1.1, 2.2, 3.3]})
    hex_str = hex_from_dataframe(df)
    df2 = hex_to_dataframe(hex_str)
    pd.testing.assert_frame_equal(df, df2)


def test_duckdb_hex_round_trip():
    import duckdb
    import pandas as pd
    from easier.dataframe_tools import hex_from_duckdb, hex_to_duckdb
    # Create an in-memory DuckDB and a test table
    conn = duckdb.connect(":memory:")
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    conn.execute("CREATE TABLE test AS SELECT * FROM df")
    # Export to hex
    hex_dump = hex_from_duckdb(conn)
    # Import back from hex
    new_conn = hex_to_duckdb(hex_dump)
    # Fetch the data from the new connection
    result_df = new_conn.execute("SELECT * FROM test").fetchdf()
    # Check that the data matches
    pd.testing.assert_frame_equal(result_df, df)
