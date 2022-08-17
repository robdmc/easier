
import datetime
from unittest import TestCase

from easier.dataframe_tools import (
    pandas_time_to_utc_seconds,
    pandas_utc_seconds_to_time,
    slugify,
)

# RUN TESTS WITH
# pytest -sv ./test_duck.py
# coverage erase && pytest -sv --cov=easier ./tests/test_dataframe_tools.py  | grep duck

# coverage erase && pytest -sv --cov=easier.dataframe_tools \
# ./tests/test_dataframe_tools.py::TestUnixTimeConversion::test_timestamps_to_ints

# python -m unittest easier.tests.test_dataframe_tools

# IF PYTEST GIVES YOU WEIRD PROBLEMS, JUST GO OLD SCHOOL
# coverage erase && coverage run  -m unittest easier.tests.test_dataframe_tools && coverage report dataframe_tools.py


class TestSlugify(TestCase):
    def test_strings(self):
        text = 'hello:  world '
        camel_text = 'helloWorld'
        out_text = slugify(text)
        out_text2 = slugify(text, sep='.')
        self.assertEqual(out_text, 'hello_world')
        self.assertEqual(out_text2, 'hello.world')
        self.assertEqual(slugify(camel_text, kill_camel=True), 'hello_world')

    def test_iterable(self):
        values = [' these ', 'are_things', 'ThatNeedToBe', '*@#$%^fixed;#$ .   rightNow']
        fixed = slugify(values, kill_camel=True)
        self.assertListEqual(
            fixed,
            ['these', 'are_things', 'that_need_to_be', 'fixed_right_now']
        )


class TestUnixTimeConversion(TestCase):
    def setUp(self):
        import pandas as pd
        dates = pd.date_range('1/1/2020', '1/10/2020')
        df = pd.DataFrame({'time1': dates.values})
        df['time2'] = df.time1
        self.df = df

    def test_conversion(self):
        dfin = self.df.copy()

        # Make sure all initial columms are datetimes
        self.assertTrue(all(['datetime64' in str(val) for val in dfin.dtypes.values]))

        # Transform all columns in frame
        df = pandas_time_to_utc_seconds(dfin)

        # Make sure all columns got transformed to ints
        self.assertTrue(all(['int64' in str(val) for val in df.dtypes.values]))

        # Make sure the inversion works
        df_back = pandas_utc_seconds_to_time(df)
        self.assertEqual((dfin - df_back).max().max().days, 0)

        # Transform only one colum in the frame
        df = pandas_time_to_utc_seconds(dfin, columns=['time1'])

        # Make sure the frame has both types proving the conversion
        self.assertTrue(any(['int64' in str(val) for val in df.dtypes.values]))
        self.assertTrue(any(['datetime64' in str(val) for val in df.dtypes.values]))

        # Make sure the inversion works
        df.loc[:, 'time1'] = pandas_utc_seconds_to_time(df.time1)
        self.assertEqual((dfin - df).max().max().days, 0)

        # Explicitely check that one conversion is right
        df = pandas_time_to_utc_seconds(dfin, ['time2'])
        epoch = datetime.datetime(1970, 1, 1)
        expected_seconds = (df.time1.iloc[0] - epoch).total_seconds()
        computed_seconds = df.time2.iloc[0]
        self.assertEqual(expected_seconds, computed_seconds)

        # Run through error checks
        with self.assertRaises(ValueError):
            pandas_utc_seconds_to_time(df.time1, columns='time')

        with self.assertRaises(ValueError):
            pandas_utc_seconds_to_time([1, 2, 3])
