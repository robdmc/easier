from easier.duck import Duck
from unittest import TestCase
import os

# # RUN TESTS WITH
# # pytest -sv ./test_duck.py


class TestDuck(TestCase):
    TEST_DB_FILE = '/tmp/test_duck.ddb'

    def setUp(self):
        import pandas as pd
        self.df_first = pd.DataFrame([{'name': 'first'}])
        self.df_second = pd.DataFrame([{'name': 'second'}])
        if os.path.isfile(self.TEST_DB_FILE):  # pragma: no cover
            os.unlink(self.TEST_DB_FILE)

    def test_non_overwrite(self):
        # Store a single frame
        duck = Duck(self.TEST_DB_FILE, overwrite=False)
        duck.df_first = self.df_first

        # Make sure the frame exists and that table names look right
        duck = Duck(self.TEST_DB_FILE, overwrite=False)
        self.assertEqual(duck.df_first.name.iloc[0], 'first')
        self.assertEqual(tuple(duck.table_names), ('first',))

        # Make sure the database file exists
        self.assertTrue(os.path.isfile(self.TEST_DB_FILE))

        # Add a second table
        duck = Duck(self.TEST_DB_FILE, overwrite=False)
        duck.df_second = self.df_second

        # Make sure both tables exist
        duck = Duck(self.TEST_DB_FILE, overwrite=False)
        self.assertEqual(duck.df_first.name.iloc[0], 'first')
        self.assertEqual(duck.df_second.name.iloc[0], 'second')
        self.assertEqual(set(duck.table_names), {'first', 'second'})

        # Make sure updates work
        duck = Duck(self.TEST_DB_FILE, overwrite=False)
        df = self.df_second.copy()
        df.name.iloc[0] = 'third'
        duck.df_second = df
        duck = Duck(self.TEST_DB_FILE, overwrite=False)
        self.assertEqual(duck.df_first.name.iloc[0], 'first')
        self.assertEqual(duck.df_second.name.iloc[0], 'third')
        self.assertEqual(set(duck.table_names), {'first', 'second'})

    def test_overwrite(self):
        # Store a single frame
        duck = Duck(self.TEST_DB_FILE, overwrite=False)
        duck.df_first = self.df_first

        # Make sure the frame exists and that table names look right
        duck = Duck(self.TEST_DB_FILE, overwrite=False)
        self.assertEqual(duck.df_first.name.iloc[0], 'first')
        self.assertEqual(tuple(duck.table_names), ('first',))

        # Add a second table
        duck = Duck(self.TEST_DB_FILE, overwrite=True)
        duck.df_second = self.df_second

        # Make sure only second table exixts
        duck = Duck(self.TEST_DB_FILE, overwrite=False)
        self.assertEqual(duck.df_second.name.iloc[0], 'second')
        self.assertEqual(set(duck.table_names), {'second'})

    def tearDown(self):  # pragma: no cover
        if os.path.isfile(self.TEST_DB_FILE):
            os.unlink(self.TEST_DB_FILE)
