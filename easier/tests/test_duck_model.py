from easier.duck_model import Duck

# from easier.minimodel import MiniModelSqlite, MiniModelPG, MiniModelDuck
from unittest import TestCase
import os

# # # RUN TESTS WITH
# # # pytest -sv ./test_duck_model.py
# # coverage erase && pytest -sv --cov=easier ./tests/test_minimodel.py  | grep minimodel
# # coverage erase && pytest -sv --cov=easier.minimodel ./tests/test_minimodel.py::TestMiniModelSqlite::test_saving

# # This will run tests and raise error on warnings
# # python -W error -munittest test_minimodel.TestMiniModel.test_saving


class TestDuckModel(TestCase):
    TEST_DB_FILE = '/tmp/test_minimodel.ddb'

    def setUp(self):
        import pandas as pd

        # Creat dataframe to store
        self.df_one = pd.DataFrame(
            [
                {'a': 1, 'b': pd.Timestamp('1/1/2022')},
                {'a': 2, 'b': pd.Timestamp('1/2/2022')}
            ]
        )

        # Create second dataframe to store
        self.df_two = self.df_one.copy()
        self.df_two.loc[0, 'a'] = 7
        self.df_two.loc[0, 'b'] = pd.Timestamp('11/18/2022')
        self._cleanup()

    def _cleanup(self):
        for file_name in [self.TEST_DB_FILE]:  # pragma: no cover
            if os.path.isfile(file_name):
                os.unlink(file_name)

    def tearDown(self):  # pragma: no cover
        self._cleanup()

    def test_saving(self):
        duck = Duck(overwrite=True, read_only=False)
        duck.tables.create('one', self.df_one)
        df1 = duck.tables.one.df
        self.assertListEqual(list(self.df_one.b), list(df1.b))

    def test_dropping(self):
        duck = Duck(overwrite=True, read_only=False)
        duck.tables.create('one', self.df_one)
        df1 = duck.tables.one.df
        self.assertListEqual(list(self.df_one.b), list(df1.b))

        self.assertTrue('one' in duck.table_names)
        duck.tables.one.drop()
        self.assertFalse('one' in duck.table_names)

        duck = Duck(overwrite=True, read_only=False)
        duck.tables.create('one', self.df_one)
        duck.tables.create('two', self.df_two)

        self.assertEqual(set(duck.table_names), {'one', 'two'})
        duck.tables.drop_all()
        self.assertEqual(len(duck.table_names), 0)

        duck = Duck(overwrite=True, read_only=False)
        duck.tables.create('one', self.df_one)
        with self.assertRaises(AttributeError):
            duck.tables.drop('table_that_doesnt exist')

        duck = Duck(overwrite=False, read_only=False)
        self.assertListEqual(duck.table_names, ['one'])
        duck.tables.drop('one')
        self.assertListEqual(duck.table_names, [])

    def test_inserting(self):
        import pandas as pd
        # Create a table
        duck = Duck(overwrite=True, read_only=False)
        duck.tables.create('one', self.df_one)

        # Make sure the table was created
        df1 = duck.tables.one.df
        self.assertListEqual(list(self.df_one.b), list(df1.b))
        del duck

        # Recreate the table with an overwrite
        duck2 = Duck(overwrite=True, read_only=False)
        duck2.tables.create('one', self.df_one)

        # Make sure the table was created
        df1 = duck2.tables.one.df
        self.assertListEqual(list(self.df_one.b), list(df1.b))
        del duck2

        # Reopen the file for additional insert
        duck3 = Duck(overwrite=False, read_only=False)
        duck3.tables.one.insert(self.df_two)

        # Make sure inserts happened
        df_expected = pd.concat([self.df_one, self.df_two])
        df = duck3.tables.one.df
        self.assertListEqual(list(df_expected.b), list(df.b))

    def test_read_only(self):

        # Make sure single table gets created
        duck = Duck(overwrite=True)
        duck.tables.create('one', self.df_one)
        self.assertListEqual(duck.table_names, ['one'])

        # Make sure overwriting doesn't keep original table
        duck = Duck(overwrite=True)
        duck.tables.create('two', self.df_two)
        self.assertListEqual(duck.table_names, ['two'])

        # Make sure non-overwrite retains original table
        duck = Duck(overwrite=False)
        duck.tables.create('one', self.df_one)
        self.assertListEqual(sorted(duck.table_names), ['one', 'two'])

        df = self.df_one

        # Create a writeable duck
        duck = Duck(overwrite=True, read_only=False)
        duck.tables.create('one', df)

        # Create a duck that is read_only
        duck = Duck(overwrite=False, read_only=True)

        # Make sure can't create, insert or drop
        with self.assertRaises(ValueError):
            duck.tables.create('one', df)

        with self.assertRaises(ValueError):
            duck.tables.one.insert(df)

        with self.assertRaises(ValueError):
            duck.tables.drop('one')

        # Check for nonsense of overwrite and read_only
        with self.assertRaises(ValueError):
            Duck(overwrite=True, read_only=True)

    def test_sql(self):
        import pandas as pd
        df = pd.DataFrame(
            [
                {'a': 1, 'b': pd.Timestamp('1/1/2022')},
                {'a': 2, 'b': pd.Timestamp('1/2/2022')}
            ]
        )
        duck = Duck()
        duck.tables.create('one', df)
        dfq = duck.query('select a, b from one')
        self.assertEqual(tuple(df.dtypes), tuple(dfq.dtypes))
        self.assertListEqual(list(df.b), list(dfq.b))
