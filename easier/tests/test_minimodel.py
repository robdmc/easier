from easier.minimodel import MiniModelSqlite, MiniModelPG
from unittest import TestCase
import os

# # RUN TESTS WITH
# # pytest -sv ./test_duck.py
# coverage erase && pytest -sv --cov=easier ./tests/test_minimodel.py  | grep minimodel
# coverage erase && pytest -sv --cov=easier.minimodel ./tests/test_minimodel.py::TestMiniModelSqlite::test_saving

# This will run tests and raise error on warnings
# python -W error -munittest test_minimodel.TestMiniModel.test_saving


class TestMiniModelSqlite(TestCase):
    TEST_DB_FILE = "/tmp/test_minimodel.ddb"
    MODEL_CLASS = MiniModelSqlite

    def setUp(self):
        import pandas as pd

        # Creat dataframe to store
        self.df_one = pd.DataFrame(
            [
                {"a": 1, "b": pd.Timestamp("1/1/2022")},
                {"a": 2, "b": pd.Timestamp("1/2/2022")},
            ]
        )

        # Create second dataframe to store
        self.df_two = self.df_one.copy()
        self.df_two.loc[0, "a"] = 7
        self.df_two.loc[0, "b"] = pd.Timestamp("11/18/2022")
        self._cleanup()

    def _cleanup(self):
        for file_name in [self.TEST_DB_FILE]:  # pragma: no cover
            if os.path.isfile(file_name):
                os.unlink(file_name)

    def tearDown(self):  # pragma: no cover
        self._cleanup()

    def get_model(self, overwrite, read_only):
        return self.MODEL_CLASS(
            self.TEST_DB_FILE, overwrite=overwrite, read_only=read_only
        )

    def test_saving(self):
        mm = self.get_model(overwrite=True, read_only=False)
        mm.create("one", self.df_one)
        # import pdb; pdb.pdb.set_trace()
        df1 = mm.tables.one.df
        self.assertListEqual(list(self.df_one.b), list(df1.b))

    def test_dropping(self):
        mm = self.get_model(overwrite=True, read_only=False)
        mm.create("one", self.df_one)
        # import pdb; pdb.pdb.set_trace()
        df1 = mm.tables.one.df
        self.assertListEqual(list(self.df_one.b), list(df1.b))

        self.assertTrue("one" in mm.table_names)
        mm.drop("one")
        self.assertFalse("one" in mm.table_names)

        mm = self.get_model(overwrite=True, read_only=False)
        mm.create("one", self.df_one)
        mm.create("two", self.df_two)

        self.assertEqual(set(mm.table_names), {"one", "two"})
        mm.drop_all_tables()
        self.assertEqual(len(mm.table_names), 0)

        mm = self.get_model(overwrite=True, read_only=False)
        mm.create("one", self.df_one)
        with self.assertRaises(ValueError):
            mm.drop("table_that_doesnt exist")

        mm = self.get_model(overwrite=False, read_only=True)
        with self.assertRaises(ValueError):
            mm.drop("one")

    def test_inserting(self):
        import pandas as pd

        # Create a table
        mm = self.get_model(overwrite=True, read_only=False)
        mm.create("one", self.df_one)

        # Make sure the table was created
        df1 = mm.tables.one.df
        self.assertListEqual(list(self.df_one.b), list(df1.b))
        del mm

        # Recreate the table with an overwrite
        mm2 = self.get_model(overwrite=True, read_only=False)
        mm2.insert("one", self.df_one)

        # Make sure the table was created
        df1 = mm2.tables.one.df
        self.assertListEqual(list(self.df_one.b), list(df1.b))
        del mm2

        # Reopen the file for additional insert
        mm3 = self.get_model(overwrite=False, read_only=False)
        mm3.insert("one", self.df_two)

        # Make sure inserts happened
        df_expected = pd.concat([self.df_one, self.df_two])
        df = mm3.tables.one.df
        self.assertListEqual(list(df_expected.b), list(df.b))

    def test_read_only(self):
        import pandas as pd

        df = pd.DataFrame(
            [
                {"a": 1, "b": pd.Timestamp("1/1/2022")},
                {"a": 2, "b": pd.Timestamp("1/2/2022")},
            ]
        )

        mm = self.get_model(overwrite=True, read_only=False)
        mm.create("one", df)

        with self.assertRaises(ValueError):
            self.get_model(overwrite=True, read_only=True)

        mm = self.get_model(overwrite=False, read_only=True)

        with self.assertRaises(ValueError):
            mm.create("one", df)

        with self.assertRaises(ValueError):
            mm.insert("one", df)

        with self.assertRaises(ValueError):
            mm.upsert("one", ["a"], df)

    def test_upserting(self):
        import pandas as pd

        df_base = pd.DataFrame(
            [
                {"a": 1, "b": pd.Timestamp("1/1/2022")},
                {"a": 2, "b": pd.Timestamp("1/2/2022")},
            ]
        )

        df_upsert = pd.DataFrame(
            [
                {"a": 2, "b": pd.Timestamp("1/2/2050")},
                {"a": 3, "b": pd.Timestamp("1/3/2050")},
            ]
        )

        df_expected = pd.DataFrame(
            [
                {"a": 1, "b": pd.Timestamp("1/1/2022")},
                {"a": 2, "b": pd.Timestamp("1/2/2050")},
                {"a": 3, "b": pd.Timestamp("1/3/2050")},
            ]
        )

        mm = self.get_model(overwrite=True, read_only=False)
        mm.create("one", df_base)
        mm.upsert("one", ["a"], df_upsert)

        df = mm.tables.one.df
        self.assertListEqual(list(df_expected.b), list(df.b))

    def test_record_upsert(self):
        import pandas as pd

        df_base = pd.DataFrame(
            [
                {"a": 1, "b": pd.Timestamp("1/1/2022")},
                {"a": 2, "b": pd.Timestamp("1/2/2022")},
            ]
        )
        df_expected = pd.DataFrame(
            [
                {"a": 1, "b": pd.Timestamp("1/1/2022")},
                {"a": 2, "b": pd.Timestamp("1/2/2050")},
                {"a": 3, "b": pd.Timestamp("11/18/2050")},
            ]
        )

        mm = self.get_model(overwrite=True, read_only=False)
        # Create a table twice to ensure creating actually drops
        mm.create("one", df_base)
        mm.create("one", df_base)
        mm.upsert("one", ["a"], {"a": 2, "b": pd.Timestamp("1/2/2050")})
        mm.upsert("one", ["a"], pd.Series({"a": 3, "b": pd.Timestamp("11/18/2050")}))

        with self.assertRaises(ValueError):
            mm.upsert("one", ["a"], [1, 2, 3])

        df = mm.tables.one.df
        self.assertListEqual(list(df_expected.b), list(df.b))

    def test_sql(self):
        import pandas as pd
        import numpy as np

        df_base = pd.DataFrame(
            [
                {"a": 1, "b": pd.Timestamp("1/1/2022")},
                {"a": 2, "b": pd.Timestamp("1/2/2022")},
            ]
        )
        mm = self.get_model(overwrite=True, read_only=False)
        mm.create("one", df_base)
        df = mm.query("select a, b from one order by b")
        df.loc[:, "b"] = df.b.astype(np.datetime64)
        self.assertListEqual(list(df_base.b), list(df.b))


class TestMiniModelPG(TestMiniModelSqlite):
    MODEL_CLASS = MiniModelPG

    def _cleanup(self):
        mm = self.get_model(overwrite=False, read_only=False)
        mm.drop_all_tables()

    def get_model(self, overwrite, read_only):
        return self.MODEL_CLASS(overwrite=overwrite, read_only=read_only)
