from easier.minimodel import MiniModelSqlite, MiniModelPG
from unittest import TestCase
import numpy as np
import os
import pandas as pd

from easier.minimodel import MiniModelSqlite, MiniModelPG
from unittest import TestCase
import os

class TestMiniModelSqlite(TestCase):
    TEST_DB_FILE = '/tmp/test_minimodel.ddb'
    MODEL_CLASS = MiniModelSqlite

    def setUp(self):
        self.df_one = pd.DataFrame([{'a': 1, 'b': pd.Timestamp('1/1/2022')}, {'a': 2, 'b': pd.Timestamp('1/2/2022')}])
        self.df_two = self.df_one.copy()
        self.df_two.loc[0, 'a'] = 7
        self.df_two.loc[0, 'b'] = pd.Timestamp('11/18/2022')
        self._cleanup()

    def _cleanup(self):
        for file_name in [self.TEST_DB_FILE]:
            if os.path.isfile(file_name):
                os.unlink(file_name)

    def tearDown(self):
        self._cleanup()

    def get_model(self, overwrite, read_only):
        return self.MODEL_CLASS(self.TEST_DB_FILE, overwrite=overwrite, read_only=read_only)

    def test_saving(self):
        mm = self.get_model(overwrite=True, read_only=False)
        mm.create('one', self.df_one)
        df1 = mm.tables.one.df
        self.assertListEqual(list(self.df_one.b), list(df1.b))

    def test_dropping(self):
        mm = self.get_model(overwrite=True, read_only=False)
        mm.create('one', self.df_one)
        df1 = mm.tables.one.df
        self.assertListEqual(list(self.df_one.b), list(df1.b))
        self.assertTrue('one' in mm.table_names)
        mm.drop('one')
        self.assertFalse('one' in mm.table_names)
        mm = self.get_model(overwrite=True, read_only=False)
        mm.create('one', self.df_one)
        mm.create('two', self.df_two)
        self.assertEqual(set(mm.table_names), {'one', 'two'})
        mm.drop_all_tables()
        self.assertEqual(len(mm.table_names), 0)
        mm = self.get_model(overwrite=True, read_only=False)
        mm.create('one', self.df_one)
        with self.assertRaises(ValueError):
            mm.drop('table_that_doesnt exist')
        mm = self.get_model(overwrite=False, read_only=True)
        with self.assertRaises(ValueError):
            mm.drop('one')

    def test_inserting(self):
        mm = self.get_model(overwrite=True, read_only=False)
        mm.create('one', self.df_one)
        df1 = mm.tables.one.df
        self.assertListEqual(list(self.df_one.b), list(df1.b))
        del mm
        mm2 = self.get_model(overwrite=True, read_only=False)
        mm2.insert('one', self.df_one)
        df1 = mm2.tables.one.df
        self.assertListEqual(list(self.df_one.b), list(df1.b))
        del mm2
        mm3 = self.get_model(overwrite=False, read_only=False)
        mm3.insert('one', self.df_two)
        df_expected = pd.concat([self.df_one, self.df_two])
        df = mm3.tables.one.df
        self.assertListEqual(list(df_expected.b), list(df.b))

    def test_read_only(self):
        df = pd.DataFrame([{'a': 1, 'b': pd.Timestamp('1/1/2022')}, {'a': 2, 'b': pd.Timestamp('1/2/2022')}])
        mm = self.get_model(overwrite=True, read_only=False)
        mm.create('one', df)
        with self.assertRaises(ValueError):
            self.get_model(overwrite=True, read_only=True)
        mm = self.get_model(overwrite=False, read_only=True)
        with self.assertRaises(ValueError):
            mm.create('one', df)
        with self.assertRaises(ValueError):
            mm.insert('one', df)
        with self.assertRaises(ValueError):
            mm.upsert('one', ['a'], df)

    def test_upserting(self):
        df_base = pd.DataFrame([{'a': 1, 'b': pd.Timestamp('1/1/2022')}, {'a': 2, 'b': pd.Timestamp('1/2/2022')}])
        df_upsert = pd.DataFrame([{'a': 2, 'b': pd.Timestamp('1/2/2050')}, {'a': 3, 'b': pd.Timestamp('1/3/2050')}])
        df_expected = pd.DataFrame([{'a': 1, 'b': pd.Timestamp('1/1/2022')}, {'a': 2, 'b': pd.Timestamp('1/2/2050')}, {'a': 3, 'b': pd.Timestamp('1/3/2050')}])
        mm = self.get_model(overwrite=True, read_only=False)
        mm.create('one', df_base)
        mm.upsert('one', ['a'], df_upsert)
        df = mm.tables.one.df
        self.assertListEqual(list(df_expected.b), list(df.b))

    def test_record_upsert(self):
        df_base = pd.DataFrame([{'a': 1, 'b': pd.Timestamp('1/1/2022')}, {'a': 2, 'b': pd.Timestamp('1/2/2022')}])
        df_expected = pd.DataFrame([{'a': 1, 'b': pd.Timestamp('1/1/2022')}, {'a': 2, 'b': pd.Timestamp('1/2/2050')}, {'a': 3, 'b': pd.Timestamp('11/18/2050')}])
        mm = self.get_model(overwrite=True, read_only=False)
        mm.create('one', df_base)
        mm.create('one', df_base)
        mm.upsert('one', ['a'], {'a': 2, 'b': pd.Timestamp('1/2/2050')})
        mm.upsert('one', ['a'], pd.Series({'a': 3, 'b': pd.Timestamp('11/18/2050')}))
        with self.assertRaises(ValueError):
            mm.upsert('one', ['a'], [1, 2, 3])
        df = mm.tables.one.df
        self.assertListEqual(list(df_expected.b), list(df.b))

    def test_sql(self):
        df_base = pd.DataFrame([{'a': 1, 'b': pd.Timestamp('1/1/2022')}, {'a': 2, 'b': pd.Timestamp('1/2/2022')}])
        mm = self.get_model(overwrite=True, read_only=False)
        mm.create('one', df_base)
        df = mm.query('select a, b from one order by b')
        df.loc[:, 'b'] = df.b.astype(np.datetime64)
        self.assertListEqual(list(df_base.b), list(df.b))

class TestMiniModelPG(TestMiniModelSqlite):
    MODEL_CLASS = MiniModelPG

    def _cleanup(self):
        mm = self.get_model(overwrite=False, read_only=False)
        mm.drop_all_tables()

    def get_model(self, overwrite, read_only):
        return self.MODEL_CLASS(overwrite=overwrite, read_only=read_only)