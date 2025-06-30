from easier.duck import Duck
from unittest import TestCase
import os
import pandas as pd
import shutil

class TestDuck(TestCase):
    TEST_DB_FILE = '/tmp/test_duck.ddb'
    RECOVER_DB_FILE = '/tmp/test_duck_recovery.ddb'
    BACKUP_DIR = '/tmp/duck_backup'

    def setUp(self):
        self.df_first = pd.DataFrame([{'name': 'first'}])
        self.df_second = pd.DataFrame([{'name': 'second'}])
        self._cleanup()

    def _cleanup(self):
        for file_name in [self.TEST_DB_FILE, self.RECOVER_DB_FILE]:
            if os.path.isfile(file_name):
                os.unlink(file_name)
        if os.path.isdir(self.BACKUP_DIR):
            shutil.rmtree(self.BACKUP_DIR)

    def test_non_overwrite(self):
        duck = Duck(self.TEST_DB_FILE, overwrite=False)
        duck.df_first = self.df_first
        duck = Duck(self.TEST_DB_FILE, overwrite=False)
        self.assertEqual(duck.df_first.name.iloc[0], 'first')
        self.assertEqual(tuple(duck.table_names), ('first',))
        self.assertTrue(os.path.isfile(self.TEST_DB_FILE))
        duck = Duck(self.TEST_DB_FILE, overwrite=False)
        duck.df_second = self.df_second
        duck = Duck(self.TEST_DB_FILE, overwrite=False)
        self.assertEqual(duck.df_first.name.iloc[0], 'first')
        self.assertEqual(duck.df_second.name.iloc[0], 'second')
        self.assertEqual(set(duck.table_names), {'first', 'second'})
        duck = Duck(self.TEST_DB_FILE, overwrite=False)
        df = self.df_second.copy()
        df.name.iloc[0] = 'third'
        duck.df_second = df
        duck = Duck(self.TEST_DB_FILE, overwrite=False)
        self.assertEqual(duck.df_first.name.iloc[0], 'first')
        self.assertEqual(duck.df_second.name.iloc[0], 'third')
        self.assertEqual(set(duck.table_names), {'first', 'second'})

    def test_overwrite(self):
        duck = Duck(self.TEST_DB_FILE, overwrite=False)
        duck.df_first = self.df_first
        duck = Duck(self.TEST_DB_FILE, overwrite=False)
        self.assertEqual(duck.df_first.name.iloc[0], 'first')
        self.assertEqual(tuple(duck.table_names), ('first',))
        duck = Duck(self.TEST_DB_FILE, overwrite=True)
        duck.df_second = self.df_second
        duck = Duck(self.TEST_DB_FILE, overwrite=False)
        self.assertEqual(duck.df_second.name.iloc[0], 'second')
        self.assertEqual(set(duck.table_names), {'second'})

    def test_read_only(self):
        duck = Duck(self.TEST_DB_FILE, overwrite=False)
        duck.df_first = self.df_first
        duck = Duck(self.TEST_DB_FILE, read_only=True)
        duck.silly = 'quack'
        duck.non_tracked_frame = self.df_first
        with self.assertRaises(ValueError):
            duck.df_first = self.df_first

    def test_export_import(self):
        self.assertFalse(os.path.isdir(self.BACKUP_DIR))
        duck_original = Duck(self.TEST_DB_FILE)
        duck_original.df_first = self.df_first
        duck_original.df_second = self.df_second
        duck_original.export_db(self.BACKUP_DIR)
        with self.assertRaises(ValueError):
            duck_original.export_db(self.BACKUP_DIR)
        self.assertTrue(os.path.isfile(os.path.join(self.BACKUP_DIR, 'schema.sql')))
        duck_recovery = Duck(self.RECOVER_DB_FILE)
        with self.assertRaises(ValueError):
            duck_recovery.import_db('/tmp/this/is/a/bogus/directory')
        duck_recovery.import_db(self.BACKUP_DIR)
        self.assertEqual(set(duck_recovery.table_names), set(duck_original.table_names))
        self.assertEqual(duck_recovery.df_first.name.iloc[0], 'first')

    def tearDown(self):
        self._cleanup()