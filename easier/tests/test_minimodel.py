from easier.duck import Duck
from unittest import TestCase
import os
import shutil

# # RUN TESTS WITH
# # pytest -sv ./test_duck.py
# coverage erase && pytest -sv --cov=easier ./tests/test_duck.py  | grep duck
# coverage erase && pytest -sv --cov=easier.duck ./tests/test_duck.py::TestDuck::test_export_import


class TestDuck(TestCase):
    TEST_DB_FILE = '/tmp/test_duck.ddb'
    RECOVER_DB_FILE = '/tmp/test_duck_recovery.ddb'
    BACKUP_DIR = '/tmp/duck_backup'

    def setUp(self):
        import pandas as pd
        self.df_first = pd.DataFrame([{'name': 'first'}])
        self.df_second = pd.DataFrame([{'name': 'second'}])
        self._cleanup()

    def _cleanup(self):
        for file_name in [self.TEST_DB_FILE, self.RECOVER_DB_FILE]:  # pragma: no cover
            if os.path.isfile(file_name):
                os.unlink(file_name)

        if os.path.isdir(self.BACKUP_DIR):
            shutil.rmtree(self.BACKUP_DIR)

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

    def test_read_only(self):
        # Create a duck file
        duck = Duck(self.TEST_DB_FILE, overwrite=False)
        duck.df_first = self.df_first

        # Load the duck file in read-only mode
        duck = Duck(self.TEST_DB_FILE, read_only=True)

        # I should be able to set non tracked attributes at will
        duck.silly = 'quack'
        duck.non_tracked_frame = self.df_first

        # Trying to assign a dataframe should barf
        with self.assertRaises(ValueError):
            duck.df_first = self.df_first

    def test_export_import(self):
        # Make sure backup directory doesn't exist
        self.assertFalse(os.path.isdir(self.BACKUP_DIR))
        duck_original = Duck(self.TEST_DB_FILE)
        duck_original.df_first = self.df_first
        duck_original.df_second = self.df_second

        # Export the db
        duck_original.export_db(self.BACKUP_DIR)

        # Try exporting again to ensure it raises an error
        with self.assertRaises(ValueError):
            duck_original.export_db(self.BACKUP_DIR)

        # Make sure the backup files exist
        self.assertTrue(os.path.isfile(os.path.join(self.BACKUP_DIR, 'schema.sql')))
        # import pdb; pdb.set_trace()

        duck_recovery = Duck(self.RECOVER_DB_FILE)

        # Try importing from bogus directory
        with self.assertRaises(ValueError):
            duck_recovery.import_db('/tmp/this/is/a/bogus/directory')

        # Import recovered db
        duck_recovery.import_db(self.BACKUP_DIR)

        # Make sure it got all backed up tables
        self.assertEqual(set(duck_recovery.table_names), set(duck_original.table_names))

        # Make sure the first table looks lright
        self.assertEqual(duck_recovery.df_first.name.iloc[0], 'first')

    def tearDown(self):  # pragma: no cover
        self._cleanup()
