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
        if os.path.isfile(self.TEST_DB_FILE):
            os.unlink(self.TEST_DB_FILE)

    def test_non_overwrite(self):
        # Store a single frame
        duck = Duck(self.TEST_DB_FILE, overwrite=False)
        duck.df_first = self.df_first

        # Make sure the frame exists and that table names look right
        duck = Duck(self.TEST_DB_FILE, overwrite=False)
        self.assertEqual(duck.df_first.name.iloc[0], 'first')
        self.assertEqual(tuple(duck.table_names), ('first',))

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

    def tearDown(self):
        if os.path.isfile(self.TEST_DB_FILE):
            os.unlink(self.TEST_DB_FILE)


# from ..utils import BlobMixin, BlobAttr

# # RUN TESTS WITH
# # pytest -sv ./test_utils_blob_attr.py

# # Define global defaults that will be checked against in tests
# m0 = 1
# m1 = {'one': 1}
# m2 = {'sub': {'one': 1}}


# # Class to test
# class Param(BlobMixin):
#     # Immutable
#     mut0 = BlobAttr(m0)

#     # Mutable
#     mut1 = BlobAttr(m1)

#     # Mutable of mutable
#     mut2 = BlobAttr(m2)


# def test_deep_defaults_dont_mutate():
#     # Create with defaults
#     p = Param()

#     # import pdb; pdb.set_trace()

#     # Make attribute assignments
#     p.mut0 = 2
#     p.mut1['one'] = 2
#     p.mut2['sub']['one'] = 2

#     # Test that attributes got updated
#     assert(p.mut0 == 2)
#     assert(p.mut1['one'] == 2)
#     assert(p.mut2['sub']['one'] == 2)

#     # Test that defaults did not
#     assert(m0 == 1)
#     assert(m1['one'] == 1)
#     assert(m2['sub']['one'] == 1)


# def test_nondeep_strict_assignments_mutate():
#     # Create with defaults
#     class Param2(Param):
#         # Mutable of mutable
#         mut2 = BlobAttr(m2, deep=False)

#     p = Param2()

#     # Load overrides from blob and check they were successful
#     blob = {'mut2': {'sub': {'one': 2}}}
#     p.from_blob(blob)
#     assert(p.mut2['sub']['one'] == 2)

#     # Change the attribute and make sure it got updated
#     p.mut2['sub']['one'] = 3
#     assert(p.mut2['sub']['one'] == 3)

#     # Check that the blob also got mutated because it was not deep
#     assert(blob['mut2']['sub']['one'] == 3)


# def test_bad_update_key():
#     import pytest
#     p = Param()

#     blob = {'bad': 2}
#     with pytest.raises(ValueError):
#         p.from_blob(blob)


# def test_bad_strict_update_key():
#     import pytest
#     p = Param()

#     blob = {'mut0': 2}
#     with pytest.raises(ValueError):
#         p.from_blob(blob, strict=True)


# def test_mutating_attribute_reference():
#     # Create with defaults
#     p = Param()

#     # Get reference to mutable attribute
#     mut1 = p.mut1
#     assert(mut1['one'] == 1)

#     # Mutate the reference
#     mut1['one'] = 2

#     # Make sure the attribute changed
#     assert(p.mut1['one'] == 2)


# def test_returned_blob_mutation():
#     # Create with defaults
#     p = Param()

#     # Dump a blob
#     blob = p.to_blob()

#     # Make sure the blob looks right
#     assert(blob['mut2']['sub']['one'] == 1)

#     # Mutate the blob
#     blob['mut2']['sub']['one'] = 3

#     # Make sure the blob mutated
#     assert(blob['mut2']['sub']['one'] == 3)

#     # Make sure the attribute didn't
#     assert(p.mut2['sub']['one'] == 1)
