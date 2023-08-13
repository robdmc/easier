from ..utils import pickle_cached_container, pickle_cache_state
import pytest
import datetime
import os


TEST_PICKLE_FILE = "/tmp/my_test_cache.pickle"


def cache_file_exists():
    return os.path.isfile(TEST_PICKLE_FILE)


def kill_cache_file():
    if cache_file_exists():
        os.unlink(TEST_PICKLE_FILE)


class TestingClass:
    num_compute_calls = 0

    @pickle_cached_container()
    def my_tuple(self):
        return (1, 2, 3)

    @pickle_cached_container(pickle_file_name=TEST_PICKLE_FILE)
    def my_list(self):
        return self.compute_list()

    def compute_list(self):
        self.num_compute_calls += 1
        if not hasattr(self, "_return"):
            self.set_return([1, 2, 3])
        return self._return

    def set_return(self, val):
        self._return = val

    def get_num_calls(self):
        if hasattr(self, "num_compute_calls"):
            return self.num_compute_calls
        else:
            return 0


class TestingClassForFileCreation:
    @pickle_cached_container()
    def my_list(self):
        return [1, 2, 3]


class TestingClassForFileCreationCustom:
    @pickle_cached_container(
        pickle_file_name="/tmp/silly_pickle.pickle", return_copy=False
    )
    def my_list(self):
        return [1, 2, 3]


def test_out_of_scope():
    try:
        kill_cache_file()
        TestingClass.pkc = pickle_cache_state(mode="active")

        def scope(expected_calls):
            obj = TestingClass()
            assert obj.get_num_calls() == 0
            result = obj.my_list
            assert obj.get_num_calls() == expected_calls
            return result

        # Grabbing the results from an object once
        # will make it go out of scope.  I want to make sure
        # that the cachefile is not deleted when it goes out of scope
        scope(1)

        # Make sure the cache_file exists now that I'm out of scope
        assert cache_file_exists()

        # Calling again should not result in any additional compuations.
        scope(0)

    # Clean up
    finally:
        kill_cache_file()
        if hasattr(TestingClass, "pkc"):
            delattr(TestingClass, "pkc")


def check_pickle_cache_refresh_or_reset_mode(mode):
    try:
        kill_cache_file()

        # Set the testing class to active mode
        TestingClass.pkc = pickle_cache_state(mode=mode)

        obj = TestingClass()

        # Make sure no calls have been made an no cache file exists
        assert obj.get_num_calls() == 0
        assert not cache_file_exists()

        # This call should run a computation and file should exist
        result1 = obj.my_list
        assert obj.get_num_calls() == 1
        assert cache_file_exists()

        # Make sure the results looks right
        assert tuple(result1) == (1, 2, 3)

        # Now change the result that computation would return
        obj.set_return([4, 5, 6])

        # The result should now be different because another computation
        # should have been triggered
        result1 = obj.my_list
        assert obj.get_num_calls() == 2
        assert tuple(result1) == (4, 5, 6)
        assert cache_file_exists()

        # Creating a new object using active mode
        TestingClass.pkc = pickle_cache_state(mode="active")
        obj = TestingClass()

        # The new result should now be the new version from the pickle cache
        # and no computations should have been done
        result1 = obj.my_list
        assert obj.get_num_calls() == 0
        assert tuple(result1) == (4, 5, 6)

        # This is weird, but tuples return the same object when they are copied.
        # I guess it's because it doesn't make sense to copy immutable obj.
        assert id(obj.my_tuple) == id(obj.my_tuple)

    # Clean up
    finally:
        del obj.my_list
        del obj.my_tuple
        if hasattr(TestingClass, "pkc"):
            delattr(TestingClass, "pkc")


def test_bad_pickle_cache_mode():
    with pytest.raises(ValueError):
        pickle_cache_state(mode="this_is_bad")


def test_pickle_cache_reset_mode():
    return check_pickle_cache_refresh_or_reset_mode("reset")


def test_pickle_cache_refresh_mode():
    return check_pickle_cache_refresh_or_reset_mode("refresh")


def test_pickle_cache_memory_mode():
    try:
        kill_cache_file()

        # # Run an object to create a cache file with weird values
        # # This will ensure that what follows ignores this file
        # TestingClass.pkc = pickle_cache_state(mode='active')
        # active_obj = TestingClass()
        # active_obj.set_return([7, 8, 9])
        # active_obj.my_list
        # assert cache_file_exists()

        # Set the testing class to active mode
        TestingClass.pkc = pickle_cache_state(mode="memory")

        obj = TestingClass()

        # Make sure no calls have been made an no cache file exists
        assert obj.get_num_calls() == 0
        assert not cache_file_exists()

        # This call should run a computation
        result1 = obj.my_list
        assert obj.get_num_calls() == 1

        # But no file should have been created
        assert not cache_file_exists()

        # Calling again should not trigger a computation or create a file
        result2 = obj.my_list
        assert obj.get_num_calls() == 1
        assert not cache_file_exists()

        # Make sure results look right
        assert tuple(result1) == (1, 2, 3)
        assert tuple(result2) == (1, 2, 3)

        # Now create another object that will create a cached file
        # With weird values
        TestingClass.pkc = pickle_cache_state(mode="active")
        active_obj = TestingClass()
        active_obj.set_return([7, 8, 9])
        active_obj.my_list
        assert cache_file_exists()
        TestingClass.pkc = pickle_cache_state(mode="memory")

        # Grab results from my original objects to make sure they don't
        # have the wacky values contained in the cache file
        result3 = obj.my_list
        assert obj.get_num_calls() == 1
        assert tuple(result3) == (1, 2, 3)

    # Clean up
    finally:
        del obj.my_list
        if hasattr(TestingClass, "pkc"):
            delattr(TestingClass, "pkc")


# def test_pickle_cache_no_copy():
#     try:
#         kill_cache_file()

#         # Make sure the nocopy flag is actually returning th same
#         # object every time
#         obj = TestingClassNoCopy()
#         assert id(obj.my_list) == id(obj.my_list)

#     # Clean up
#     finally:
#         del obj.my_list
#         if hasattr(TestingClass, 'pkc'):
#             delattr(TestingClass, 'pkc')


def test_pickle_cache_active_mode():
    try:
        kill_cache_file()

        # Set the testing class to active mode
        TestingClass.pkc = pickle_cache_state(mode="active")

        obj = TestingClass()

        # Make sure no calls have been made an no cache file exists
        assert obj.get_num_calls() == 0
        assert not cache_file_exists()

        # This call should run a computation
        first_result = obj.my_list

        # Make sure the results look right
        assert tuple(first_result) == (1, 2, 3)

        # Make sure it was actually called and created a file
        assert obj.get_num_calls() == 1
        assert cache_file_exists()

        # This call should hit the memory cache
        second_result = obj.my_list

        # Make sure the results look right
        assert tuple(second_result) == (1, 2, 3)

        # Make sure the function was not evaluated
        assert obj.get_num_calls() == 1

        # Now delete the cache file and load property again
        kill_cache_file()
        third_result = obj.my_list

        # Make sure the result looks right
        assert tuple(third_result) == (1, 2, 3)

        # Function should still not be evaluated because data was in memory
        assert obj.get_num_calls() == 1

        # The call should not have recreated the cache file
        assert not cache_file_exists()

        # Now create a new object and make sure its value looks right
        obj = TestingClass()
        result4 = obj.my_list
        assert tuple(result4) == (1, 2, 3)

        # This should have called the compute function once
        assert obj.get_num_calls() == 1

        # And the cache file should now exist
        assert cache_file_exists()

        # Now create yes another object and make sure the results look right
        obj = TestingClass()
        result5 = obj.my_list
        assert tuple(result5) == (1, 2, 3)

        # Results should have been loaded from pickle file, so no calls
        assert obj.get_num_calls() == 0

    # Clean up
    finally:
        del obj.my_list
        if hasattr(TestingClass, "pkc"):
            delattr(TestingClass, "pkc")


def test_pickle_cache_ignore_mode():
    try:
        # Set the testing class to ignore mode
        TestingClass.pkc = pickle_cache_state(mode="ignore")

        # Call the property multiple times making sure it's computed each time
        obj = TestingClass()
        assert obj.get_num_calls() == 0
        obj.my_list
        assert obj.get_num_calls() == 1
        obj.my_list
        assert obj.get_num_calls() == 2

    # Clean up
    finally:
        del obj.my_list
        if hasattr(TestingClass, "pkc"):
            delattr(TestingClass, "pkc")


def test_pickle_cache_default_file_creation():
    try:
        today = str(datetime.datetime.now().date())
        expected_name = "/tmp/TestingClassForFileCreation.my_list_{}.pickle".format(
            today
        )

        # Delete any existing picklefile and make sure it is gone
        if os.path.isfile(expected_name):
            os.unlink(expected_name)
        assert not os.path.isfile(expected_name)

        # Make sure the property has the expected value
        obj = TestingClassForFileCreation()
        assert obj.my_list == [1, 2, 3]

        # Make sure the expected cache file exists
        assert os.path.isfile(expected_name)

        # Make sure each call to the cache returns a new identical object
        assert obj.my_list == obj.my_list

        # import pdb; pdb.set_trace()
        assert id(obj.my_list) != id(obj.my_list)

    finally:
        del obj.my_list


def test_pickle_cache_custom_file_creation():
    try:
        expected_name = "/tmp/silly_pickle.pickle"

        # Delete any existing picklefile and make sure it is gone
        if os.path.isfile(expected_name):
            os.unlink(expected_name)
        assert not os.path.isfile(expected_name)

        # Make sure the property has the expected value
        obj = TestingClassForFileCreationCustom()
        assert obj.my_list == [1, 2, 3]

        # Make sure the expected cache file exists
        assert os.path.isfile(expected_name)

        # Make sure each call to the cache returns a new identical object
        assert obj.my_list == obj.my_list

        # Make sure the same object is returned
        assert id(obj.my_list) == id(obj.my_list)

    finally:
        del obj.my_list
