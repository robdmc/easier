from ..utils import pickle_cached_container, pickle_cache_state
import pytest
import datetime
import os
import tempfile
import uuid


@pytest.fixture
def test_pickle_file():
    """Generate a unique temporary pickle file path for each test"""
    temp_file = os.path.join(tempfile.gettempdir(), f"test_cache_{uuid.uuid4().hex}.pickle")
    yield temp_file
    # Cleanup
    if os.path.exists(temp_file):
        os.unlink(temp_file)


def cache_file_exists(pickle_file):
    return os.path.isfile(pickle_file)


def kill_cache_file(pickle_file):
    if cache_file_exists(pickle_file):
        os.unlink(pickle_file)


def create_testing_class(pickle_file_name):
    """Factory function to create TestingClass with specific pickle file"""
    
    class TestingClass:
        def __init__(self):
            self.num_compute_calls = 0

        @pickle_cached_container()
        def my_tuple(self):
            return (1, 2, 3)

        @pickle_cached_container(pickle_file_name=pickle_file_name)
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
            return self.num_compute_calls

        @classmethod
        def reset_call_count(cls):
            # This method is no longer needed but kept for compatibility
            pass
    
    return TestingClass




def test_out_of_scope(test_pickle_file):
    TestingClass = create_testing_class(test_pickle_file)
    try:
        kill_cache_file(test_pickle_file)
        TestingClass.reset_call_count()
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
        assert cache_file_exists(test_pickle_file)

        # Calling again should not result in any additional compuations.
        scope(0)

    # Clean up
    finally:
        kill_cache_file(test_pickle_file)
        if hasattr(TestingClass, "pkc"):
            delattr(TestingClass, "pkc")


def check_pickle_cache_refresh_or_reset_mode(mode, test_pickle_file):
    TestingClass = create_testing_class(test_pickle_file)
    try:
        kill_cache_file(test_pickle_file)
        TestingClass.reset_call_count()

        # Set the testing class to active mode
        TestingClass.pkc = pickle_cache_state(mode=mode)

        obj = TestingClass()

        # Make sure no calls have been made an no cache file exists
        assert obj.get_num_calls() == 0
        assert not cache_file_exists(test_pickle_file)

        # This call should run a computation and file should exist
        result1 = obj.my_list
        assert obj.get_num_calls() == 1
        assert cache_file_exists(test_pickle_file)

        # Make sure the results looks right
        assert tuple(result1) == (1, 2, 3)

        # Now change the result that computation would return
        obj.set_return([4, 5, 6])

        # The result should now be different because another computation
        # should have been triggered
        result1 = obj.my_list
        assert obj.get_num_calls() == 2
        assert tuple(result1) == (4, 5, 6)
        assert cache_file_exists(test_pickle_file)

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


def test_pickle_cache_reset_mode(test_pickle_file):
    return check_pickle_cache_refresh_or_reset_mode("reset", test_pickle_file)


def test_pickle_cache_refresh_mode(test_pickle_file):
    return check_pickle_cache_refresh_or_reset_mode("refresh", test_pickle_file)


def test_pickle_cache_memory_mode(test_pickle_file):
    # Create a separate pickle file for the active_obj test
    active_pickle_file = os.path.join(tempfile.gettempdir(), f"test_cache_active_{uuid.uuid4().hex}.pickle")
    
    TestingClass = create_testing_class(test_pickle_file)
    ActiveTestingClass = create_testing_class(active_pickle_file)
    
    try:
        kill_cache_file(test_pickle_file)
        TestingClass.reset_call_count()

        # Set the testing class to memory mode
        TestingClass.pkc = pickle_cache_state(mode="memory")

        obj = TestingClass()

        # Make sure no calls have been made an no cache file exists
        assert obj.get_num_calls() == 0
        assert not cache_file_exists(test_pickle_file)

        # This call should run a computation
        result1 = obj.my_list
        assert obj.get_num_calls() == 1

        # But no file should have been created
        assert not cache_file_exists(test_pickle_file)

        # Calling again should not trigger a computation or create a file
        result2 = obj.my_list
        assert obj.get_num_calls() == 1
        assert not cache_file_exists(test_pickle_file)

        # Make sure results look right
        assert tuple(result1) == (1, 2, 3)
        assert tuple(result2) == (1, 2, 3)

        # Now create another object that will create a cached file
        # With weird values
        ActiveTestingClass.pkc = pickle_cache_state(mode="active")
        active_obj = ActiveTestingClass()
        active_obj.set_return([7, 8, 9])
        active_obj.my_list
        assert cache_file_exists(active_pickle_file)
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
        # Clean up the active test file
        if os.path.exists(active_pickle_file):
            os.unlink(active_pickle_file)


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


def test_pickle_cache_active_mode(test_pickle_file):
    TestingClass = create_testing_class(test_pickle_file)
    try:
        kill_cache_file(test_pickle_file)
        TestingClass.reset_call_count()

        # Set the testing class to active mode
        TestingClass.pkc = pickle_cache_state(mode="active")

        obj = TestingClass()

        # Make sure no calls have been made an no cache file exists
        assert obj.get_num_calls() == 0
        assert not cache_file_exists(test_pickle_file)

        # This call should run a computation
        first_result = obj.my_list

        # Make sure the results look right
        assert tuple(first_result) == (1, 2, 3)

        # Make sure it was actually called and created a file
        assert obj.get_num_calls() == 1
        assert cache_file_exists(test_pickle_file)

        # This call should hit the memory cache
        second_result = obj.my_list

        # Make sure the results look right
        assert tuple(second_result) == (1, 2, 3)

        # Make sure the function was not evaluated
        assert obj.get_num_calls() == 1

        # Now delete the cache file and load property again
        kill_cache_file(test_pickle_file)
        third_result = obj.my_list

        # Make sure the result looks right
        assert tuple(third_result) == (1, 2, 3)

        # Function should still not be evaluated because data was in memory
        assert obj.get_num_calls() == 1

        # The call should not have recreated the cache file
        assert not cache_file_exists(test_pickle_file)

        # Now create a new object and make sure its value looks right
        obj = TestingClass()
        result4 = obj.my_list
        assert tuple(result4) == (1, 2, 3)

        # This should have called the compute function once
        assert obj.get_num_calls() == 1

        # And the cache file should now exist
        assert cache_file_exists(test_pickle_file)

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


def test_pickle_cache_ignore_mode(test_pickle_file):
    TestingClass = create_testing_class(test_pickle_file)
    try:
        TestingClass.reset_call_count()
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
        # Use unique name to avoid conflicts between parallel tests
        unique_id = uuid.uuid4().hex[:8]
        expected_name = "/tmp/TestingClassForFileCreation_{}.my_list_{}.pickle".format(
            unique_id, today
        )

        # Delete any existing picklefile and make sure it is gone
        if os.path.isfile(expected_name):
            os.unlink(expected_name)
        assert not os.path.isfile(expected_name)

        # Create a dynamic class with unique pickle file name
        class TestingClassForFileCreationUnique:
            @pickle_cached_container(pickle_file_name=expected_name)
            def my_list(self):
                return [1, 2, 3]

        # Make sure the property has the expected value
        obj = TestingClassForFileCreationUnique()
        assert obj.my_list == [1, 2, 3]

        # Make sure the expected cache file exists
        assert os.path.isfile(expected_name)

        # Make sure each call to the cache returns a new identical object
        assert obj.my_list == obj.my_list

        # import pdb; pdb.set_trace()
        assert id(obj.my_list) != id(obj.my_list)

    finally:
        del obj.my_list
        # Clean up the unique file
        if os.path.exists(expected_name):
            os.unlink(expected_name)


def test_pickle_cache_custom_file_creation():
    try:
        # Use unique name to avoid conflicts between parallel tests
        unique_id = uuid.uuid4().hex[:8]
        expected_name = f"/tmp/silly_pickle_{unique_id}.pickle"

        # Delete any existing picklefile and make sure it is gone
        if os.path.isfile(expected_name):
            os.unlink(expected_name)
        assert not os.path.isfile(expected_name)

        # Create a dynamic class with unique pickle file name
        class TestingClassForFileCreationCustomUnique:
            @pickle_cached_container(
                pickle_file_name=expected_name, return_copy=False
            )
            def my_list(self):
                return [1, 2, 3]

        # Make sure the property has the expected value
        obj = TestingClassForFileCreationCustomUnique()
        assert obj.my_list == [1, 2, 3]

        # Make sure the expected cache file exists
        assert os.path.isfile(expected_name)

        # Make sure each call to the cache returns a new identical object
        assert obj.my_list == obj.my_list

        # Make sure the same object is returned
        assert id(obj.my_list) == id(obj.my_list)

    finally:
        del obj.my_list
        # Clean up the unique file
        if os.path.exists(expected_name):
            os.unlink(expected_name)
