from ..utils import pickle_cached_container, pickle_cache_state
import pytest
import datetime
import os
import mock

# @pytest.fixture
# def test_class():


class NameTestingClass:
    def my_list(self):
        return [1, 2, 3]


class TestingClass:
    @pickle_cached_container()
    def my_list(self):
        return self.compute_list()

    def compute_list(self):
        return [1, 2, 3]


class TestingClassForFileCreation:
    @pickle_cached_container()
    def my_list(self):
        return [1, 2, 3]


class TestingClassForFileCreationCustom:
    @pickle_cached_container(pickle_file_name='/tmp/silly_pickle.pickle', return_copy=False)
    def my_list(self):
        return [1, 2, 3]



@mock.patch(TestingClass.compute_list)
def test_pickle_cache_ignore_mode(compute_list_mock):
    try:
        TestingClass.pkc = pickle_cache_state(mode='ignore')
        import pdb;
        pdb.set_trace()

        obj = TestingClass()
        obj.my_list
        assert compute_list_mock.called

    finally:
        del obj.my_list
        if hasattr(TestingClass, 'pkc'):
            delattr(TestingClass, 'pkc')

def test_pickle_cache_default_file_creation():
    try:
        today = str(datetime.datetime.now().date())
        expected_name = '/tmp/TestingClassForFileCreation.my_list_{}.pickle'.format(today)

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
        # Clean up
        if os.path.isfile(expected_name):
            os.unlink(expected_name)


def test_pickle_cache_custom_file_creation():
    try:
        expected_name = '/tmp/silly_pickle.pickle'

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
        # Clean up
        if os.path.isfile(expected_name):
            os.unlink(expected_name)
    