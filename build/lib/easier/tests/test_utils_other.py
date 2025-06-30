from ..utils import cached_container, cached_property, Scaler
import unittest


class ClassToTest:
    def __init__(self, start_value=0):
        self.num_compute_calls = 0
        self.start_value = start_value

    @cached_container
    def my_list(self):
        self.num_compute_calls += 1
        return list(range(self.start_value, self.start_value + 3))

    @my_list.deleter
    def my_list(self):
        if self.num_compute_calls > 0:
            self.num_compute_calls -= 1

    @cached_container
    def my_tuple(self):
        self.num_compute_calls += 1
        return (1, 2, 3)

    @cached_container
    def my_other_list(self):
        self.num_compute_calls += 1
        return [10 * v for v in self.my_list]

    def get_num_calls(self):
        if hasattr(self, "num_compute_calls"):
            return self.num_compute_calls
        else:
            return 0


class ClassToTestNoDeleter:
    def __init__(self, start_value=0):
        self.num_compute_calls = 0
        self.start_value = start_value

    @cached_container
    def my_list(self):
        self.num_compute_calls += 1
        return list(range(self.start_value, self.start_value + 3))


class TestCachedContainer(unittest.TestCase):
    def test_cache_hits(self):
        obj = ClassToTest()
        self.assertEqual(obj.num_compute_calls, 0)

        _ = obj.my_list
        self.assertEqual(obj.num_compute_calls, 1)

        _ = obj.my_list
        self.assertEqual(obj.num_compute_calls, 1)

    def test_correct_values(self):
        obj = ClassToTest()
        result = obj.my_list
        self.assertListEqual(result, [0, 1, 2])

    def test_copies_returned(self):
        obj = ClassToTest()
        r1 = obj.my_list
        r2 = obj.my_list
        self.assertFalse(id(r1) == id(r2))

    def test_mutation(self):
        obj = ClassToTest()
        r1 = obj.my_list
        self.assertListEqual(r1, [0, 1, 2])

        r1[0] = 777

        self.assertListEqual(r1, [777, 1, 2])

        r2 = obj.my_list
        self.assertListEqual(r2, [0, 1, 2])

    def test_deleting(self):
        obj = ClassToTest()
        self.assertEqual(obj.num_compute_calls, 0)

        _ = obj.my_list
        self.assertEqual(obj.num_compute_calls, 1)

        del obj.my_list
        self.assertEqual(obj.num_compute_calls, 0)

        result2 = obj.my_list
        self.assertEqual(obj.num_compute_calls, 1)
        self.assertListEqual(result2, [0, 1, 2])

    def test_deleting_with_no_deleter(self):
        obj = ClassToTestNoDeleter()
        self.assertEqual(obj.num_compute_calls, 0)

        _ = obj.my_list
        self.assertEqual(obj.num_compute_calls, 1)

        del obj.my_list
        self.assertEqual(obj.num_compute_calls, 1)

        result2 = obj.my_list
        self.assertEqual(obj.num_compute_calls, 2)
        self.assertListEqual(result2, [0, 1, 2])


# class TestingClass:
#     num_compute_calls = 0

#     @cached_container
#     def my_list(self):
#         return self.compute_list()

#     @cached_container
#     def my_tuple(self):
#         return (1, 2, 3)

#     @cached_property
#     def my_other_list(self):
#         return self.compute_list()

#     def compute_list(self):
#         self.num_compute_calls += 1
#         if not hasattr(self, "_return"):
#             self.set_return([1, 2, 3])
#         return self._return

#     def set_return(self, val):
#         self._return = val

#     def get_num_calls(self):
#         if hasattr(self, "num_compute_calls"):
#             return self.num_compute_calls
#         else:
#             return 0


# def test_cached_container():
#     # Create the object and make sure no calls
#     obj = TestingClass()
#     assert obj.num_compute_calls == 0

#     # First call should hit computation
#     result1 = obj.my_list
#     assert obj.num_compute_calls == 1

#     # Second call should not
#     result2 = obj.my_list
#     assert obj.num_compute_calls == 1

#     # The returned objects should be copies of each other
#     assert id(result1) != result2

#     # Oddly enough copy.copy returns the same object when it's a tuple
#     assert id(obj.my_tuple) == id(obj.my_tuple)

#     # I'm not sure why django asks for return of cached property on
#     # unbound class, but test that the right thing is returned.
#     assert "func" in TestingClass.my_list.__dict__


# def test_cached_property():
#     # Create the object and make sure no calls
#     obj = TestingClass()
#     assert obj.num_compute_calls == 0

#     # First call should hit computation
#     result1 = obj.my_other_list
#     assert obj.num_compute_calls == 1

#     # Second call should not
#     result2 = obj.my_other_list
#     assert obj.num_compute_calls == 1

#     # The returned objects should actually be the same object
#     assert id(result1) == id(result2)

#     # I'm not sure why django asks for return of cached property on
#     # unbound class, but test that the right thing is returned.
#     assert "func" in TestingClass.my_other_list.__dict__


def test_scaler():
    import numpy as np

    x = np.arange(5, 11)
    scaler = Scaler()
    xt = scaler.fit_transform(x)
    max_diff = np.abs(np.max(xt - np.linspace(0, 1, 6)))
    assert max_diff < 1e-6

    blob = scaler.to_blob()
    scaler2 = Scaler()
    scaler2.from_blob(blob)
    xr = scaler2.inverse_transform(xt)
    max_diff = np.abs(np.max(x - xr))
    assert max_diff < 1e-6

    assert isinstance(blob, dict)
