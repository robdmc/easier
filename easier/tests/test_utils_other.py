from ..utils import cached_container, cached_property, Scaler


class TestingClass:
    num_compute_calls = 0

    @cached_container
    def my_list(self):
        return self.compute_list()

    @cached_container
    def my_tuple(self):
        return (1, 2, 3)

    @cached_property
    def my_other_list(self):
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


def test_cached_container():
    # Create the object and make sure no calls
    obj = TestingClass()
    assert obj.num_compute_calls == 0

    # First call should hit computation
    result1 = obj.my_list
    assert obj.num_compute_calls == 1

    # Second call should not
    result2 = obj.my_list
    assert obj.num_compute_calls == 1

    # The returned objects should be copies of each other
    assert id(result1) != result2

    # Oddly enough copy.copy returns the same object when it's a tuple
    assert id(obj.my_tuple) == id(obj.my_tuple)

    # I'm not sure why django asks for return of cached property on
    # unbound class, but test that the right thing is returned.
    assert "func" in TestingClass.my_list.__dict__


def test_cached_property():
    # Create the object and make sure no calls
    obj = TestingClass()
    assert obj.num_compute_calls == 0

    # First call should hit computation
    result1 = obj.my_other_list
    assert obj.num_compute_calls == 1

    # Second call should not
    result2 = obj.my_other_list
    assert obj.num_compute_calls == 1

    # The returned objects should actually be the same object
    assert id(result1) == id(result2)

    # I'm not sure why django asks for return of cached property on
    # unbound class, but test that the right thing is returned.
    assert "func" in TestingClass.my_other_list.__dict__


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
