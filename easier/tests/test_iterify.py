from ..iterify import iterify
import numpy as np


class Tester:
    def __call__(self, x, iter_constructor=list):
        wrapper = iterify(iter_constructor=iter_constructor)
        x = wrapper.get_input(x)
        self.inner_type = type(x)
        out = [v + 10 for v in x]
        return wrapper.get_output(out)


def test_scalars():
    tester = Tester()
    result = tester(7)
    assert type(result) == int
    assert result == 17
    assert tester.inner_type == list


def test_lists():
    tester = Tester()
    result = tester([7])
    assert type(result) == list
    assert(len(result)) == 1
    assert result[0] == 17
    assert tester.inner_type == list


def test_arrays():
    tester = Tester()
    result = tester(np.array([7]))
    assert type(result) == np.ndarray
    assert(len(result)) == 1
    assert result[0] == 17
    assert tester.inner_type == np.ndarray


def test_forcing_type():
    tester = Tester()
    result = tester([7], iter_constructor=np.array)
    assert type(result) == np.ndarray
    assert(len(result)) == 1
    assert result[0] == 17
    assert tester.inner_type == np.ndarray
