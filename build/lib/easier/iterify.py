class iterify:
    """
    This class wraps arguments that can either be iterable or not.
    If an iterable variable is provided, then that iterable variable
    is provided back from the .get_input_value() method and can be used
    in the function.

    If a non-iterable variable is provided, that variable is transformed
    to a single-element iterable and returned from the get_input_value() method.

    The get_output_value() method will return an object with the iterable property
    match what the input object had.

    This is useful if you want a function to handle both iterable and non-iterable
    versions of an argument.  You can just write the iterable code and this wrapper
    will trasform non iterable inputs to iterable, and then tranform outputs back
    to non-iterable.
    """

    def __init__(self, iter_constructor=list):
        self.iter_constructor = iter_constructor

    def _update_constructor(self, obj):
        import numpy as np

        if isinstance(obj, np.ndarray) and self.iter_constructor is list:
            self.iter_constructor = np.array

    def _make_iterable(self, x):
        if hasattr(x, "__iter__"):
            if self._constructed_type == type(x):
                return x
            else:
                return self.iter_constructor(x)
        else:
            return self.iter_constructor([x])

    def _store_state(self, obj):
        self._is_iterable = hasattr(obj, "__iter__")
        self._input_type = type(obj)

        if self._is_iterable:
            self._constructed_type = type(self.iter_constructor(obj[:2]))
        else:
            self._constructed_type = list

    def get_input(self, obj):
        self._update_constructor(obj)
        self._store_state(obj)
        return self._make_iterable(obj)

        if self.is_iterable:
            return obj
        else:
            return self.iter_constructor([obj])

    def get_output(self, obj):
        if self._is_iterable:
            if not isinstance(obj, self._constructed_type):
                obj = self.iter_constructor(obj)
            return obj
        else:
            return obj[0]
