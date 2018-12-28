from typing import Union, List, Iterable
import datetime
import re

from dateutil.parser import parse
import numpy as np
import pandas as pd
from scipy.stats import scoreatpercentile


def mute_warnings():
    """
    Mute all Python warnings
    """
    import warnings
    warnings.filterwarnings("ignore")


class ChattyDict(dict):
    """
    A dict subclass that throws keyerror with existing
    keys.
    """
    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            raise KeyError(f'{key!r} not in {list(self.keys())}')


class cached_property(object):
    """
    Decorator that converts a method with a single self argument into a
    property cached on the instance.

    Clear the cache by just deleteing the property

    class Person:
        @cached_property
        def first_name(self):
            return 'Monte'

    p = Person()

    # Compute and return first name
    f = p.first_name

    # Accessed cached first name by calling again
    f = p.first_name

    # Clear the cache by deleting the property
    # Will raise attribute error if cache is empty so try/catch
    # is a good idea
    try:
        del r.first_name
    except AttributeError:
        pass

    # Recompute and return first name (after clearing cache)
    f = p.first_name

    This is a direct copy-paste of Django's cached property from
    https://github.com/django/django/blob/2456ffa42c33d63b54579eae0f5b9cf2a8cd3714/django/utils/functional.py#L38-50
    """
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, type=None):
        if instance is None:
            return self
        res = instance.__dict__[self.func.__name__] = self.func(instance)
        return res
