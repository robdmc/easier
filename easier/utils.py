import datetime
import glob
import os
import pickle
import sys
import traceback
import warnings
from copy import copy, deepcopy
from textwrap import dedent


def tqdm_flex(iterable):
    """
    Adds the appropriate tqdm wrapper around an iterable
    """
    import easier as ezr

    if ezr.in_notebook():
        try:
            import tqdm.notebook as tqdm

            return tqdm.tqdm(iterable)
        except:
            return iterable
    else:
        try:
            import tqdm

            return tqdm.tqdm(iterable)
        except:
            return iterable


def python_type():
    """
    A utility to determine if running under ipython or jupyter
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return "jupyter"  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return "ipython"  # Terminal running IPython
        else:
            return "other"  # Other type (?)
    except NameError:
        return "other"  # Probably standard Python interpreter


def in_notebook():
    """
    Determine if running in notebook (see python_type)
    """
    return python_type() == "jupyter"


def django_reconnect():  # pragma: no cover
    """
    Fixes dropped postgres connection in jupyter notebooks.
    """
    from django.db import connections

    conn = connections["default"]
    conn.connect()


def mute_warnings():  # pragma: no cover
    """
    Mute all Python warnings
    """
    import warnings

    warnings.filterwarnings("ignore")


def screen_width_full():  # pragma: no cover
    from IPython.core.display import display, HTML

    display(HTML("<style>.container { width:100% !important; }</style>"))


def print_error(tag="", verbose=False, buffer=None):  # pragma: no cover
    """
    Function for printing errors in except block.
    Args:
        tag: Optional string to print after exception info
        verbose: Only print traceback when verbose = True
        buffer: The buffer to print to (default: sys.stdout)
    """
    exc_type, exc_value, exc_traceback = sys.exc_info()

    if buffer is None:
        buffer = sys.stderr

    if verbose:
        traceback.print_tb(exc_traceback, limit=None, file=buffer)

    if tag:
        tag = f" :: {tag.strip()}"

    print(f"{exc_type.__name__}: {exc_value}{tag}", file=buffer)


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


class cached_container(object):
    """
    Decorator to cache containers in such a way that only copies are returned
    """

    def __init__(self, func):
        self.func = func

    def __get__(self, instance, type=None):
        if instance is None:
            return self

        cached_var_name = "_cached_container_for_" + self.func.__name__
        self._cached_var_name = cached_var_name

        if cached_var_name not in instance.__dict__:
            instance.__dict__[cached_var_name] = self.func(instance)
        try:
            out = instance.__dict__[cached_var_name].copy()
        except AttributeError:
            out = copy(instance.__dict__[cached_var_name])
        return out

    def __delete__(self, obj):
        delattr(obj, self._cached_var_name)


class cached_dataframe(cached_container):  # pragma: no cover
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn("@cached_dataframe is deprecated.  @Use cached_container")


class pickle_cache_state:
    """
    This is a descriptor that stores optional state for pickle cache
    """

    def __init__(self, mode=None):
        self.set_mode(mode)

    def set_mode(self, mode=None):
        allowed_modes = ["active", "ignore", "refresh", "reset", "memory"]
        if mode not in allowed_modes:
            raise ValueError(f"You must set mode to be one of {allowed_modes}")
        self.mode = mode

    def __get__(self, instance, owner):
        pass

    def __set__(self, instance, value):
        raise NotImplementedError("You cannot set this attribute")


class pickle_cache_mixin:
    """
    Inherit from this mixin to gain ability to enable/disable pickle_cache
    Has the following methods:
    .enable_pickle_cache              # Enables the pickle cache
    .disable_pickle_cache             # Sets pickle cache to reset mode
    .clear_all_default_pickle_cashes  # Removes ALL default-named cache files

    """

    @classmethod
    def enable_pickle_cache(cls):
        for name, obj in vars(cls).items():
            if isinstance(obj, pickle_cache_state):
                obj.set_mode("active")

    @classmethod
    def disable_pickle_cache(cls):
        for name, obj in vars(cls).items():
            if isinstance(obj, pickle_cache_state):
                obj.set_mode("reset")

    @classmethod
    def clear_all_default_pickle_cashes(cls):
        for file in glob.glob("/tmp/*_*-*-*.pickle"):
            os.unlink(file)


class pickle_cached_container:
    """
    This decorator creates cached containers (i.e. lists, dataframes, etc.)
    It will cache at two different levels.  Calling the property
    multiple times on the same object will return a copy
    of the in-memory cached object.
    If a new object is created, accessing the cached attribute
    will first look for a pickle file with the name specified in
    the decorator.  If that file exists, it will be loaded into
    the in-memory cache and returned.  If it does not exist, the
    logic in the decorated method is exectuted with the results
    being saved both in-memory and to pickle.
    Busting the cache is as simple as deleting the attribute.

    Example:

    class Loader:

        ## This is an optionalal class variable you can add
        ## for when you want to temporarily remove caching.
        ## The name of the class attribute does not matter.
        ##
        ## Modes:
        ##   'active': same as not even specifying this class attribute
        ##   'ignore': maintains all pickle files as they were but ignores the cache
        ##   'refresh': Recomputes and refreshes pickle file
        ##   'reset': Alias for refresh
        ##   'memory': Will recompute and cache on object only.  Not to file.
        pcs = ezr.pickle_cache_state(mode='active')

        # Default cache file /tmp/<cls>.<meth>.pickle
        @ezr.pickle_cached_property()
        def df(self):
            # expensive code to create a dataframe or dict or list
            out = my_expensive_function()
            return out
        @ezr.pickle_cached_property('/tmp/account_data.pickle', return_copy=False)
        def my_dict(self):
            '''
            This property will be cached, but any mutation you make to the property
            will actually mutate the cache.  The reason you might want to use this
            is that it saves a possibly expensive memory copy.  This defaults to
            True because mutating a cache can lead to all sorts of weird bugs.
            '''
            # expensive code to create a dataframe or dict or list
            out = my_expensive_function()
            return out
    loader = Loader()
    # Accesses the cached property, computing/storing if necessary
    # Note: will return a copy of the property to avoid mutation.
    df = loader.df
    # Bust the cache for the property.  This will remove the
    # in-memory cache and delete the pickle file.
    del loader.df
    """

    def __init__(self, pickle_file_name=None, return_copy=True):
        """
        This constructs the class that will decorate the property.
        It is used to record state we will need later
        """
        self._pickle_file_name = pickle_file_name
        self.return_copy = return_copy

    def __call__(self, func):
        """
        Once the decorator object has been initiated, this method
        will be called to do the actual decoration.  All it does is
        replace the method definition on the class with an instance
        of this decorator object, which is also a descriptor.  A
        reference to the initial method is stored on the
        decorator/descriptor class.
        """
        self.func = func
        self.cached_var_name = "_pickle_cache_for_" + self.func.__name__
        return self

    @property
    def default_pickle_file_name(self):
        return "/tmp/{}_{}.pickle".format(
            self.func.__qualname__, str(datetime.datetime.now().date())
        )

    @property
    def pickle_file_name(self):
        if self._pickle_file_name:
            return self._pickle_file_name
        else:
            return self.default_pickle_file_name

    def _get_cache_mode(self, instance):
        cache_mode = None
        for att in instance.__class__.__dict__.values():
            # If a cache state attribute was found on the class
            if isinstance(att, pickle_cache_state):
                # Get the mode from the state
                cache_mode = att.mode
                break
        return cache_mode

    def _get_pickle_or_compute(self, instance):
        """
        This method either pulls data from pickle file if it exists
        otherwise it will compute results, populate the pickle file and
        return the computed object.
        """
        # If pickle file exists, load its contents
        if os.path.isfile(self.pickle_file_name):
            with open(self.pickle_file_name, "rb") as buffer:
                obj = pickle.load(buffer)
        # If pickle file doesn't exist, evaluate the wrapped method
        # and save results to pickle file
        else:
            obj = self.func(instance)
            with open(self.pickle_file_name, "wb") as buffer:
                pickle.dump(obj, buffer)

        return obj

    def _get_memory_pickle_or_compute(self, instance):
        """
        Try to get the object trying in this order:
        1) From memory cached on the object
        2) From the pickle file
        3) By running the decorated method

        The retrieved value will be stored on the object if it isn't
        already there
        """
        if self.cached_var_name not in instance.__dict__:
            instance.__dict__[self.cached_var_name] = self._get_pickle_or_compute(
                instance
            )
        return instance.__dict__[self.cached_var_name]

    def _get_memory_or_compute(self, instance):
        """
        Try to get the object trying in this order:
        1) From memory cached on the object
        2) By running the decorated method

        The retrieved value will be stored on the object if it isn't
        already there
        """
        if self.cached_var_name not in instance.__dict__:
            instance.__dict__[self.cached_var_name] = self.func(instance)
        return instance.__dict__[self.cached_var_name]

    def _get_or_compute(self, instance, cache_mode):
        # If ignoring the cache, always call the decorated method
        if cache_mode == "ignore":
            return self.func(instance)
        # If memory, ignore the pickle file but use object caching
        elif cache_mode == "memory":
            return self._get_memory_or_compute(instance)

        # This looks weird, but it is the same thing as deleting
        # the pickle-cached property on the host object.  Doing so
        # will bust the cache. So refreshing busts the cache
        # and repopulates it by computing.
        elif cache_mode in ["refresh", "reset"]:
            self.__delete__(instance)
            return self._get_memory_pickle_or_compute(instance)

        # Otherwise use object and pickle caching
        else:
            return self._get_memory_pickle_or_compute(instance)

    def _copy_object(self, obj):
        try:
            out = obj.copy()
        except AttributeError:
            out = copy(obj)
        return out

    def __get__(self, instance, type=None):
        """
        After decoration the decorated method will be replaced with
        an instance of the decorator/descriptor.  Every time this instance
        is accessed this method will be called to return the value
        of the method, which has been turned into a pickle-backed property.
        """
        # Get the cache mode
        cache_mode = self._get_cache_mode(instance)

        # Grab the object, (persisting to cache if required)
        obj = self._get_or_compute(instance, cache_mode)

        # Copy the object if appropriate
        if self.return_copy:
            return self._copy_object(obj)
        else:
            return obj

    def __delete__(self, instance):
        """
        This method handles busting the cache.
        """
        # Delete the cached copy of the data
        if self.cached_var_name in instance.__dict__:
            del instance.__dict__[self.cached_var_name]

        # Delete the pickle file
        if os.path.isfile(self.pickle_file_name):
            os.unlink(self.pickle_file_name)


class BlobAttr:
    def __init__(self, default, deep=True):
        if deep:
            self.copy_func = deepcopy
        else:
            self.copy_func = copy

        self._default = default
        self.name = None

    @property
    def default(self):
        return self.copy_func(self._default)

    def __get__(self, obj, cls=None):
        if obj is None:
            return
        else:
            return obj._blob_attr_state[self.name]
            # return self.copy_func(obj._blob_attr_state[self.name])

    def __set__(self, obj, value):
        if obj is None:
            return
        else:
            obj._blob_attr_state[self.name] = self.copy_func(value)


class BlobMixin:
    """
    Inherit from this mixin to get serializable attributes.  This mixin
    defines two methods on the inherited class.  .to_blob() and .from_blob()

    These methods will return (deep) copied versions of all BlobAttr instances
    defined on the base class.  All BlobAttr definitions must include a default
    value.
    Look at the output of BlobMixin.example() too see examples
    """

    @staticmethod
    def example():
        return dedent(
            """
            import easier as ezr


            class Parameters(ezr.BlobMixin):
                drums = ezr.BlobAttr({
                    'first': 'Ringo',
                    'last': 'Star',
                })

                bass = ezr.BlobAttr({
                    'first': 'Paul',
                    'last': 'McCartney',
                })


            # Instantiate a default instance and look at parameters
            params = Parameters()
            print(params.drums, params.bass)

            # Change an attribute explicity
            params.drums = {'first': 'Charlie ', 'last': 'Watts'}

            # Update attributes from a blob
            params.from_blob({'bass': {'first': 'Bill', 'last': 'Wyman'}})

            # Dump the updated attributes to a blob
            blob = params.to_blob()
            print(blob)

            # Update the return blob back to defaults
            blob.update(params.blob_defaults)

            # Load the updated blob back into the params
            params.from_blob(blob)

            # Print the updated results
            print(params.drums, params.bass)
        """
        )

    def __init__(self):
        self._blob_attr_state = {}

        for att_name, att_kind in self.__class__.__dict__.items():
            if isinstance(att_kind, BlobAttr):
                att_kind.name = att_name
                self._blob_attr_state[att_name] = att_kind.default

        self._blob_attr_state_defaults = deepcopy(self._blob_attr_state)

    @property
    def blob_defaults(self):
        return deepcopy(self._blob_attr_state_defaults)

    def to_blob(self):
        return {
            name: deepcopy(getattr(self, name)) for name in self._blob_attr_state.keys()
        }

    def from_blob(self, blob, strict=False):
        blob = deepcopy(blob)
        msg = ""
        extra_keys = set(blob.keys()) - set(self._blob_attr_state.keys())
        missing_keys = set(self._blob_attr_state.keys()) - set(blob.keys())
        # TODO: I need to write tests aroud this.  This is new functionality where when not in strict
        # node, extra blob keys just get ignored
        if extra_keys:
            if strict:
                msg += f"\nBad Blob. These keys unrecognized: {list(extra_keys)}"
            else:
                for key in extra_keys:
                    del blob[key]
        if strict and missing_keys:
            msg += f"\nBad Blob.  These required keys not found: {list(missing_keys)}"
        if msg:
            raise ValueError(msg)

        for key, val in blob.items():
            setattr(self, key, val)

        return self


class Scaler(BlobMixin):
    """
    Scales an arry to have values between 0 and 1.
    With min/max appearing at 0/1 respectively.

    Follows the sklearn transformer api.

    The transformer state is (de)serialized with the
    (from/to)_blob methods.
    """

    limits = BlobAttr(None)

    def fit(self, x):
        import numpy as np

        self.limits = [np.min(x), np.max(x)]
        return self

    def _ensure_fitted(self):
        if self.limits is None:
            raise ValueError("You must fit or load params before you can transform")

    def transform(self, x):
        self._ensure_fitted()
        xf = (x - self.limits[0]) / (self.limits[1] - self.limits[0])
        return xf

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x):
        self._ensure_fitted()
        xr = self.limits[0] + x * (self.limits[1] - self.limits[0])
        return xr


def get_logger(name, level="info"):
    import logging
    import daiquiri

    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    allowed_levels = list(level_map.keys())
    if level not in allowed_levels:
        raise ValueError(f"level must be in {allowed_levels}")

    daiquiri.setup(level=level_map[level])
    logger = daiquiri.getLogger(name)
    return logger
