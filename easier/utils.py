try:
    from IPython.display import display, HTML
except ImportError:

    def display(obj):
        print(obj)

    def HTML(text):
        return text


from copy import copy, deepcopy
from django.db import connections
from textwrap import dedent
import daiquiri
import datetime
import difflib
import easier as ezr
import glob
import html
import logging
import numpy as np
import os
import pickle
import sys
import traceback
import warnings


def mute_warnings():
    """
    Mute all Python warnings
    """
    warnings.filterwarnings("ignore")


def print_error(tag="", verbose=False, buffer=None):
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


class cached_container:
    """
    Decorates a method so that it's return value is cached.  The returned value is
    always a copy to prevent mutation of the cached data.

    The cache can be invalidated by simply deleting the property.



    """

    def __init__(self, fget=None, fdel=None, doc=None):
        """
        Example:

        ```python
        class MyClass:

            # The cached_container decorator was based on the
            # pure-python implementation of property in the python docs.
            # https://docs.python.org/3/howto/descriptor.html#properties

            def __init__(self):
                self._computation_count = 0

            @cached_container
            def df_expensive(self):
                # This will only be computed once per instance
                self._computation_count += 1
                return some_expensive_operation()

            @df_expensive.deleter
            def df_expensive(self):
                # DEFINING THIS METHOD IS OPTIONAL
                # You should only need it if you need to do any other cleanup on invalidation
                # This will be called immediately before the cache is invalidated
                print(f"Cache invalidated after {self._computation_count} computations")
                self._computation_count = 0

        obj = MyClass()

        # First call computes and caches the result
        result1 = obj.df_expensive  # _computation_count = 1

        # Second call returns cached copy
        result2 = obj.df_expensive  # _computation_count still = 1

        # Clear the cache - this will call the deleter
        del obj.df_expensive  # Prints: "Cache invalidated after 1 computations"

        # Next access will recompute
        result3 = obj.df_expensive  # _computation_count = 1 again
        ```

        Note:
            - The cached values are stored per instance
            - Returned values are always copies to prevent mutation of cached data
            - The cache can be cleared by deleting the property
            - The deleter-decorated method is called just before the cache is cleared

        Args:
            fget: The getter function that computes the value to be cached
            fdel: Optional deleter function to be called when cache is cleared
            doc: Optional docstring for the property. If None and fget has a docstring,
                the fget's docstring will be used.
        """
        self.fget = fget
        self.fdel = fdel
        self._cache = {}
        if doc is None and fget is not None:
            doc = fget.__doc__
        self.__doc__ = doc

    def __set_name__(self, owner, name):
        """Set the name of the property on the owner class.

        This is called automatically when the descriptor is assigned to a class.
        """
        self.__name__ = name

    def __get__(self, obj, objtype=None):
        """Get the cached value for the instance.

        If the value hasn't been computed yet, it will be computed and cached.
        The returned value is always a copy to prevent mutation of the cached data.

        Args:
            obj: The instance to get the cached value for
            objtype: The class of the instance (unused)

        Returns:
            A copy of the cached value

        Raises:
            AttributeError: If no getter function was provided
        """
        if obj is None:
            return self
        if self.fget is None:
            raise AttributeError
        if obj not in self._cache:
            self._cache[obj] = self.fget(obj)
        try:
            return self._cache[obj].copy()
        except AttributeError:
            pass
        try:
            return self._cache[obj].clone()
        except AttributeError:
            pass
        return copy(self._cache[obj])

    def __delete__(self, obj):
        """Clear the cached value for the instance.

        This will:
        1. Call the deleter function if one was provided
        2. Remove the cached value for this instance

        Args:
            obj: The instance to clear the cache for
        """
        if self.fdel is not None:
            self.fdel(obj)
        if obj in self._cache:
            del self._cache[obj]

    def deleter(self, fdel):
        """Create a new descriptor with the given deleter function.

        This allows the descriptor to be used with the @property.deleter syntax.

        Args:
            fdel: The deleter function to be called when the cache is cleared

        Returns:
            A new cached_container instance with the given deleter
        """
        return type(self)(self.fget, fdel, self.__doc__)


class cached_dataframe(cached_container):

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
            if isinstance(att, pickle_cache_state):
                cache_mode = att.mode
                break
        return cache_mode

    def _get_pickle_or_compute(self, instance):
        """
        This method either pulls data from pickle file if it exists
        otherwise it will compute results, populate the pickle file and
        return the computed object.
        """
        if os.path.isfile(self.pickle_file_name):
            with open(self.pickle_file_name, "rb") as buffer:
                obj = pickle.load(buffer)
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
        if cache_mode == "ignore":
            return self.func(instance)
        elif cache_mode == "memory":
            return self._get_memory_or_compute(instance)
        elif cache_mode in ["refresh", "reset"]:
            self.__delete__(instance)
            return self._get_memory_pickle_or_compute(instance)
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
        cache_mode = self._get_cache_mode(instance)
        obj = self._get_or_compute(instance, cache_mode)
        if self.return_copy:
            return self._copy_object(obj)
        else:
            return obj

    def __delete__(self, instance):
        """
        This method handles busting the cache.
        """
        if self.cached_var_name in instance.__dict__:
            del instance.__dict__[self.cached_var_name]
        if os.path.isfile(self.pickle_file_name):
            os.unlink(self.pickle_file_name)


class BlobAttr:
    """
    A descriptor class for managing serializable attributes with optional deep copying.

    Args:
        default: The default value for the attribute
        deep (bool, optional): Whether to use deep copy for the attribute. Defaults to True.
    """

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

    def __set__(self, obj, value):
        if obj is None:
            return
        else:
            if self.copy_func == deepcopy:
                # For deep attributes, use the copy function
                obj._blob_attr_state[self.name] = self.copy_func(value)
            else:
                # For non-deep attributes, store a reference to allow mutations
                obj._blob_attr_state[self.name] = value


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
            "\n            import easier as ezr\n\n\n            class Parameters(ezr.BlobMixin):\n                drums = ezr.BlobAttr({\n                    'first': 'Ringo',\n                    'last': 'Star',\n                })\n\n                bass = ezr.BlobAttr({\n                    'first': 'Paul',\n                    'last': 'McCartney',\n                })\n\n\n            # Instantiate a default instance and look at parameters\n            params = Parameters()\n            print(params.drums, params.bass)\n\n            # Change an attribute explicity\n            params.drums = {'first': 'Charlie ', 'last': 'Watts'}\n\n            # Update attributes from a blob\n            params.from_blob({'bass': {'first': 'Bill', 'last': 'Wyman'}})\n\n            # Dump the updated attributes to a blob\n            blob = params.to_blob()\n            print(blob)\n\n            # Update the return blob back to defaults\n            blob.update(params.blob_defaults)\n\n            # Load the updated blob back into the params\n            params.from_blob(blob)\n\n            # Print the updated results\n            print(params.drums, params.bass)\n        "
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
        # Check if we have any non-deep attributes that need special handling
        has_non_deep = False
        for key in blob.keys():
            # Check the entire class hierarchy for the attribute
            attr = None
            for cls in self.__class__.__mro__:
                if key in cls.__dict__ and isinstance(cls.__dict__[key], BlobAttr):
                    attr = cls.__dict__[key]
                    break
            if attr is not None and attr.copy_func != deepcopy:
                has_non_deep = True
                break

        if has_non_deep:
            # Don't deepcopy the blob if we have non-deep attributes
            # This allows mutations to propagate back to the original blob
            blob_copy = blob
        else:
            # Deepcopy the blob for deep attributes
            blob_copy = deepcopy(blob)

        msg = ""
        extra_keys = set(blob_copy.keys()) - set(self._blob_attr_state.keys())
        missing_keys = set(self._blob_attr_state.keys()) - set(blob_copy.keys())
        if extra_keys:
            if strict:
                msg += f"\nBad Blob. These keys unrecognized: {list(extra_keys)}"
            else:
                # In non-strict mode, we should still raise an error for unrecognized keys
                # but we can be more lenient about missing keys
                msg += f"\nBad Blob. These keys unrecognized: {list(extra_keys)}"
        if strict and missing_keys:
            msg += f"\nBad Blob.  These required keys not found: {list(missing_keys)}"
        if msg:
            raise ValueError(msg)

        # Handle non-deep assignments by checking each attribute's deep setting
        for key, val in blob_copy.items():
            # Use setattr to go through the BlobAttr __set__ method
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


def _generate_html_diff_side(words, opcodes, side):
    """Helper function to generate HTML for one side of the diff view."""
    html = []
    for tag, i1, i2, j1, j2 in opcodes:
        if side == "left":
            content = " ".join(words[i1:i2])
            if tag == "equal":
                html.append(f'<span style="color:#333;">{content}</span>')
            elif tag in ("replace", "delete"):
                html.append(
                    f'<span style="background-color:#ffecec; color:#c33;">{content}</span>'
                )
        else:
            content = " ".join(words[j1:j2])
            if tag == "equal":
                html.append(f'<span style="color:#333;">{content}</span>')
            elif tag in ("replace", "insert"):
                html.append(
                    f'<span style="background-color:#eaffea; color:#282;">{content}</span>'
                )
    return html


def diff_strings(original_text, modified_text, as_html=False, stand_alone=False):
    """
    Compare two text strings and generate a human-readable diff.

    Args:
        original_text: First string to compare
        modified_text: Second string to compare
        as_html: If True, returns HTML for a side-by-side diff view; if False, returns text-based diff
        stand_alone: If True and as_html=True, returns a complete HTML document with proper headers and styling

    Returns:
        A string containing either:
        - Text-based diff with insertions and deletions marked (if as_html=False)
        - HTML for a side-by-side diff view (if as_html=True, stand_alone=False)
        - Complete HTML document with the diff view (if as_html=True, stand_alone=True)

    Examples:
        # Get a text-based diff
        diff = diff_strings("Hello world", "Hello there world")

        # Get HTML for embedding in a notebook or web page
        html_diff = diff_strings("Hello world", "Hello there world", as_html=True)

        # Get a complete HTML document that can be saved to a file
        standalone_html = diff_strings("Hello world", "Hello there world",
                                     as_html=True, stand_alone=True)
        with open('diff.html', 'w') as f:
            f.write(standalone_html)
    """
    words1 = original_text.split()
    words2 = modified_text.split()
    if as_html:
        words1 = [html.escape(word) for word in words1]
        words2 = [html.escape(word) for word in words2]
    matcher = difflib.SequenceMatcher(None, words1, words2)
    opcodes = matcher.get_opcodes()
    if as_html:
        diff_content = ['<div style="display:flex; width:100%;">']
        diff_content.extend(
            [
                '<div style="flex:1; padding-right:10px;">',
                "<h3>Original</h3>",
                '<pre style="background-color:#f8f8f8; padding:10px; border-radius:5px;">',
            ]
        )
        diff_content.extend(_generate_html_diff_side(words1, opcodes, "left"))
        diff_content.extend(["</pre>", "</div>"])
        diff_content.extend(
            [
                '<div style="flex:1; padding-left:10px;">',
                "<h3>Modified</h3>",
                '<pre style="background-color:#f8f8f8; padding:10px; border-radius:5px;">',
            ]
        )
        diff_content.extend(_generate_html_diff_side(words2, opcodes, "right"))
        diff_content.extend(["</pre>", "</div>", "</div>"])
        diff_html = "\n".join(diff_content)
        if stand_alone:
            html_doc = [
                "<!DOCTYPE html>",
                "<html>",
                "<head>",
                "    <meta charset='utf-8'>",
                "    <meta name='viewport' content='width=device-width, initial-scale=1'>",
                "    <title>Text Difference</title>",
                "    <style>",
                "        body { font-family: Arial, sans-serif; padding: 20px; line-height: 1.4; }",
                "        h3 { margin-top: 0; }",
                "        pre { white-space: pre-wrap; word-wrap: break-word; margin: 0; }",
                "    </style>",
                "</head>",
                "<body>",
                diff_html,
                "</body>",
                "</html>",
            ]
            return "\n".join(html_doc)
        else:
            return diff_html
    else:
        result = []
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == "equal":
                result.append(" ".join(words1[i1:i2]))
            elif tag == "replace":
                result.append(
                    f"[-{' '.join(words1[i1:i2])}] [+{' '.join(words2[j1:j2])}]"
                )
            elif tag == "delete":
                result.append(f"[-{' '.join(words1[i1:i2])}]")
            elif tag == "insert":
                result.append(f"[+{' '.join(words2[j1:j2])}]")
        return " ".join(result)
