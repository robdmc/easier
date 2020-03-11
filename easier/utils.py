import copy
import os
import pickle
import sys
import traceback
import warnings


def mute_warnings():
    """
    Mute all Python warnings
    """
    import warnings
    warnings.filterwarnings("ignore")


def screen_width_full():
    from IPython.core.display import display, HTML
    display(HTML("<style>.container { width:100% !important; }</style>"))


def print_error(tag='', verbose=False, buffer=None):
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
        tag = f' :: {tag.strip()}'

    print(f'{exc_type.__name__}: {exc_value}{tag}', file=buffer)


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


class cached_container(object):
    """
    Decorator to cache containers in such a way that only copies are returned
    """
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, type=None):
        if instance is None:
            return self

        cached_var_name = '_cached_frame_for_' + self.func.__name__
        if cached_var_name not in instance.__dict__:
            instance.__dict__[cached_var_name] = self.func(instance)
        try:
            out = instance.__dict__[cached_var_name].copy()
        except AttributeError:
            out = copy.copy(instance.__dict__[cached_var_name])
        return out


class cached_dataframe(cached_container):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn('@cached_dataframe is deprecated.  @Use cached_container')


class pickle_cache_state:
    """
    This is a descriptor that stores optional state for pickle cache
    """
    def __init__(self, mode=None):
        allowed_modes = ['active', 'ignore', 'purge', 'refresh']
        if mode not in allowed_modes:
            raise ValueError(f'You must set mode to be one of {allowed_modes}')
        self.mode = mode

    def __get__(self, instance, owner):
        for att in owner.__dict__.values():
            if isinstance(att, pickle_cache_state):
                print('pickle_mute mode', att.mode)

    def __set__(self, instance, value):
        raise NotImplementedError('You cannot set this attribute')


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
        ##   'purge': deletes corresponding pickle file and ignores the cache
        ##   'refresh': Reccomputes and refreshes pickle file
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
        return self

    @property
    def default_pickle_file_name(self):
        return '/tmp/{}.pickle'.format(
            self.func.__qualname__.lower()
        )

    @property
    def pickle_file_name(self):
        if self._pickle_file_name:
            return self._pickle_file_name
        else:
            return self.default_pickle_file_name

    def __get__(self, instance, type=None):
        """
        After decoration the decorated method will be replaced with
        an instance of the decorator/descriptor.  Every time this instance
        is accessed this method will be called to return the value
        of the method, which has been turned into a pickle-backed property.
        """
        # This is extra logic that will check for pickle cache state
        for att in instance.__class__.__dict__.values():
            # If a cache state attribute was found on the class
            if isinstance(att, pickle_cache_state):
                # Get the mode from the state
                cache_mode = att.mode

                # If cache mode is refresh, then delete and recache
                if cache_mode == 'refresh':
                    self.__delete__(instance)

                # If cache mode is purge, then just delete and run callable
                elif cache_mode == 'purge':
                    self.__delete__(instance)
                    return self.func(instance)

                # If ignoring the cache, don't do anything to file, just return callable
                elif cache_mode == 'ignore':
                    return self.func(instance)

        # # This was part of the original django cached_property code.
        # # I don't think I need it. I'm going to leave it in here commented
        # # out though for reference.
        # if instance is None:
        #     return self

        # An attribute of this name will be placed on the object having
        # the decorated method
        cached_var_name = '_pickle_cache_for_' + self.func.__name__

        # Populate the in-memory cache if needed
        if cached_var_name not in instance.__dict__:
            instance.__dict__[cached_var_name] = self.get_pickled(instance)

        # If you don't need to return a copy, just return the cached object itself
        if not self.return_copy:
            return instance.__dict__[cached_var_name]

        # Otherwise return a copy of the cached object
        try:
            out = instance.__dict__[cached_var_name].copy()
        except AttributeError:
            out = copy.copy(instance.__dict__[cached_var_name])
        return out

    def get_pickled(self, instance):
        """
        This method either pulls data from pickle file if it exists
        otherwise it will populate the pickle file.
        """
        # If pickle file exists, load its contents
        if os.path.isfile(self.pickle_file_name):
            with open(self.pickle_file_name, 'rb') as buffer:
                obj = pickle.load(buffer)
        # If pickle file doesn't exist, evaluate the wrapped method
        # and save results to pickle file
        else:
            obj = self.func(instance)
            with open(self.pickle_file_name, 'wb') as buffer:
                pickle.dump(obj, buffer)

        return obj

    def __delete__(self, instance):
        """
        This method handles busting the cache.
        """
        # Delete the cached copy of the data
        cached_var_name = '_pickle_cache_for_' + self.func.__name__
        if cached_var_name in instance.__dict__:
            del instance.__dict__[cached_var_name]

        # Delete the pickle file
        if os.path.isfile(self.pickle_file_name):
            os.unlink(self.pickle_file_name)
