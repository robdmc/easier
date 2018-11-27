from copy import deepcopy
from typing import Union, Callable

import pandas as pd


class CachedLoader:
    """
    """
    CACHE = {}

    def __init__(
            self,
            uid: str,
            method_name_or_callable: Union[str, Callable],
            *args,
            **kwargs
    ):
        """
        Args:
            uid: the unique cache-key to associate with this data

            method_name_or_callable: use this to load the data

            *args: args to pass to the loading  method/callable

            **kwargs: kwargs to pass to the loading callable


        This is a descriptor that provides data loading/caching capability.  The
        main use case if for working in Jupyter notebooks where the same unchanged
        files or database queries must be performed over and over as you iterate
        your analysis.

        Here are examples of common use cases:


        class SimpleCSV:
            # Define the filename you want to load
            FILE_NAME = 'my_file.csv'

            # Use FILE_NAME as the uid in which to store data
            # read from FILE_NAME using pandas read_csv
            df = CachedLoader(FILE_NAME, pd.read_csv, FILE_NAME)

        class HugeCSV:
            # Define the filename you want to load
            FILE_NAME = 'my_file.csv'

            # Use FILE_NAME as the uid in which to store data
            # read from FILE_NAME using pandas read_csv
            df = CachedLoader(FILE_NAME, pd.read_csv, FILE_NAME)

            # By default, the cache will return a deepcopy of contents
            # For huge data, you may not want that.  This allows you
            # to simply return a reference to the cache
            df.set_copy(False)


        class ProcessedCSV:
            # Define the filename you want to load
            FILE_NAME = 'my_file.csv'

            # Use FILE_NAME as the uid in which to store data
            # read from FILE_NAME using pandas read_csv
            df = CachedLoader(FILE_NAME, 'load_csv_file', FILE_NAME)

            def load_csv_file(self, file_name):
                # Load and transform the data
                df = pd.read_csv(file_name)
                df = df.head(2)
                return df

        class CachedQuery:
            # Define a cached
            json_blob = CachedLoader(
                'json_blob',
                'my_db_query',
                min_time=parse('1/1/2017'),
                max_time=parse('1/31/2017')
            )

            def my_db_query(self, min_time, max_time):
                return run_my_query(min_time, max_time)


        class SimpleJson:
            # Define the filename you want to load
            FILE_NAME = 'my_file.json'

            # Use FILE_NAME as the uid in which to store data
            # read from FILE_NAME using pandas read_csv
            df = CachedLoader(FILE_NAME, 'load_json', FILE_NAME)

            def load_json(self, file_name):
                import json
                with open(file_name) as f:
                    blob = json.loads(f.read())
                return blob
        """

        self.uid = uid
        self.method_name_or_callable = method_name_or_callable
        self.args = args
        self.kwargs = kwargs
        self._copy = True

    def _get_loader(self, obj):
        if isinstance(self.method_name_or_callable, str):
            loader = getattr(obj, self.method_name_or_callable)
        elif hasattr(self.method_name_or_callable, '__call__'):
            loader = self.method_name_or_callable
        else:
            raise ValueError('Loader must be either a string or a callable')
        return loader

    def __get__(self, obj, objtype):
        if self.uid not in self.CACHE:
            data = self._get_loader(obj)(*self.args, **self.kwargs)
            self.CACHE[self.uid] = data
        else:
            data = self.CACHE[self.uid]

        # You probably want to return copies to prevent mutating the cache
        if self._copy:
            if isinstance(data, pd.DataFrame):
                data = data.copy()
            else:
                data = deepcopy(data)
        return data

    def set_copy(self, true_or_false):
        self._copy = true_or_false

    def __set__(self, obj, val):
        raise RuntimeError('Atrribute cannot be set')

    def __delete__(self, obj):
        if self.uid in self.__class__.CACHE:
            del self.__class__.CACHE[self.uid]

    @classmethod
    def clear(cls):
        cls.CACHE = {}
