import pandas as pd


class Duckmirror:
    def __init__(
        self, file_name: str, overwrite: bool = False, cache_in_memory: bool = True
    ):
        """
        Duckmirror is a class that provides a caching system for pandas dataframes.
        The file_name parameter is the name of the duckdb file to cache the dataframes to.
        The overwrite parameter is a boolean that indicates whether to overwrite the file if it already exists.
        The cache_in_memory parameter is a boolean that indicates whether to cache the dataframes in memory as
        well as the duckdb file.
        """

        def register(self, func):
            """
            Register is a decorator that registers a function as a dataframe.
            The function must return a pandas dataframe.
            The dataframe will be cached to the duckdb file.  The name of the table
            in the duckdb file will be the name of the function.

            Whenever a function is registered, DuckMirror will acquire an attribute
            named df_<function_name> that will return a copy of the cached dataframe.
            If intantiated with the cache_in_memory parameter set to True, then the source for the returned copy will be
            a dataframe held in the state of the class.  If instantiated with the cache_in_memory parameter set to False,
            then the source for the returned copy will be returned straight from the duckdb file with no copy necessary.

            # The only difference between decorating functions and classes in terms of the mirror attributes is that
            # the class name is prepended to the function name.  So... the attribute name will be df_<class_name>_<function_name>

            I want this decorator to work with both funtions and methods.  I want it to preserve the docstring of the decorated function.

            If decorating a function, the cached tables can live in a schema called "main".
            If decorating a method, I want the cached tables to live in a schema called the name of the class.

            """

        def sync(self, force=False):
            """
            In an ideal world, I would like the DuckMirror class to keep a hash of the code of the function/method it is decorating.
            If that hash has changed since the last time it was run, then, DuckMirror will delete the cached table in the database
            and run the changed function to create a new table.  Setting force to True will force the sync even if the hash has not changed.

            It is important that the hash creation/comparison is triggered when either the sync method is called or when the df_<function_name> attribute is accessed. Not at the time of registration.

            The state for the hashes can be stored in the duckdb file in a schema called "duckmirror".  DuckMirror will maintain
            a table named "duckmirror_state" with columns of schema, function_name, is_mirrored,, and hash.  It will use these to determine when resyncs are required.


            """


# This would be the basic usage of the Duckmirror class when used with functions.

# Instantiate the Duckmirror class..  Creates or overwrites the duckdb file appropriately.
mirror = Duckmirror(file_name, overwrite=False)


# Register two functions with the mirror with different names.
# Nothing actually happens in the database at decoration time
@mirror.register
def my_table_name1() -> pd.dataframe: ...


@mirror.register
def my_table_name2() -> pd.DataFrame: ...


# Sync the mirror to the duckdb file.  This will create or update all tables in the duckdb file.
# Calling sync should not be required to access the respective dataframe atttributes.
mirror.sync()

# Get the dataframes from the mirror.  This will return a copy of the dataframe from the duckdb file.
# If these are called before calling sync, then an implicit sync will be performed for the corresponding table only
mirror.df_my_table_name1
mirror.df_my_table_name2


# This would be basic usage for decorating methods

# Instantiate the Duckmirror class..  Creates or overwrites the duckdb file appropriately.
mirror = Duckmirror(file_name, overwrite=False)


# Defines the class
class MyClass:
    # Nothing actually happens in the database at decoration time
    @mirror.register
    def my_table_name3(self) -> pd.dataframe: ...

    @mirror.register
    def my_table_name4(self) -> pd.DataFrame: ...


# Instantiate the class.  Nothing happens in the database at this point.
my_object = MyClass()


# Sync the mirror to the duckdb file.  This will create or update all tables in the duckdb file.
# Calling sync should not be required to access the respective dataframe atttributes.
# Sync will fail if the class has not been instantiated.
mirror.sync()

# Get the dataframes from the mirror.  This will return a copy of the dataframe from the duckdb file.
# If these are called before calling sync, then an implicit sync will be performed for the corresponding table only
# This is not the standard way of accessing the dataframes for classes.  See below for the standard way.
mirror.df_my_table_name3
mirror.df_my_table_name4


# For method decorations, there is additional magic.  Accessing the decorated method will simply return the
# corresponding attribute of the mirror instance.

# This will sync the decorated method to the database if required and return the corresponding attribute of the mirror instance.
my_object.my_table_name3
my_object.my_table_name4
