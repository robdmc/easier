class Importable:
    """
    This is a descriptor that knows how to load modules
    """

    def __init__(self, module_path, artifact_name, instantiate=False):
        self.instantiate = instantiate
        self._module_path = module_path
        self._artifact_name = artifact_name

    def __get__(self, instance, owner):
        # Import the module
        import importlib

        module = importlib.import_module(self._module_path)

        # Get the item (class/function/object) from the module
        thing = getattr(module, self._artifact_name)

        # If an object was requested, instantiate it
        if self.instantiate:
            thing = thing()
        return thing

    def __set__(self, instance, value):
        raise NotImplementedError("You cannot set this attribute")


class API:
    """
    Every object in the api is lazy loaded through a descriptor.  The actual
    attributes returned are classes/functions unless the instantiate flag is set.

    The lazy loading ensures that you only run imports if they are needed.
    """

    Animator = Importable("easier.hvtools", "Animator")
    Bernstein = Importable("easier.bernstein", "Bernstein")
    BernsteinFitter = Importable("easier.bernstein", "BernsteinFitter")
    BlobAttr = Importable("easier.utils", "BlobAttr")
    BlobMixin = Importable("easier.utils", "BlobMixin")
    Clock = Importable("easier.clock", "Clock")
    Crypt = Importable("easier.crypt", "Crypt")
    DistFitter = Importable("easier.distributions", "DistFitter")
    Duck = Importable("easier.duck", "Duck")
    DuckModel = Importable("easier.duck_model", "DuckModel")
    Elliptic = Importable("easier.filtering.elliptic_filter", "Elliptic")
    Fitter = Importable("easier.fit_lib", "Fitter")
    GlobalClock = Importable("easier.clock", "GlobalClock")
    Gsheet = Importable("easier.gsheet", "GSheet")
    Item = Importable("easier.item", "Item")
    MiniModel = Importable("easier.minimodel", "MiniModel")
    MiniModelPG = Importable("easier.minimodel", "MiniModelPG")
    MiniModelSqlite = Importable("easier.minimodel", "MiniModelSqlite")
    PG = Importable("easier.postgres", "PG")
    Parallel = Importable("easier.parallel", "Parallel")
    ParamState = Importable("easier.param_state", "ParamState")
    PrintCatcher = Importable("easier.print_catcher", "PrintCatcher")
    Proportion = Importable("easier.proportion_lib", "Proportion")
    SalesForceReport = Importable("easier.salesforce", "SalesForceReport")
    Scaler = Importable("easier.utils", "Scaler")
    Shaper = Importable("easier.shaper", "Shaper")
    Soql = Importable("easier.salesforce", "Soql")
    Timer = Importable("easier.timer", "Timer")
    beta_plots = Importable("easier.hvtools", "beta_plots")
    cached_container = Importable("easier.utils", "cached_container")
    cached_dataframe = Importable("easier.utils", "cached_dataframe")
    cached_property = Importable("easier.utils", "cached_property")
    cc = Importable("easier.plotting", "ColorCyle", instantiate=True)
    date_formatter = Importable("easier.nice_dates", "nice_dates")
    django_reconnect = Importable("easier.utils", "django_reconnect")
    ecdf = Importable("easier.ecdf_lib", "ecdf")
    events_from_starting_ending = Importable(
        "easier.dataframe_tools", "events_from_starting_ending"
    )
    figure = Importable("easier.plotting", "figure")
    get_logger = Importable("easier.utils", "get_logger")
    hist = Importable("easier.hvtools", "hist")
    ibis_conn_to_sqlalchemy_conn = Importable(
        "easier.ibis_tools", "ibis_conn_to_sqlalchemy_conn"
    )
    ibis_duck_connection = Importable("easier.ibis_tools", "ibis_duck_connection")
    ibis_get_sql = Importable("easier.ibis_tools", "get_sql")
    ibis_postgres_connection = Importable(
        "easier.ibis_tools", "ibis_postgres_connection"
    )
    ibis_sql_to_frame = Importable("easier.ibis_tools", "sql_to_frame")
    in_notebook = Importable("easier.utils", "in_notebook")
    iterify = Importable("easier.iterify", "iterify")
    kill_outliers_iqr = Importable("easier.outlier_tools", "kill_outliers_iqr")
    kill_outliers_sigma_edit = Importable(
        "easier.outlier_tools", "kill_outliers_sigma_edit"
    )
    localize_utc_to_timezone = Importable(
        "easier.dataframe_tools", "localize_utc_to_timezone"
    )
    mem_get = Importable("easier.memory", "mem_get")
    mem_show = Importable("easier.memory", "mem_show")
    month_string = Importable("easier.dataframe_tools", "month_string")
    mute_warnings = Importable("easier.utils", "mute_warnings")
    nice_dates = Importable("easier.nice_dates", "nice_dates")
    pandas_time_to_utc_seconds = Importable(
        "easier.dataframe_tools", "pandas_time_to_utc_seconds"
    )
    pandas_utc_seconds_to_time = Importable(
        "easier.dataframe_tools", "pandas_utc_seconds_to_time"
    )
    pg_creds_from_env = Importable("easier.postgres", "pg_creds_from_env")
    pickle_cache_mixin = Importable("easier.utils", "pickle_cache_mixin")
    pickle_cache_state = Importable("easier.utils", "pickle_cache_state")
    pickle_cached_container = Importable("easier.utils", "pickle_cached_container")
    print_error = Importable("easier.utils", "print_error")
    python_type = Importable("easier.utils", "python_type")
    screen_width_full = Importable("easier.utils", "screen_width_full")
    slugify = Importable("easier.dataframe_tools", "slugify")
    tqdm_flex = Importable("easier.utils", "tqdm_flex")
    tvd = Importable("easier.filtering.tvd", "tvd")
    warnings_mute = Importable("easier.utils", "mute_warnings")
    weekday_string = Importable("easier.dataframe_tools", "weekday_string")
