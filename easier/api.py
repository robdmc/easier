import warnings


class Unsupported:
    def __init__(self, name):
        warnings.warn(f"{name} is not supported without installing additional packages")


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

        try:
            module = importlib.import_module(self._module_path)

            # Get the item (class/function/object) from the module
            thing = getattr(module, self._artifact_name)

            # If an object was requested, instantiate it
            if self.instantiate:
                thing = thing()
        except (ImportError, AttributeError, ModuleNotFoundError):
            thing = Unsupported(".".join([self._module_path, self._artifact_name]))

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
    DuckMirror = Importable("easier.duck_mirror", "DuckMirror")
    Elliptic = Importable("easier.filtering.elliptic_filter", "Elliptic")
    Fitter = Importable("easier.fit_lib", "Fitter")
    Gemini = Importable("easier.gemini", "Gemini")
    GlobalClock = Importable("easier.clock", "GlobalClock")
    Gsheet = Importable("easier.gsheet", "GSheet")
    Item = Importable("easier.item", "Item")
    Metabase = Importable("easier.metabase", "Metabase")
    MiniModel = Importable("easier.minimodel", "MiniModel")
    MiniModelPG = Importable("easier.minimodel", "MiniModelPG")
    MiniModelSqlite = Importable("easier.minimodel", "MiniModelSqlite")
    OrderedSchema = Importable(
        "easier.ibis_tools", "get_order_schema_class", instantiate=True
    )
    PG = Importable("easier.postgres", "PG")
    Parallel = Importable("easier.parallel", "Parallel")
    ParamState = Importable("easier.param_state", "ParamState")
    PrintCatcher = Importable("easier.print_catcher", "PrintCatcher")
    Proportion = Importable("easier.proportion_lib", "Proportion")
    QuickSchema = Importable(
        "easier.dataframe_tools", "get_quick_schema_class", instantiate=True
    )
    PandasSql = Importable(
        "easier.dataframe_tools", "get_pandas_sql_class", instantiate=True
    )
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
    classifier_evaluation_plots = Importable(
        "easier.fit_lib", "classifier_evaluation_plots"
    )
    column_level_flattener = Importable(
        "easier.dataframe_tools", "column_level_flattener"
    )
    date_formatter = Importable("easier.nice_dates", "nice_dates")
    diff_strings = Importable("easier.utils", "diff_strings")
    django_reconnect = Importable("easier.utils", "django_reconnect")
    ecdf = Importable("easier.ecdf_lib", "ecdf")
    events_from_starting_ending = Importable(
        "easier.dataframe_tools", "events_from_starting_ending"
    )
    figure = Importable("easier.plotting", "figure")
    get_logger = Importable("easier.utils", "get_logger")
    heatmap = Importable("easier.dataframe_tools", "heatmap")
    hex_from_dataframe = Importable("easier.dataframe_tools", "hex_from_dataframe")
    hex_to_dataframe = Importable("easier.dataframe_tools", "hex_to_dataframe")
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
    NormalSampleJoiner = Importable("easier.normal_sampler", "NormalSampleJoiner")
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
    sql_file_to_df = Importable("easier.postgres", "sql_file_to_df")
    sql_string_to_df = Importable("easier.postgres", "sql_string_to_df")
    StreamPlotter = Importable("easier.stream_plotter", "StreamPlotter")
    tqdm_flex = Importable("easier.utils", "tqdm_flex")
    Tracker = Importable("easier.tracker", "Tracker")
    tvd = Importable("easier.filtering.tvd", "tvd")
    VonMisesFitter = Importable("easier.vonmises", "VonMisesFitter")
    warnings_mute = Importable("easier.utils", "mute_warnings")
    weekday_string = Importable("easier.dataframe_tools", "weekday_string")
