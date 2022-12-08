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
        raise NotImplementedError('You cannot set this attribute')


class API:
    """
    Every object in the api is lazy loaded through a descriptor.  The actual
    attributes returned are classes/functions unless the instantiate flag is set.

    The lazy loading ensures that you only run imports if they are needed.
    """
    BernsteinFitter = Importable('easier.bernstein', 'BernsteinFitter')
    Bernstein = Importable('easier.bernstein', 'Bernstein')
    Clock = Importable('easier.clock', 'Clock')
    GlobalClock = Importable('easier.clock', 'GlobalClock')
    Crypt = Importable('easier.crypt', 'Crypt')
    slugify = Importable('easier.dataframe_tools', 'slugify')
    pandas_utc_seconds_to_time = Importable('easier.dataframe_tools', 'pandas_utc_seconds_to_time')
    pandas_time_to_utc_seconds = Importable('easier.dataframe_tools', 'pandas_time_to_utc_seconds')
    DistFitter = Importable('easier.distributions', 'DistFitter')
    ecdf = Importable('easier.ecdf_lib', 'ecdf')
    Fitter = Importable('easier.fit_lib', 'Fitter')
    Gsheet = Importable('easier.gsheet', 'GSheet')
    hist = Importable('easier.hvtools', 'hist')
    Animator = Importable('easier.hvtools', 'Animator')
    beta_plots = Importable('easier.hvtools', 'beta_plots')
    in_notebook = Importable('easier.utils', 'in_notebook')
    Item = Importable('easier.item', 'Item')
    iterify = Importable('easier.iterify', 'iterify')
    mem_show = Importable('easier.memory', 'mem_show')
    mem_get = Importable('easier.memory', 'mem_get')
    Parallel = Importable('easier.parallel', 'Parallel')
    ParamState = Importable('easier.param_state', 'ParamState')
    figure = Importable('easier.plotting', 'figure')
    PG = Importable('easier.postgres', 'PG')
    pg_creds_from_env = Importable('easier.postgres', 'pg_creds_from_env')
    PrintCatcher = Importable('easier.print_catcher', 'PrintCatcher')
    SalesForceReport = Importable('easier.salesforce', 'SalesForceReport')
    Soql = Importable('easier.salesforce', 'Soql')
    Shaper = Importable('easier.shaper', 'Shaper')
    Timer = Importable('easier.timer', 'Timer')
    Elliptic = Importable('easier.filtering.elliptic_filter', 'Elliptic')
    tvd = Importable('easier.filtering.tvd', 'tvd')
    nice_dates = Importable('easier.nice_dates', 'nice_dates')
    date_formatter = Importable('easier.nice_dates', 'nice_dates')
    Duck = Importable('easier.duck', 'Duck')
    DuckModel = Importable('easier.duck_model', 'DuckModel')
    MiniModel = Importable('easier.minimodel', 'MiniModel')
    MiniModelSqlite = Importable('easier.minimodel', 'MiniModelSqlite')
    MiniModelPG = Importable('easier.minimodel', 'MiniModelPG')
    kill_outliers_iqr = Importable('easier.outlier_tools', 'kill_outliers_iqr')
    kill_outliers_sigma_edit = Importable('easier.outlier_tools', 'kill_outliers_sigma_edit')
    cached_container = Importable('easier.utils', 'cached_container')
    cached_dataframe = Importable('easier.utils', 'cached_dataframe')
    cached_property = Importable('easier.utils', 'cached_property')
    mute_warnings = Importable('easier.utils', 'mute_warnings')
    warnings_mute = Importable('easier.utils', 'mute_warnings')
    pickle_cache_mixin = Importable('easier.utils', 'pickle_cache_mixin')
    pickle_cache_state = Importable('easier.utils', 'pickle_cache_state')
    pickle_cached_container = Importable('easier.utils', 'pickle_cached_container')
    print_error = Importable('easier.utils', 'print_error')
    python_type = Importable('easier.utils', 'python_type')
    django_reconnect = Importable('easier.utils', 'django_reconnect')
    screen_width_full = Importable('easier.utils', 'screen_width_full')
    BlobMixin = Importable('easier.utils', 'BlobMixin')
    BlobAttr = Importable('easier.utils', 'BlobAttr')
    Scaler = Importable('easier.utils', 'Scaler')
    get_logger = Importable('easier.utils', 'get_logger')
    cc = Importable('easier.plotting', 'ColorCyle', instantiate=True)
    Proportion = Importable('easier.proportion_lib', 'Proportion')
    tqdm_flex = Importable('easier.utils', 'tqdm_flex')
