__version__ = "1.9.1"


from .hvtools import beta_plots, hist
from .bernstein import Bernstein, BernsteinFitter
from .utils import (
    BlobAttr,
    BlobMixin,
    cached_container,
    cached_dataframe,
    cached_property,
    diff_strings,
    get_logger,
    mute_warnings,
    pickle_cache_mixin,
    pickle_cache_state,
    pickle_cached_container,
    print_error,
    Scaler,
)
from .clock import Clock, GlobalClock
from .crypt import Crypt
from .distributions import DistFitter
from .duckcacher import DuckCacher, duckloader_factory
from .duckcacher import duckloader_factory as get_duckloader  # This is an alias
from .duck_mirror import DuckMirror
from .filtering.elliptic_filter import Elliptic
from .fit_lib import Fitter, classifier_evaluation_plots
from .gemini import Gemini
from .gsheet import GSheet as Gsheet
from .item import Item
from .metabase import Metabase
from .postgres import PG, pg_creds_from_env, sql_file_to_df, sql_string_to_df
from .param_state import ParamState
from .print_catcher import PrintCatcher
from .proportion_lib import Proportion
from .shaper import Shaper
from .timer import Timer
from .plotting import figure
from .ecdf_lib import ecdf
from .dataframe_tools import (
    column_level_flattener,
    events_from_starting_ending,
    heatmap,
    hex_from_dataframe,
    hex_from_duckdb,
    hex_to_dataframe,
    hex_to_duckdb,
    localize_utc_to_timezone,
    month_string,
    pandas_time_to_utc_seconds,
    pandas_utc_seconds_to_time,
    slugify,
    weekday_string,
)
from .iterify import iterify
from .outlier_tools import kill_outliers_iqr, kill_outliers_sigma_edit
from .lomb_scargle import lomb_scargle
from .memory import mem_get, mem_show
from .normal_sampler import NormalSampleJoiner
from .stream_plotter import StreamPlotter
from .tracker import Tracker
from .filtering.tvd import tvd
from .vonmises import VonMisesFitter
from .ez_agent import EZAgent

from .dataframe_tools import get_quick_schema_class, get_pandas_sql_class
from .plotting import ColorCyle

QuickSchema = get_quick_schema_class()
PandasSql = get_pandas_sql_class()

cc = ColorCyle()

# Aliases for backwards compatibility
warnings_mute = mute_warnings

# Export list
__all__ = [
    "Bernstein",
    "BernsteinFitter",
    "BlobAttr",
    "BlobMixin",
    "Clock",
    "Crypt",
    "DistFitter",
    "DuckCacher",
    "DuckMirror",
    "duckloader_factory",
    "Elliptic",
    "EZAgent",
    "Fitter",
    "Gemini",
    "GlobalClock",
    "Gsheet",
    "Item",
    "Metabase",
    "PG",
    "ParamState",
    "PrintCatcher",
    "Proportion",
    "QuickSchema",
    "PandasSql",
    "Scaler",
    "Shaper",
    "Timer",
    "beta_plots",
    "cached_container",
    "cached_dataframe",
    "cached_property",
    "cc",
    "classifier_evaluation_plots",
    "column_level_flattener",
    "diff_strings",
    "ecdf",
    "events_from_starting_ending",
    "figure",
    "get_duckloader",
    "get_logger",
    "heatmap",
    "hex_from_dataframe",
    "hex_from_duckdb",
    "hex_to_dataframe",
    "hex_to_duckdb",
    "hist",
    "iterify",
    "kill_outliers_iqr",
    "kill_outliers_sigma_edit",
    "localize_utc_to_timezone",
    "lomb_scargle",
    "mem_get",
    "mem_show",
    "month_string",
    "mute_warnings",
    "NormalSampleJoiner",
    "pandas_time_to_utc_seconds",
    "pandas_utc_seconds_to_time",
    "pg_creds_from_env",
    "pickle_cache_mixin",
    "pickle_cache_state",
    "pickle_cached_container",
    "print_error",
    "slugify",
    "sql_file_to_df",
    "sql_string_to_df",
    "StreamPlotter",
    "Tracker",
    "tvd",
    "VonMisesFitter",
    "warnings_mute",
    "weekday_string",
]
