# flake8: noqa
from .version import __version__

from .bernstein import Bernstein
from .clock import Clock, GlobalClock
from .crypt import Crypt
from .dataframe_tools import slugify
from .ecdf import ecdf
from .fit import Fitter
from .gsheet import GSheet
from .hvtools import cc, hist, Animator, beta_plots
from .item import Item
from .iterify import iterify
from .memory import mem_show, mem_get
from .param_state import ParamState
from .plotting import figure, get_cc
from .postgres import PG
from .print_catcher import PrintCatcher
from .salesforce import SalesForceReport, Soql
from .shaper import Shaper
from .timer import Timer
import easier.hvtools as hv  # Need this weird import to make hv symbol work 

from .outlier_tools import (
    kill_outliers_iqr,
    kill_outliers_sigma_edit
)
from .utils import (
    cached_container,
    cached_dataframe,
    cached_property,
    mute_warnings,
    pickle_cache_mixin,
    pickle_cache_state,
    pickle_cached_container,
    print_error,
    django_reconnect,
    screen_width_full,
)

# alias for ease of remembering the name
warnings_mute = mute_warnings

# For backwards compatibility alias Fit to Fitter
class Fit(Fitter):
    def __init__(self, *args, **kwargs):
        import warnings
        warnings.warn('Fit is deprecated.  Use Fitter instead')
        super().__init__(*args, **kwargs)

# loads a holoviews color cycler as cc defaulting to None if not available
try:
    cc = get_cc()
except:  # noqa
    cc = None




