# flake8: noqa
from .version import __version__

from .timer import Timer
from .clock import Clock, GlobalClock
from .param_state import ParamState
from .item import Item
from .simple_cron import Cron
from .memory import mem_show, mem_get
import easier.hvtools as hv
from .hvtools import cc, hist
from .print_catcher import PrintCatcher
from .plotting import figure
from .fit import Fit
from .crypt import Crypt
from .ecdf import ecdf
from .dataframe_tools import (
    date_diff,
    slugify,
    iqr_outlier_killer,
    mute_warnings
)

# alias for ease of remembering the name
outlier_iqr_killer = iqr_outlier_killer
warnings_mute = mute_warnings

# loads a holoviews color cycler as cc defaulting to None if not available
try:
    def get_cc():
        from string import ascii_lowercase
        import holoviews as hv
        cc = type('color', (), dict(zip(ascii_lowercase, hv.Cycle().default_cycles['default_colors'])))
        return cc
    cc = get_cc()
except:  # noqa
    cc = None




