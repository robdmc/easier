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
from .postgres import PG
from .print_catcher import PrintCatcher
from .plotting import figure, get_cc
from .fit import Fit
from .crypt import Crypt
from .ecdf import ecdf
from .cached_loader import CachedLoader
from .dataframe_tools import (
    date_diff,
    slugify,
    iqr_outlier_killer,
)
from .utils import ChattyDict, mute_warnings, cached_property

# alias for ease of remembering the name
outlier_iqr_killer = iqr_outlier_killer
warnings_mute = mute_warnings
DictChatty = ChattyDict

# loads a holoviews color cycler as cc defaulting to None if not available
try:
    cc = get_cc()
except:  # noqa
    cc = None




