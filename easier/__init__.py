# flake8: noqa
from .version import __version__

from .timer import Timer
from .clock import Clock, GlobalClock
from .param_state import ParamState
from .item import Item
from .memory import mem_show, mem_get
import easier.hvtools as hv
from .hvtools import cc, hist, Animator
from .postgres import PG
from .print_catcher import PrintCatcher
from .plotting import figure, get_cc
from .fit import Fitter
from .crypt import Crypt
from .ecdf import ecdf
from .cached_loader import CachedLoader
from .shaper import Shaper
from .dataframe_tools import (
    date_diff,
    slugify,
)

from .outlier_tools import (
    kill_outliers_iqr,
    kill_outliers_sigma_edit
)
from .utils import (
    ChattyDict,
    mute_warnings,
    print_error,
    screen_width_full,
    cached_property,
    cached_dataframe,
    cached_container,
    pickle_cached_container,
    pickle_cache_state,
)

# alias for ease of remembering the name
warnings_mute = mute_warnings
DictChatty = ChattyDict

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




