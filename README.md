# Easier
Easier is a tool-kit for creating shortcuts that are useful in data science
work. The toolkit is not really tailored to the needs of a wide audience as
it has evolved to fasciliate my (Rob deCarvalho) common workflows. That being
said, as long as you don't mind an opinionated take on common tasks, you may find some of the
tools useful.

## List of tools
* Timer: Context manager for timing sections of your code




```python
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
from .utils import ChattyDict, mute_warnings, cached_property, cached_dataframe

# alias for ease of remembering the name
outlier_iqr_killer = iqr_outlier_killer
warnings_mute = mute_warnings
DictChatty = ChattyDict

# loads a holoviews color cycler as cc defaulting to None if not available
try:
    cc = get_cc()
except:  # noqa
    cc = None
```