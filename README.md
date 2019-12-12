# Easier
Easier is a tool-kit for creating shortcuts that are useful in data science
work. The toolkit is not really tailored to the needs of a wide audience as
it has evolved to fasciliate my (Rob deCarvalho) common workflows. That being
said, as long as you don't mind an opinionated take on common tasks, you may find some of the
tools useful.

## List of tools
### Timer
A context manager for timing sections of code.

* Args:
   * **name**: The name you want to give the contextified code
   * **silent**: Setting this to true will mute all printing
   * **pretty**: When set to true, prints elapsed time in hh:mm:ss.mmmmmm

```python
# Example
# ---------------------------------------------------------------------------
# Example code for timing different parts of your code
import time
from pandashells import Timer
with Timer('entire script'):
    for nn in range(3):
        with Timer('loop {}'.format(nn + 1)):
            time.sleep(.1 * nn)
# Will generate the following output on stdout
#     col1: a string that is easily found with grep
#     col2: the time in seconds (or in hh:mm:ss if pretty=True)
#     col3: the value passed to the 'name' argument of Timer

__time__,2.6e-05,loop 1
__time__,0.105134,loop 2
__time__,0.204489,loop 3
__time__,0.310102,entire script

# ---------------------------------------------------------------------------
# Example for measuring how a piece of of code scales (measuring "big-O")
import time
from pandashells import Timer

# initialize a list to hold results
results = []

# run a piece of code with different values of the var you want to scale
for nn in range(3):
    # time each iteration
    with Timer('loop {}'.format(nn + 1), silent=True) as timer:
        time.sleep(.1 * nn)
    # add results
    results.append((nn, timer))

# print csv compatible text for further pandashells processing/plotting
print 'nn,seconds'
for nn, timer in results:
    print '{},{}'.format(nn,timer.seconds)
```



```python
from .timer import Timer
from .clock import Clock, GlobalClock
from .param_state import ParamState
from .item import Item
from .simple_cron import s
from .memory import mem_show, mem_get
import easier.hvtools as hv
from .hvtools import cc, hist
from .postgres import PG
from .print_catcher import v
from .plotting import figure, vc
from .fit import Fit
from .crypt import Crypt
from .ecdf import ecdf
from .cached_loader import CachedLoader
from .dataframe_tools import (
    date_diff,
    slugifyv
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