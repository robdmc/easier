# Easier
Easier is a tool-kit for creating shortcuts that are useful in data science
work. The toolkit is not really tailored to the needs of a wide audience as
it has evolved to fasciliate my (Rob deCarvalho) common workflows. That being
said, as long as you don't mind an opinionated take on common tasks, you may find some of the
tools useful.

## Timer
A context manager for timing sections of code.

* **Args**:
   * **name**: The name you want to give the contextified code
   * **silent**: Setting this to true will mute all printing
   * **pretty**: When set to true, prints elapsed time in hh:mm:ss.mmmmmm

```python
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

# Initialize a list to hold results
results = []

# Run a piece of code with different values of the var you want to scale
for nn in range(3):
    # time each iteration
    with Timer('loop {}'.format(nn + 1), silent=True) as timer:
        time.sleep(.1 * nn)
    # add results
    results.append((nn, timer))

# Print csv compatible text for further pandashells processing/plotting
print 'nn,seconds'
for nn, timer in results:
    print '{},{}'.format(nn,timer.seconds)
```

## Clock
A clock that enables you to measure different parts of your code like a stopwatch.
There are two versions.  GlobalClock and Clock.  They are idential except that GlobalClock
stores clocks globally on the class whereas Clock stores them locally on the instance.

```python
# ---------------------------------------------------------------------------
# Example code for explicitly starting and stopping the clock
import easier as ezr

# Intantiate a clock
clock = ezr.Clock()


for nn in range(10):
    # Time different parts of your code
    clock.start('outer', 'inner')
    time.sleep(.1)
    clock.stop('outer')
    time.sleep(.05)
    clock.start('outer')

clock.stop()
print(clock)

# ---------------------------------------------------------------------------
# Example code for timing with context managers
import easier as ezr

# Intantiate a clock
clock = ezr.Clock()

for nn in range(10):
    with clock.running('outer', 'inner'):
        time.sleep(.1)
        with clock.paused('outer'):
            time.sleep(.05)
print(clock)

```

# scratch pad from here on down.  work in progress
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