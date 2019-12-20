# Easier  (This readme still a WIP.  Does not fully reflect tools)
Easier is a rather eclectic set of tools that I (Rob deCarvalho) have developed to minimize
boilerplate code in my Jupyter Notebook analysis work. I am an old-school matplotlib user
who has recently become an enthusiastic user of the Holoviews project for visualizations.
Pretty much all the plotting stuff in this project relies on holoviews.  If that's not your
thing, then maybe you should make it your thing, because Holoviews is amazing!

Although I do think the tools are widely useful, I will evolve this project to suit the needs
of my everyday work.  As such, the tool selection and implementation may be a bit opinionated
and the API subject to change without notice.

If you can stomach these disclaimers, you may find, as I do, that these tools significantly
improve your workflow efficiency.

The documentation for the tools is all in this README in the form of examples.  I have tried
to put reasonable docstrings in functions/methods to detail additional features they contain,
but that is a work in progress.

Enjoy.

# Tool Directory
- **Optimization tools**
    - [Fitter](#fitter)  A curve fitting tool
    - [ParamState](#paramstate) A class for managing optimization parameters

- **System Tools**
    - [Timer](#timer) Time sections of your code
    - [Clock](#clock) A stopwatch for your code (good for timing logic inside of loops)
    - [Memory](#memory) A tool for monitoring memory usage

- **Plotting Tools**
    - [ColorCycle](#colorcyle) A convenience tool for colorcycles
    - [Figure](#figure) A tool for generating nice matplotlib axes
    - [Histogram](#histogram) Creates a holoviews histogram plot

- **Programming Tools**
    - [Cached Property](#cached-property) Copy-paste of Django cached_property
    - [Cached Dataframe](#cached-dataframe) A cached property for pandas dataframes
    - [Crypt](#crypt) Enable password encrypting/decrypting of strings
    - [Chatty Dict](#chatty-dict) A dict subclass that dumps existing keys on KeyError

- **Data Tools**
    - [Item](#item) A generic data class with both dictionary and attribute access
    - [Slugify](#slugify) Turns list of strings into lists of slugs.  (think dataframe column names)
    - [Postgres Tool](#postgres) Makes querying postgres into dataframes easy

- **Stats tools**
    - [IQR Outlier detection](#outlier-iqr-killer)  Sets outliers to NaN using IQR detection
    - [ECDF](#ecdf) Computes the emperical distribution function


## Timer
A context manager for timing sections of code.

* **Args**:
   * **name**: The name you want to give the contextified v
   * **silent**: Setting this to true will mute all v
   * **pretty**: When set to true, prints elapsed time in hh:mm:ss.s

```python
# ---------------------------------------------------------------------------
# Example code for timing different parts of your code
import time
import easier as ezr
with ezr.Timer('entire script'):
    for nn in range(3):
        with ezr.Timer('loop {}'.format(nn + 1)):
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
import easier as ezr

# Initialize a list to hold results
results = []

# Run a piece of code with different values of the var you want to scale
for nn in range(3):
    # time each iteration
    with ezr.Timer('loop {}'.format(nn + 1), silent=True) as timer:
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

## ParamState
This class is intented to simplify working with the scipy optimize libraries.  In those
libraries, the parameters are always expressed as numpy arrays.  It's always kind of a 
pain to translate you parameters into variable names that have meaning within the loss function.
The ParamState class was written to ease this pain.

You instantiate a ParamState object by defining the variables of your problem.

```python
# Create a param_state object
p = ezr.ParamState(
    # Define vars a and b to use in your problem
    # (initialized to a default of 1)
    'a',
    'b',
    'c',

    # Define a variable with explicit initialization
    d=10
)

# Add givens to the ParamState.  These will remain fixed in a way that makes
# it easy for the optimizer functions to ignore them.

p.given(
    a=7,
    x_data=[1, 2, 3],
    y_date = [4, 5, 6]
)
print(p)
```
When printed, an asterisk is placed after the "given" variables
```
              val const
b               1
c               1
d              10
a               7     *
x_data  [1, 2, 3]     *
y_date  [4, 5, 6]     *
```
The values for your variables are accessed with their correspondingly named attributes
on the ParamState object.

At any point, an array of variables can be accessed by accessing `.array` attribute. The elements
of this array will contain only the non "fixed" variables of your problem.  This is the array you will supply to the scipy optimization functions.
```python
print(p.array)
```
```
[ 1.  1. 10.]
```

The values of the variables can be updated from an array by using the `.ingest()` method

```python
import numpy as np
p.ingest(np.array([10, 20, 30]))
print(p)
```
```
              val const
b              10
c              20
d              30
a               7     *
x_data  [1, 2, 3]     *
y_date  [4, 5, 6]     *
```

Here is a complete example of using ParamState with the fmin function from scipy
```python
# Do imports
import numpy as np
from scipy.optimize import fmin
from easier import ParamState

# Define a model that gives response values in terms of params
def model(p):
    return p.a * p.x_train ** p.n

# Define a cost function for the optimizer to minimize
def cost(args, p):
    '''
    args: a numpy array of parameters that scipy optimizer passes in
    p: a ParamState object
    '''

    # Update paramstate with the latest values from the optimizer
    p.ingest(args)

    # Use the paramstate to generate a "fit" based on current params
    y_fit = model(p)

    # Compute the errors
    err = y_fit - p.y_train

    # Compute and return the cost
    cost = np.sum(err ** 2)
    return cost

# Make some fake data
x_train = np.linspace(0, 10, 100)
y_train = -7 * x_train ** 2
y_train = y_train + .5 * np.random.randn(len(x_train))


# Create a paramstate with variable names
p = ParamState('a n')

# Specify the data you are fitting
p.given(
    x_train=x_train,
    y_train=y_train
)


# Get the initial values for params
x0 = p.array

# Run the minimizer to get the optimal params
xf = fmin(cost, x0, args=(p,))

# Update ParamState with optimal params
p.ingest(xf)

# Print the optimized results
print(p)
```

## Item
This is a really simple container class that is kind of dumb, but convenient.
It supports both object and dictionary access to its attributes. So, for
example, all of the following statements are supported.
```python
item = Item(a=1, b=2)
item['c'] = 2
item.d = 7
a = item['a']
b = item.b
item_dict = item.as_dict()
```


# Fitter
The fitter class enables a convenient api for curve fitting.  It is just a wrapper around
the various scipy optimization libaries.

## Simple Curve Fitting Example
```python
# Make data from noise-corrupted sinusoid
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x) - .7 * np.cos(x) + .1 * np.random.randn(len(x))


# Define a model function you want to fit to
# All model parameters are on the p object.
# The names "x", and "y" are reserved for the data you are fitting
def model(p):
    return p.a * np.sin(p.k * p.x) + p.b * np.cos(p.k * p.x)

# Initialize a fitter with purposefully bad guesses
fitter = ezr.Fitter(a=-1, b=2, k=.2)

# Fit the data and plot fit quality every 5 iterations
fitter.fit(x=x, y=y, model=model, plot_every=5)

# Plot the final results
display(fitter.plot())
display(fitter.params.df)
```


## Advanced Curve Fitting Example
```python
# Make data from noise-corrupted sinusoid
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x) - .7 * np.cos(x) + .1 * np.random.randn(len(x))

# Define a model function you want to fit to
# All model parameters are on the p object.
# The names "x", and "y" are reserved for the data you are fitting
def model(p):
    return p.a * np.sin(p.k * p.x) + p.b * np.cos(p.k * p.x)

# Initialize a fitter with purposefully bad guesses
fitter = ezr.Fitter(a=-1, b=2, k=.2)

# Fit the data and plot fit quality every 5 iterations
fitter.fit(
    x=x,                   # The independent data
    y=y,                   # The dependent data
    model=model,           # The model function
    plot_every=5,          # Plot fit every this number of iterations
    algorithm='fmin_bfgs', # Scipy optimization routine to use
    verbose=False          # Don't print convergence info
)

# Get predictions at specific values
x_predict = np.linspace(0, 6 * np.pi, 300)
y_predict = fitter.predict(x_predict)

# Get the components of the fit chart
components = fitter.plot(
    x=x_predict,
    scale_factor=10,
    label='10X Scaled Fit',
    line_color='red',
    scatter_color='blue',
    size=15,
    xlabel='My X Label',
    ylabel='My Y Label',
    as_components=True,
)

# Display the components as a layout rather than overlay
display(hv.Layout(components))
```

## Postgres
This tool is a straightforward wrapper that provides a convenient API for running queries against
a postgres database.  Credentials can either be passed into the constructor or read from the
standard psql environment variables.

### Simple query example
```python
import easier as ezr

# Query a database whos credentials are given by the environment variables:
# PGHOST PGUSER PGPASSWORD PGDATABASE
df = ezr.PG().query(
    'SELECT email, first_name, last_name FROM users LIMIT 5'
).to_dataframe()

# Run the same query, but manually provide credentials
df = ezr.PG(
    host='MY_HOST',
    user='MY_USER',
    password='MY_PASSWORD',
    dbname='MY_DATABASE',
).query(
    'SELECT email, first_name, last_name FROM users LIMIT 5'
).to_dataframe()
```

### Advanced Example
The PG class leverages the excellent [JinjaSQL](https://github.com/hashedin/jinjasql) library
to enable creating dynamic queries based on variables in your code.  See the
[JinjaSQL README](https://github.com/hashedin/jinjasql) file for documentation on how to
use templating features.  An example is shown here

```python
# Intantiate the postgres object
pg = ezr.PG()

# Specify the query
pg.query(
    # Write a query with template placeholders for dynamic variables
    """
        SELECT
            email, first_name, last_name 
        FROM 
            {{ table_name | sqlsafe }}
        WHERE 
            {{field_name | sqlsafe}} IN {{my_values | inclause}}
        LIMIT
            {{limit}}; 
    """,

    # Specify the values the templated variables should take            
    table_name='prod_py.users',
    field_name='first_name',
    my_values=['Matthew', 'Tyler'],
    limit=4
)

# Fully rendered query. Ready for pasting to REPL
print(pg.sql)

# Save results to tuples
tups = pg.to_tuples()

# Save results to named tuples
named_tups = pg.to_named_tuples()

# Save result to list of dicts
dicts = pg.to_dicts()

# Save query results to a Pandas dataframe
df = pg.to_dataframe()
```