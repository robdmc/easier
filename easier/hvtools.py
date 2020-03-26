__all__ = ['hv_to_html', 'shade', 'hist', 'cc']

ALLOWED_REDUCTIONS = {
    'any',
    'count',
    'max',
    'mean',
    'min',
    'std',
    'sum',
}

# loads a holoviews color cycler as cc defaulting to None if not available
# See Defaults section of http://holoviews.org/user_guide/Styling_Plots.html for all available colors
try:
    def get_cc():
        from string import ascii_lowercase
        import holoviews
        cc = type('color', (), dict(zip(ascii_lowercase, holoviews.Cycle().default_cycles['default_colors'])))
        return cc
    cc = get_cc()
except:  # noqa
    cc = None


def hv_to_html(obj, file_name):
    """
    Save a holovies object to html
    :param obj: The holovies object
    :param file_name:  the file name to save. .html will get appended to name
    :return:
    """
    import holoviews as hv
    renderer = hv.renderer('bokeh')

    # Using renderer save
    renderer.save(obj, file_name)


def shade(hv_obj, reduction='any', color=None, spread=False):
    """
    Apply datashading to a holoviews object.

    hv_obj: a holovies object like Curve, Scatter, etc.
    reduction: Most common will be 'any' and 'count'.
               Supply any name here to see list of valid reductions
    color: Mostly used for 'any' aggregation to specify a color
    spread: Smear out points slightly bigger than 1 pixel for easier
            visibility
    """
    import datashader as ds
    from holoviews.operation.datashader import datashade, dynspread

    if reduction not in ALLOWED_REDUCTIONS:
        raise ValueError(
            'Allowed reductions are {}'.format(ALLOWED_REDUCTIONS))

    reducer = getattr(ds.reductions, reduction)

    kwargs = dict(aggregator=reducer())
    if color is None and reduction == 'any':
        kwargs.update(cmap=['blue'])
    else:
        kwargs.update(cmap=[color])

    obj = datashade(hv_obj, **kwargs)
    if spread:
        obj = dynspread(obj)
    return obj


def hist(x, logx=False, logy=False, label=None, color=None, **kwargs):
    """
    Creates holoviews histrogram object.
        logx=True: Create log spaced bins between min/max
        logy=True: Compute histogram of (1 + x) with y on db scale
        **kwargs are passed to numpy.histogram

    """
    # If logx was specified, create log spaced bins
    import numpy as np
    import holoviews as hv
    if logx:
        nbins = kwargs.get('bins', 10)
        if not isinstance(nbins, int):
            raise ValueError('Bins must be an integer when logx=True')

        range_vals = kwargs.get('range', None)
        if range_vals:
            minval, maxval = range_vals
        else:
            minval, maxval = np.min(x), np.max(x)

        bins = np.logspace(np.log10(minval), np.log10(maxval), nbins)
        kwargs.update(bins=bins)

    # Build up kwargs for the holoviews call
    hv_kwargs = {}
    if label is not None:
        hv_kwargs['label'] = label

    # If logy was specified, create a histogram of the db of (counts + 1)
    if logy:
        counts, edges = np.histogram(x, **kwargs)
        counts = 10 * np.log10(1 + counts)
        c = hv.Histogram((counts, edges), vdims='dB of (counts + 1)', **hv_kwargs)
    # If not logy, just to a histogram of counts
    else:
        c = hv.Histogram(np.histogram(x, **kwargs), **hv_kwargs)

    # Default the x axis to log if logx was specified
    if logx:
        c = c.options(logx=True)
    c = c.options(alpha=.3)
    if color is not None:
        c = c.options(color=color)
    return c


class Animator:
    """
    Creates animated plots with holoviews.

    Constructor Args:
        dynamic_range: bool = if set to True, auto-scale the chart each time it is drawn

    # ------------ Simple Example -------------------
    # Define a function to make your plots.
    # Data will be whatever object is passed to animator.send()
    def myplot(data):
        return hv.Curve(data)

    # Pass your plotting function to the animator constructor
    animator = Animator(myplot)

    # Simply send the animator updated versions of your data
    # to update the plot
    t0 = np.linspace(0, 6., 100)
    for delta in np.linspace(0, 3, 30):
        t = t0 - delta
        y = 2 + np.sin(t)
        animator.send((t, y))

    # ------------ Advanced Example -------------------
    def myplot(data):
        # First two elements of data are to be ploted
        # Third element of data is going to update a label
        c = hv.Curve(data[:2], 'a', 'b', label=f'hello {data[-1]}').options(color=ezr.cc.c, logy=True)
        c *= hv.Scatter((data[0], 2 * data[1])).options(color=ezr.cc.d)
        return c


    # Send data for animation
    for ind, delta in enumerate(np.linspace(0, 3, 300)):
        t = t0 - delta
        y = 2 + np.sin(t)
        # Can control when animations are drawn with this
        if ind % 1 == 0:
            animator.send((t, y))
    """
    def __init__(self, plot_func, dynamic_range=True, plot_every=1):
        import holoviews as hv
        from holoviews.streams import Pipe
        # Store the user-defined plot function
        self.plot_func = plot_func
        self.dynamic_range = dynamic_range
        self.pipe = Pipe(data=[])
        self.data_has_been_sent = False
        self.dmap = hv.DynamicMap(self.plot_wrapper, streams=[self.pipe])
        self.plot_every = plot_every
        self.plot_count = 0

    def plot_wrapper(self, *args, **kwargs):
        data = kwargs.get('data', ([0], [0]))
        hv_obj = self.plot_func(data)
        if self.dynamic_range:
            hv_obj = hv_obj.opts(norm=dict(framewise=True))
        return hv_obj

    def send(self, data):
        from IPython.display import display
        if self.plot_count % self.plot_every == 0:
            self.pipe.send(data)
            if not self.data_has_been_sent:
                display(self.dmap)
                self.data_has_been_sent = True
        self.plot_count += 1
