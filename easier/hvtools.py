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


def hist(x, logx=False, logy=False, **kwargs):
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

    # If logy was specified, create a histogram of the db of (counts + 1)
    if logy:
        counts, edges = np.histogram(x, **kwargs)
        counts = 10 * np.log10(1 + counts)
        c = hv.Histogram((counts, edges), vdims='dB of (counts + 1)')
    # If not logy, just to a histogram of counts
    else:
        c = hv.Histogram(np.histogram(x, **kwargs))

    # Default the x axis to log if logx was specified
    if logx:
        c = c.options(logx=True)
    return c
