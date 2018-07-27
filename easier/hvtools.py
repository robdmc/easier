ALLOWED_REDUCTIONS = {
    'any',
    'count',
    'max',
    'mean',
    'min',
    'std',
    'sum',
}


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
