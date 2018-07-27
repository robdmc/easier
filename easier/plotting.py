import pylab as pl


def figure(*args, grid=True, style='default', figsize=(9, 5), **kwargs):
    available = [s for s in pl.style.available + ['default'] if not s.startswith('_')]
    if style not in available:
        raise ValueError(f'\n\n Valid Styles are {available}')
    pl.style.use(style)
    pl.figure(*args, figsize=figsize, **kwargs)
    pl.grid(grid)
