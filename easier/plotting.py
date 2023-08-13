from itertools import repeat, chain


def figure(*args, grid=True, style="default", figsize=(9, 5), **kwargs):
    """
    Returns a matplotlib axis object.
    """
    import pylab as pl

    available = [s for s in pl.style.available + ["default"] if not s.startswith("_")]
    if style not in available:
        raise ValueError(f"\n\n Valid Styles are {available}")
    pl.style.use(style)
    pl.figure(*args, figsize=figsize, **kwargs)
    pl.grid(grid)
    ax = pl.gca()
    return ax


class ColorCyle:
    def __init__(self):
        from string import ascii_lowercase
        import holoviews as hv

        self.names, self.codes = zip(
            *(zip(ascii_lowercase, hv.Cycle().default_cycles["default_colors"]))
        )
        for name, code in zip(self.names, self.codes):
            setattr(self, name, code)

        self.reset()

    def reset(self):
        self._code_repeats = list(chain(*repeat(self.codes, 20)))
        self._code_iterator = iter(self._code_repeats)
        return self

    def __iter__(self):
        return self._code_iterator

    def __next__(self):
        return next(self._code_iterator)

    def __getitem__(self, index):
        return self._code_repeats[index]
