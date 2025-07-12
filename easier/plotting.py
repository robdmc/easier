from itertools import repeat, chain
from string import ascii_lowercase


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
        self._holoviews_initialized = False
        self._current_iterator = None

    def _initialize_holoviews(self):
        if not self._holoviews_initialized:
            import holoviews as hv

            hv.extension("bokeh")  # type: ignore
            self._names, self._codes = zip(*zip(ascii_lowercase, hv.Cycle().default_cycles["default_colors"]))
            for name, code in zip(self._names, self._codes):
                setattr(self, name, code)
            self._holoviews_initialized = True

    @property
    def names(self):
        self._initialize_holoviews()
        return self._names

    @property
    def codes(self):
        self._initialize_holoviews()
        return self._codes

    def reset(self):
        if hasattr(self, "codes"):
            self._code_repeats = list(chain(*repeat(self.codes, 20)))
            self._current_iterator = iter(self._code_repeats)
        return self

    @property
    def _code_iterator(self):
        if not hasattr(self, "_code_repeats"):
            self.reset()
        return self._current_iterator

    def __iter__(self):
        return self._code_iterator

    def __next__(self):
        if not hasattr(self, "_code_repeats"):
            self.reset()
        if self._current_iterator is None:
            self._current_iterator = iter(self._code_repeats)
        return next(self._current_iterator)

    def __getitem__(self, index):
        if not hasattr(self, "_code_repeats"):
            self.reset()
        return self._code_repeats[index]
