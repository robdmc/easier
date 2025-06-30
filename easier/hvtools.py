from scipy.stats import beta
from string import ascii_lowercase
import holoviews
import holoviews as hv
import numpy as np

__all__ = ["hv_to_html", "hist", "cc"]
from typing import Iterable

try:

    def get_cc():
        cc = type(
            "color",
            (),
            dict(
                zip(ascii_lowercase, holoviews.Cycle().default_cycles["default_colors"])
            ),
        )
        return cc

    cc = get_cc()
except Exception:
    cc = None


def hv_to_html(obj, file_name):
    """
    Save a holovies object to html
    :param obj: The holovies object
    :param file_name:  the file name to save. .html will get appended to name
    :return:
    """
    renderer = hv.renderer("bokeh")
    renderer.save(obj, file_name)


def hist(x, logx=False, logy=False, label=None, color=None, **kwargs):
    """
    Creates holoviews histrogram object.
        logx=True: Create log spaced bins between min/max
        logy=True: Compute histogram of (1 + x) with y on db scale
        **kwargs are passed to numpy.histogram

    """
    if logx:
        nbins = kwargs.get("bins", 10)
        if not isinstance(nbins, int):
            raise ValueError("Bins must be an integer when logx=True")
        range_vals = kwargs.get("range", None)
        if range_vals:
            minval, maxval = range_vals
        else:
            minval, maxval = (np.min(x), np.max(x))
        bins = np.logspace(np.log10(minval), np.log10(maxval), nbins)
        kwargs.update(bins=bins)
    hv_kwargs = {}
    if label is not None:
        hv_kwargs["label"] = label
    if logy:
        counts, edges = np.histogram(x, **kwargs)
        counts = 10 * np.log10(1 + counts)
        c = hv.Histogram((counts, edges), vdims="dB of (counts + 1)", **hv_kwargs)
    else:
        c = hv.Histogram(np.histogram(x, **kwargs), **hv_kwargs)
    if logx:
        c = c.options(logx=True)
    c = c.options(alpha=0.3)
    if color is not None:
        c = c.options(color=color)
    return c


def beta_plots(
    wins: Iterable,
    losses: Iterable,
    labels: Iterable,
    legend_position="right",
    alpha=0.5,
    normed=False,
    xlabel=None,
    ylabel=None,
):
    """
    Make beta plots for win/loss type scenarios.  The wins/losses are provided in arrays.
    Each element of the array corresponds to a specific win/loss scenario you want plotted.

    So, to just determine the beta distirbution of a single scenario, these should be
    one-element arrays.

    Args:
        wins: the number of "wins"
        losses: the number of "losses"
        labels: The label for each scenario
        legend_postion: Where to put the legend
        alpha: the opacity of the area plots
        xlabel: The x label [default "Win Perdentage"]
        ylabel: The y label [default, "Density"]
    """
    if xlabel is None:
        xlabel = "Win Percentage"
    if ylabel is None:
        ylabel = "Density"
    c_list = []
    x = np.linspace(0, 1, 500)
    xpoints = []
    ypoints = []
    for won, lost, label in zip(wins, losses, labels):
        dist = beta(won + 1, lost + 1)
        y = dist.pdf(x)
        if normed:
            y_max = np.max(y)
        else:
            y_max = 1.0
        y = y / y_max
        win_frac = won / (won + lost + 1e-12)
        xpoints.append(win_frac * 100)
        ypoints.append(dist.pdf(win_frac) / y_max)
        c = hv.Area((100 * x, y), xlabel, ylabel, label=label).options(alpha=alpha)
        c_list.append(c)
    c1 = hv.Overlay(c_list).options(legend_position="right")
    c2 = hv.Scatter((xpoints, ypoints), xlabel, ylabel).options(
        color="black", size=8, tools=["hover"]
    )
    return (c1 * c2).options(legend_position=legend_position)
