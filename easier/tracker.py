import textwrap


class Tracker:
    example = textwrap.dedent(
        "\n    # REQUIRES CODE IN TWO DIFFERENT NOTEBOOK CELLS\n    \n\n    ***\n    Note: There can be some weird holovies syncing issues\n    between python and javascript for this.  So occational\n    page refreshes might be needed.\n    \n    **** FOR THIS REASON YOU MAY WANT TO USE ezr.StreamPlotter instead ****\n    ***\n\n    The way this works is to start one notebook cell\n    with the tracker code.  This will remain live\n    and will get updated whenever you run the logging cell.\n\n    After the tracking cell has been set up, you can\n    run the logging cell multiple times and the\n    tracking cell's visualization will be updated.\n\n    # -- Single-trace use case -----------------------\n\n    # In tracking notebook cell\n    # (Note: logy can be a bit tricky.  You might need to first run without it.)\n    tracker = ezr.Tracker(logy=True)\n    tracker.init()\n\n    # In logging cell\n\n    for nn in np.linspace(1, 20, 50):\n        x = np.linspace(.1, nn, 500)\n        y = 1 / x\n        tracker.update(x, y)\n        time.sleep(.02)\n\n\n\n    # -- Multi-trace use case -----------------------\n    # In tracking notebook cell\n    ylim = (-2, 2)\n    tracker1 = ezr.Tracker(label='+s', ylim=ylim)\n    tracker2 = ezr.Tracker(label='-s', ylim=ylim)\n    tracker3 = ezr.Tracker(label='+c', ylim=ylim)\n    tracker4 = ezr.Tracker(label='-c', ylim=ylim)\n\n    hv.Overlay([\n        tracker1.init(),\n        tracker2.init(),\n        tracker3.init(),\n        tracker4.init()\n    ]).collate()\n\n    # In logging cell\n    for nn in np.linspace(0, 20, 50):\n        x = np.linspace(0, nn, 500)\n        y = np.sin(2 * np.pi * x / (4))\n        y2 = np.cos(2 * np.pi * x / (4))\n        tracker1.update(x, y)\n        tracker2.update(x, -y)\n        tracker3.update(x, y2)\n        tracker4.update(x, -y2)\n        time.sleep(.01)\n    "
    )

    def __init__(self, label="metric", ylim=None, logy=False, width=800, height=400):
        from holoviews.streams import Pipe
        import holoviews as hv

        self.label = label
        self.ylim = ylim
        self.logy = logy
        self.pipe = Pipe(data=None)
        if hv.Store.current_backend == "bokeh":
            self.dmap = hv.DynamicMap(self._plotter, streams=[self.pipe]).opts(
                framewise=True, width=width, height=height, logy=self.logy
            )
        else:
            self.dmap = hv.DynamicMap(self._plotter, streams=[self.pipe]).opts(
                framewise=True, fig_inches=12, aspect=2, logy=self.logy
            )

    def _plotter(self, data):
        import holoviews as hv

        default_val = hv.Curve(([], []))
        if data is None:
            return default_val
        x, y = data
        c = hv.Curve((x, y), label=self.label)
        return c

    def init(self):
        return self.dmap

    def update(self, x, y):
        self.pipe.send((x, y))
