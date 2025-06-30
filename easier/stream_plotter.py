from matplotlib import pyplot as plt
import textwrap

import textwrap

class StreamPlotter:
    """
    Make jupyter streaming plots using matplotlib.
    You can track multiple overlaid plots on multiple
    subplots.  The entire matplotlib figure is redrawn
    for each call to .track(), so the rendering performanc
    is slow.  Requires the ipympl package.

    pip install ipympl

    Then, in the notebook:
    %matplotlib ipympl
    """
    example = textwrap.dedent("\n        # Somewhere at the top of the notebook\n        %matplotlib ipympl\n\n        # In a jupyter cell you want to show the live plots\n        # (Don't need any kwargs if you only want a single plot)\n        plotter = ezr.StreamPlotter(rows=1, cols=2)\n        plotter.init()\n\n        # In a different jupyter cell with code you want to track\n        t = np.linspace(0, 2 * np.pi, 30)\n        for t0 in np.linspace(0, 2 * np.pi, 50):\n            ys = np.sin(t0 - t)\n            yc = np.cos(t0 - t)\n            plotter.update(t, ys, subplot=0, name='left sin', plotspec='b.-', )\n            plotter.update(t, yc, subplot=0, name='left cos', plotspec='r', )\n            plotter.update(t, yc, subplot=1, name='right ysp', plotspec='b', )\n            plotter.update(t, -yc, subplot=1, name='right ysm', plotspec='r', )\n    ")

    def __init__(self, rows=1, cols=1):
        """
        Args:
            rows: number of subplot rows
            cols: number of subplot cols
        """
        self.line_dict = {}
        self.rows = rows
        self.cols = cols

    def init(self):
        """
        Initializes a tracker by setting up the axes on which
        plots will be made.  This should be done in a different
        jupyter notebook cell than that in which the .track()
        method is called.
        """
        self.figure, self.ax_list = plt.subplots(self.rows, self.cols, figsize=(9, 5))
        if self.rows * self.cols == 1:
            self.ax_list = [self.ax_list]
        else:
            self.ax_list = self.ax_list.flatten()
        for ax in self.ax_list:
            ax.grid(True)
        plt.show(block=False)

    def update(self, x, y, name='metric', subplot=0, plotspec='k-', logy=False):
        """
        Call this method in your loop to update the plots.
        Args:
            x: The x values to plot
            y: The y values to plot
            name: A label to attach to this curve
            subplot: Which subplot to place the plot in
            plotspec: The matplotlib plot specification.
            logy: Set the y-axis to log scale

        """
        if not 0 <= subplot < len(self.ax_list):
            raise ValueError(f'Must have 0 <= subplot < {len(self.ax_list)}')
        ax = self.ax_list[subplot]
        if logy:
            ax.set_yscale('log')
        line = self.line_dict.get(name)
        if line is None:
            line = self.line_dict[name] = ax.plot(x, y, plotspec, label=name)[0]
            ax.legend(loc='best')
        else:
            line.set_data(x, y)
            ax.relim()
            ax.autoscale_view()
        self.figure.canvas.draw()