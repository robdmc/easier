class DistFitter:
    def __init__(self, dist_class, num_hist_bins=None, hist_range=None):
        """
        Args:
            dist_class: A scipy.stats distribution class to fit to
            num_hist_bins: The number of bins to use for the empirical fitter
            hist_range: The range to use in the impirical fitter
        """
        self.dist_class = dist_class
        self._num_hist_bins = num_hist_bins
        self._hist_range = hist_range

    def fit(self, data, **fit_kwargs):
        """
        Args:
            data: iterable of data you want to fit to
        Kwargs:
            shape: The initial guess for the shape parameter(s) you want to pass to the fitter
            f0, f1, ...: Pin the shape parameter(s) to these values
            loc: The starting guess for the location parameter
            scale: the starting guess for the scale parameter
            floc: pin the lcoation parameter to this value
            fscale: pin the scale parameter to this value
        """
        from scipy import stats
        import numpy as np
        data = np.array(data)
        shape = fit_kwargs.pop('shape', None)
        if shape:
            args = (data, shape)
        else:
            args = (data,)
        self.dist_params = self.dist_class.fit(*args, **fit_kwargs)
        self.dist = self.dist_class(*self.dist_params)
        if self._num_hist_bins is None:
            hist_bins = int(len(data) / 5)
        else:
            hist_bins = self._num_hist_bins

        self.hist_dist = stats.rv_histogram(np.histogram(data, bins=hist_bins, range=self._hist_range))
        min_val, max_val = self.hist_dist.ppf([.01, .99])
        if self._hist_range:
            min_val, max_val = self._hist_range
        self.x = np.linspace(min_val, max_val, 600)

        (self.q_theory, self.q_data), (self._a, self._b, _) = stats.probplot(data, dist=self.dist, plot=None, fit=True)

        return self

    def _plot(self, label, method_name, logx=False, logy=False, data_color=None, fit_color=None, xlabel='value'):
        import holoviews as hv
        import easier as ezr
        if data_color is None:
            data_color = ezr.cc[0]
        if fit_color is None:
            fit_color = ezr.cc[1]
        func = getattr(self.dist, method_name)
        hist_func = getattr(self.hist_dist, method_name)
        c1 = hv.Curve(
            (self.x, 1e-9 + func(self.x)),
            xlabel,
            'Density',
            label=f'{self.dist.dist.name} Fit'
        ).options(logx=logx, logy=logy, color=fit_color)
        c2 = hv.Curve((self.x, 1e-9 + hist_func(self.x)), label=f'{label} Empirical').options(color=data_color)
        return hv.Overlay([c1, c2]).options(legend_position='top')

    def plot_pdf(self, label='', logx=False, logy=False, data_color=None, fit_color=None, xlabel='value'):
        """
        Plot the pdf
        """
        return self._plot(
            label, method_name='pdf', logx=logx, logy=logy, data_color=data_color, fit_color=fit_color, xlabel=xlabel)

    def plot_cdf(self, label='', logx=False, logy=False, data_color=None, fit_color=None, xlabel='value'):
        """
        Plot the cdf
        """
        return self._plot(
            label, method_name='cdf', logx=logx, logy=logy, data_color=data_color, fit_color=fit_color, xlabel=xlabel)

    def plot_qq(self, label='', logx=False, logy=False, data_color=None, fit_color=None):
        """
        Make a q-q plot.  X axis will hold theoretical quantiles, Y axis the impirircal
        """
        import holoviews as hv
        import numpy as np
        if data_color is None:
            data_color = 'blue'
        if fit_color is None:
            fit_color = 'red'

        q_data_fit = np.polyval([self._a, self._b], self.q_theory)
        c1 = hv.Scatter(
            (self.q_theory, self.q_data),
            'Theoretical Quantiles',
            'Empirical Quantiles',
            label=label
        ).options(size=5, alpha=.5, color=data_color, logx=logx, logy=logy)
        label = f'{self.dist.dist.name}  params = {["%0.4g" % p for p in self.dist_params]}'.replace("'", "")
        c2 = hv.Curve((self.q_theory, q_data_fit), label='Best Fit Line').options(color=fit_color, alpha=.2)
        c3 = hv.Curve((self.q_theory, self.q_theory), label=label).options(color='green', alpha=.2)
        return hv.Overlay([c1, c2, c3]).options(legend_position='top')
