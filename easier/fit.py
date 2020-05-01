from textwrap import dedent


class examples():
    """
    A descriptor whose only purpose is to print help text
    """
    def __get__(self, *args, **kwargs):
        print(
            dedent("""
            from easier import Fit

            #=================================================================
            #                Simple Model Fitting:  y ~ model(x)
            #=================================================================
            # Make data from noise-corrupted sinusoid
            x = np.linspace(0, 2 * np.pi, 100)
            y = np.sin(x) - .7 * np.cos(x) + .1 * np.random.randn(len(x))


            # Define a model function you want to fit to
            # All model parameters are on the p object.
            # The names "x", and "y" are reserved for the data you are fitting
            def model(p):
                return p.a * np.sin(p.k * p.x) + p.b * np.cos(p.k * p.x)

            # Initialize a fitter with purposefully bad guesses
            fitter = ezr.Fitter(a=-1, b=2, k=.2)

            # Fit the data and plot fit quality every 5 iterations
            fitter.fit(x=x, y=y, model=model, plot_every=5)

            # Plot the final results
            display(fitter.plot())
            display(fitter.params.df)

            #=================================================================
            #                Advanced Fitting:  y ~ model(x)
            #=================================================================
            # Make data from noise-corrupted sinusoid
            x = np.linspace(0, 2 * np.pi, 100)
            y = np.sin(x) - .7 * np.cos(x) + .1 * np.random.randn(len(x))

            # Define a model function you want to fit to
            # All model parameters are on the p object.
            # The names "x", and "y" are reserved for the data you are fitting
            def model(p):
                return p.a * np.sin(p.k * p.x) + p.b * np.cos(p.k * p.x)

            # Initialize a fitter with purposefully bad guesses
            fitter = ezr.Fitter(a=-1, b=2, k=.2)

            # Fit the data and plot fit quality every 5 iterations
            fitter.fit(
                x=x,                   # The independent data
                y=y,                   # The dependent data
                model=model,           # The model function
                plot_every=5,          # Plot fit every this number of iterations
                algorithm='fmin_bfgs', # Scipy optimization routine to use
                verbose=False          # Don't print convergence info
            )

            # Get predictions at specific values
            x_predict = np.linspace(0, 6 * np.pi, 300)
            y_predict = fitter.predict(x_predict)

            # Get the components of the fit chart
            components = fitter.plot(
                x=x_predict,
                scale_factor=10,
                label='10X Scaled Fit',
                line_color='red',
                scatter_color='blue',
                size=15,
                xlabel='My X Label',
                ylabel='My Y Label',
                as_components=True,
            )

            # Display the components as a layout rather than overlay
            display(hv.Layout(components))

            #=================================================================
            #   Set a function to a target value using a cost function
            #   THIS IS NOT CURRENTLY MAINTAINED SO IT MAY OR MAY NOT WORK.
            #=================================================================
            # Define a cost function to mimize for the function you supply
            # Here we just define a quadratic cost function
            def cost(p):
                return (p.target - p.my_func(p.x)) ** 2


            # Make a fitter with the arguments to your function
            f = Fitter(x=0)

            # Set the function you want to set to the desired target
            f.extra(my_func=lambda x: x ** 4)

            # Set a target you want your function to hit
            f.given(target=7)

            # Print the results of the fit
            print(f.fit(cost=cost))
            """)
        )
        return None


class Fitter:
    """
    This class is a convenience wrapper for scipy optimization.
    Look at Fitter.examples attribute to see usage.

    The constructor takes exactly the same arguments as ParamState.
    """
    OPTIMIZER_NAMES = {'fmin', 'fmin_powell', 'fmin_cg', 'fmin_bfgs'}

    DEFAULT_VERBOSE = True

    examples = examples()

    def __init__(self, *args, **kwargs):
        """
        *args and **kwargs define/initialize the model variables.
        They are passed directly to a ParamState constructor
        """
        from easier import ParamState
        self._params = ParamState(*args, **kwargs)
        self._algorithm = 'fmin'
        self._optimizer_kwargs = {}
        self._model = None
        self._givens = {}
        self._cost = self._default_cost
        self._verbose = self.DEFAULT_VERBOSE

    @property
    def all_params(self):
        """
        Show all parameters (even data variables)
        """
        p = self._params.clone()
        return p

    @property
    def params(self):
        """
        Show all params except data variables
        """
        p = self.all_params
        p.drop('x')
        p.drop('y')
        return p

    def extra(self, **kwargs):
        """
        Put extra attributes on the params object.  These params will
        be completely ignored by the optimizer.  This is good for passing
        utility functions to your model or cost fuctions.
        """
        for key, val in kwargs.items():
            setattr(self._params, key, val)

    def given(self, **kwargs):
        """
        Supply kwargs that will set constant variables.  These can either
        be new variables or already defined variables.  This allows you
        to easily turn on and off variables to optimize.
        """
        self._givens = kwargs
        return self

    def optimizer_kwargs(self, **kwargs):
        """
        Additional kwargs to pass to optimizer.
        """
        self._optimizer_kwargs = kwargs
        return self

    def _cost_wrapper(self, args, p):
        """
        Allows users to define their cost functions as only functions
        of p, the ParamState object.
        """
        p.ingest(args)
        return self._cost(p)

    def _default_cost(self, p):
        """
        Allows users to define their cost functions as only functions
        of p, the ParamState object.
        """
        import numpy as np
        err = self._model(p) - p.y

        return np.sum(err ** 2)

    def _plotter(self, *args, **kwargs):
        """
        A function to plot fits in real time
        """
        import holoviews as hv
        import easier as ezr
        if kwargs['data']:
            xd, yd = kwargs['data'][-2:]

        else:
            xd, yd = [0], [0]

        fit_line = hv.Curve(*args, **kwargs)
        data = hv.Scatter((xd, yd))
        overlay = hv.Overlay([fit_line, data])

        overlay = overlay.opts(
            hv.opts.Curve(color=ezr.cc.b),
            hv.opts.Scatter(color=ezr.cc.a, size=5, alpha=.5)
        )
        return overlay

    def _model_wrapper(self, model):
        """
        The model wrapper will wrap the model with a function
        that updates real time plots of fit if needed.
        """
        if self.plot_every is None:
            return model
        else:
            self.plot_counter = 0

            def wrapped(*args, **kwargs):
                yfit = model(*args, **kwargs)
                if self.plot_counter % self.plot_every == 0:
                    self.pipe.send((self._params.x, yfit, self._params.x, self._params.y))
                self.plot_counter += 1
                return yfit

            return wrapped

    def fit(self, *, x=None, y=None, model=None, cost=None, plot_every=None, algorithm='fmin', verbose=True):
        """
        The method to use for training the fitter.
        Args:
                     x: the indepenant data
                     y: the depenant data
                 model: a model function to fit the data to
                  cost: a custom cost function (defaults to least squares)
            plot_every: Plot solution in real time every this number of iterations
             algorithm: Scipy optimization routine to use.  Enter nonsense to see list of valid.
               verbose: Print convergence information
        """
        import numpy as np
        from scipy import optimize
        import holoviews as hv
        from holoviews.streams import Pipe
        from IPython.display import display
        self.plot_every = plot_every

        if algorithm not in self.OPTIMIZER_NAMES:
            raise ValueError(f'Invalid optimizer {algorithm}.  Choose one of {self.OPTIMIZER_NAMES}')

        givens = dict(
            x=x,
            y=y
        )
        givens.update(self._givens)

        self._params.given(**givens)

        # This stuff only needs to happen if we are iteratively plotting fits
        if plot_every is not None:
            x, y = self._params.x, self._params.y
            xmin, xmax = np.min(x), np.max(x)
            ymin, ymax = np.min(y), np.max(y)

            scale = .1
            delta_x = scale * (xmax - xmin)
            delta_y = scale * (ymax - ymin)

            xmin, xmax = xmin - delta_x, xmax + delta_x
            ymin, ymax = ymin - delta_y, ymax + delta_y

            xlim = (xmin, xmax)
            ylim = (ymin, ymax)
            self.pipe = Pipe(data=[])
            try:
                dmap = hv.DynamicMap(self._plotter, streams=[self.pipe])
                dmap.opts(hv.opts.Overlay(xlim=xlim, ylim=ylim))
                display(dmap)
            except AttributeError:
                raise RuntimeError('You must import holoviews and set bokeh backround for plotting to work')

        if model is None:
            raise ValueError('You must supply a model function')

        self._raw_model = model
        self._model = self._model_wrapper(self._raw_model)

        a0 = self._params.array

        if cost is not None:
            self._cost = cost

        optimizer = getattr(optimize, algorithm)
        self._optimizer_kwargs.update(disp=verbose)
        a_fit = optimizer(self._cost_wrapper, a0, args=(self._params,), **self._optimizer_kwargs)
        a_fit = np.array(a_fit, ndmin=1)
        self._params.ingest(a_fit)
        return self

    def predict(self, x=None):
        """
        Returns an array of predictions based on trained models.  If
        x is not supplied, than the fit over x values in training set is used
        """
        if x is not None:
            p = self.all_params
            p.x = x
        else:
            p = self._params
        return self._raw_model(p)

    def df(self, x=None):
        import pandas as pd
        p = self._params

        if x is None:
            x = p.x

        y = self.predict(x)
        return pd.DataFrame({'x': x, 'y': y})

    def plot(
            self, *,
            x=None,
            scale_factor=1,
            label=None,
            line_color=None,
            scatter_color=None,
            size=10,
            xlabel='x',
            ylabel='y',
            as_components=False

    ):
        """
        Draw plots for model fit results.
        Params:
            x: A custom x over which to draw fits
            scale_factor: Scale all y values by this factor
            label: A string with which to label the fit
            line_color: Color for fit line
            scatter_color: Color for data points
            size: size of scatter points
            xlabel: x axis label
            ylabel: y axis label
            as_components: if True, return chart components rather than overlay
        """
        import easier as ezr
        p = self._params

        if x is None:
            x = p.x

        line_color = line_color if line_color else ezr.cc.b
        scatter_color = scatter_color if scatter_color else ezr.cc.a

        import holoviews as hv

        label_val = label if label else 'Fit'

        try:
            scatter = hv.Scatter(
                (p.x, scale_factor * p.y), xlabel, ylabel, label=label_val
            ).options(color=scatter_color, size=size, alpha=.5)
        except Exception:
            raise RuntimeError('You must import holoviews and set bokeh backround for plotting to work')

        line = hv.Curve((x, scale_factor * self.predict(x)), label=label_val).options(color=line_color)
        traces = [
            scatter,
            line
        ]
        if as_components:
            return traces
        else:
            return hv.Overlay(traces)
