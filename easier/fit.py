from textwrap import dedent
import pandas as pd


class examples():
    """
    A descriptor whose only purpose is to print help text
    """
    def __get__(self, *args, **kwargs):
        print(
            dedent("""
            from easier import Fit

            #=================================================================
            #                Model Fitting:  y ~ model(x)
            #=================================================================
            # Get training values
            x_train, y_train = df.months.values, df.rev.values


            # Define three different models for fitting the data
            def exp_model(p):
                return p.R0 * (1 + p.alpha) ** p.x_train

            def quad_model(p):
                return p.bias + p.lin * p.x_train + p.quad * p.x_train ** 2

            def lin_model(p):
                return p.bias + p.lin * p.x_train


            # Fit the exponential model
            fe = ezr.Fit(R0=1e6, alpha=.02)
            fe.fit(x=x_train, y=y_train, model=exp_model)

            # Fit the linear model
            fl = ezr.Fit(bias=1e6, lin=1)
            fl.fit(x=x_train, y=y_train, model=lin_model)

            # Fit the quadratic model
            # Note, if initialization not needed, could also do:
            #     fq = ezr.Fit('bias lin quad')
            # Or
            #     fq = ezr.Fit('bias', 'lin', 'quad')
            fq = ezr.Fit(bias=1e6, lin=1, quad=.1)
            fq.fit(x=x_train, y=y_train, model=quad_model)

            # Create (optional) x values for plotting fits
            x = np.linspace(0, x_train.max() + 12, 300)

            # Scale y variables to manageable numbers
            scale_factor = 1e-6

            # Create an overlay plot of data with model fits
            (
                fe.plot(x=x, show_label=True, label='Exponential', color=ezr.cc.b, scale_factor=scale_factor)
                * fq.plot(x=x, show_label=True, label='Quadratic', color=ezr.cc.c, scale_factor=scale_factor)
                * fl.plot(x=x, show_label=True, label='Linear', color=ezr.cc.d, scale_factor=scale_factor)
            )


            #=================================================================
            #   Set a function to a target value using a cost function
            #=================================================================
            # Define a cost function to mimize for the function you supply
            # Here we just define a quadratic cost function
            def cost(p):
                return (p.target - p.my_func(p.x)) ** 2


            # Make a fitter with the arguments to your function
            f = Fit(x=0)

            # Set the function you want to set to the desired target
            f.extra(my_func=lambda x: x ** 4)

            # Set a target you want your function to hit
            f.given(target=7)

            # Print the results of the fit
            print(f.fit(cost=cost))
            """)
        )
        return None


class Fit:
    """
    This class is a convenience wrapper for scipy optimization.
    Look at Fit.examples attribute to see usage.

    The constructor takes exactly the same arguments as ParamState.
    """
    OPTIMIZER_NAMES = {'fmin', 'fmin_powell', 'fmin_cg', 'fmin_bfgs', 'fmin_ncg'}

    DEFAULT_VERBOSE = True

    examples = examples()

    def __init__(self, *args, **kwargs):
        from easier import ParamState
        self._params = ParamState(*args, **kwargs)
        self._algorithm = 'fmin'
        self._optimizer_kwargs = {}
        self._model = None
        self._givens = {}
        self._cost = self._default_cost
        self._verbose = self.DEFAULT_VERBOSE

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, val):
        self._verbose = val

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
        p.drop('x_train')
        p.drop('y_train')
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

    def algorithm(self, algorithm_name):
        """
        Provide a scipy optimizer name to use.  Enter bogus name to see valid
        options.
        """
        if algorithm_name not in self.OPTIMIZER_NAMES:
            raise ValueError(f'algorithm must be one of {self.OPTIMIZER_NAMES}')
        self._algorithm = algorithm_name
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
        err = self._model(p) - p.y_train

        return np.sum(err ** 2)

    def fit(self, *, x=None, y=None, model=None, cost=None):
        """
        The method to use for training the fitter.  The fitter can be
        used in two modes.

        1) Fit a model function of x to observed values y.
        2) Define an arbitrary cost function to miniize

        Because of this, all the kwargs are optional.  You only need
        to specify the variables that are needed for the task you are doing.
        See the examples.
        """
        import numpy as np
        from scipy import optimize

        self._model = model

        givens = dict(
            x_train=x,
            y_train=y
        )
        givens.update(self._givens)

        self._params.given(**givens)

        a0 = self._params.array

        if cost is not None:
            self._cost = cost

        optimizer = getattr(optimize, self._algorithm)
        self._optimizer_kwargs.update(disp=self.verbose)
        a_fit = optimizer(self._cost_wrapper, a0, args=(self._params,), **self._optimizer_kwargs)
        a_fit = np.array(a_fit, ndmin=1)
        self._params.ingest(a_fit)
        return self.params

    def predict(self, x=None):
        """
        Returns an array of predictions based on trained models.  If
        x is not supplied, than the fit over x_train is returned.
        """
        if x is not None:
            p = self.all_params
            p.x_train = x
        else:
            p = self._params
        return self._model(p)

    def df(self, x=None):
        p = self._params

        if x is None:
            x = p.x_train

        y = self.predict(x)
        return pd.DataFrame({'x': x, 'y': y})

    def plot(
            self, *, x=None, scale_factor=1,
            show=False, show_label=False, label=None,
            color=None, size=10,
            xlabel='x', ylabel='y'

    ):
        """
        Draw plots for model fit results.
        Params:
            x: A custom x over which to draw fits
            scale_factor: Scale all y values by this factor
            show: Controls whether display show the plot or return the hv obj
            show_label: Controls whether to add labels to traces or not
            label: A string with which to label the fit
            color: The color of the fit line

        """
        import easier as ezr
        p = self._params

        if x is None:
            x = p.x_train

        line_color = color if color else ezr.cc.b

        import holoviews as hv
        if 'bokeh' not in hv.Store.registry:
            hv.extension('bokeh')
        # label_val = 'Data' if show_label else ''
        label_val = ''
        c = hv.Scatter(
            (p.x_train, scale_factor * p.y_train), xlabel, ylabel, label=label_val
        ).options(color=ezr.cc.a, size=size, alpha=.5)

        label_val = label if label else 'Fit'
        label_val = label_val if show_label else ''
        c = c * hv.Curve((x, scale_factor * self.predict(x)), label=label_val).options(color=line_color)
        if show:
            display(c)  # noqa
        else:
            return c
