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
            #                Model Fitting:  y ~ model(x)
            #=================================================================

            # Function to generate fake data
            def get_model_data(*, const=7, factor=.2, power=2, noise=.5):
                import numpy as np
                # Make some fake data
                x = np.linspace(0, 10, 100)
                y = const + factor * x ** power
                y = y + noise * np.random.randn(len(x))
                return x, y

            # Generate some fake data
            x_train, y_train = get_model_data()


            # Define a model that returns y_fit for a set of params
            def model(p):
                return p.const + p.factor * p.x_train ** p.power

            # Get your training data
            x_train, y_train = get_model_data()

            # Create a fitter with vars matching what's in your model
            f = Fit('const factor power')

            # Train the fitter using the data and the model
            params = f.fit(x=x_train, y=y_train, model=model)

            # Just show how you can get prediction values.
            y_fit = f.predict()

            # Print params and plot results
            print(f.params)
            f.plot(show=True)



            #=================================================================
            #     Function Minimizing: optimal_vals ~ Minimize(cost)
            #=================================================================

            # Define a cost function to mimize based on
            def cost(p):
                return (p.target - p.estimate) ** 2

            # Create a loop to show use of all available miminizers
            for algo in ['fmin', 'fmin_powell', 'fmin_bfgs', 'fmin_cg']:

                # Make a fitter with only one parameter, 'estimate'
                f = Fit('estimate')

                # Set the optimization algorithm
                f.algorithm(algo)

                # Set a constant for the optimization
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
    """
    OPTIMIZER_NAMES = {'fmin', 'fmin_powell', 'fmin_cg', 'fmin_bfgs', 'fmin_ncg'}

    examples = examples()

    def __init__(self, *args, **kwargs):
        from easier import ParamState
        self._params = ParamState(*args, **kwargs)
        self._algorithm = 'fmin'
        self._optimizer_kwargs = {}
        self._model = None
        self._givens = {}
        self._cost = self._default_cost

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

    def plot(self, show=False, show_label=False):
        """
        Draw plots for model fit results.
        If show=True, than just display the plot
        If show=False, don't display, just return the holovies object.
        """
        import easier as ezr
        p = self._params

        import holoviews as hv
        if 'bokeh' not in hv.Store.registry:
            hv.extension('bokeh')
        label_val = 'Data' if show_label else ''
        c = hv.Scatter((p.x_train, p.y_train), label=label_val).options(color=ezr.cc.a)
        label_val = 'Fit' if show_label else ''
        c = c * hv.Curve((p.x_train, self.predict()), label=label_val).options(color=ezr.cc.b)
        if show:
            display(c)  # noqa
        else:
            return c
