class Fit:
    OPTIMIZER_NAMES = {'fmin', 'fmin_powell', 'fmin_cg', 'fmin_bfgs', 'fmin_ncg'}

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
        p = self._params.clone()
        return p

    @property
    def params(self):
        p = self.all_params
        p.drop('x_train')
        p.drop('y_train')
        return p

    def given(self, **kwargs):
        self._givens = kwargs
        return self

    def algorithm(self, algorithm_name):
        if algorithm_name not in self.OPTIMIZER_NAMES:
            raise ValueError(f'algorithm must be one of {self.OPTIMIZER_NAMES}')
        self._algorithm = algorithm_name
        return self

    def optimizer_kwargs(self, **kwargs):
        self._optimizer_kwargs = kwargs
        return self

    def _cost_wrapper(self, args, p):
        p.ingest(args)
        return self._cost(p)

    def _default_cost(self, p):
        import numpy as np
        err = self._model(p) - p.y_train
        return np.sum(err ** 2)

    def fit(self, *, x=None, y=None, model=None, cost=None):
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
        self._params.ingest(a_fit)
        return self.params

    def predict(self, x=None):
        if x is not None:
            p = self.all_params
            p.x_train = x
        else:
            p = self._params
        return self._model(p)

    def plot(self, using='holoviews', ax=None):
        allowed_usings = {'holoviews', 'matplotlib'}
        if using not in allowed_usings:
            raise ValueError(f'{using} not in {allowed_usings}')
        p = self._params

        if using == 'holoviews':
            import holoviews as hv
            c = hv.Scatter((p.x_train, p.y_train))
            c = c * hv.Curve((p.x_train, self.predict()))
            display(c)  # noqa

        elif using == 'matplotlib':
            import pylab as pl
            if ax is None:
                pl.figure()
                ax = pl.gca()
            ax.plot(p.x_train, p.y_train, 'o')
            ax.plot(p.x_train, self.predict(), '-')

    def example(self):
        print('this will soon be example')
