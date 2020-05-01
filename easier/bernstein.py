from typing import Optional, Tuple


class Compress:
    def __init__(self, min_val=None, max_val=None):
        """
        A utility class that will map numbers into a the range [0, 1]
        Args:
            min_val: Manually set the limits to be used
            max_val: Manually set the limits to be used
        """
        self._min_val = min_val
        self._max_val = max_val
        self.xmin = None
        self.ymin = None
        self._set_limits()

    def _set_limits(self, x=None):
        import numpy as np

        xmin, xmax = None, None

        if x is not None:
            xmin = np.min(x)
            xmax = np.max(x)

        if self._min_val is not None:
            xmin = self._min_val
        if self._max_val is not None:
            xmax = self._max_val

        self.xmin = xmin
        self.xmax = xmax

    def compress(self, x, learn_limits=True):
        """
        Compresses the array to [0, 1]
        """
        import numpy as np

        if not isinstance(x, np.ndarray):
            raise ValueError('Input must be numpy array')

        if learn_limits:
            self._set_limits(x)

        if None in {self.xmin, self.xmax}:
            raise ValueError('Compressor never learned range limit')

        return (x - self.xmin) / (self.xmax - self.xmin)

    def expand(self, x):
        """
        Expands an array to go from [0, 1] interval to original interval
        """
        x = x.flatten()
        xmin = self.xmin
        xmax = self.xmax

        x = xmin + (xmax - xmin) * x
        return x

    @property
    def _dc_dx(self):
        """
        Returns the derivative of compressed to input
        """
        return self.xmax - self.xmin


class Bernstein(Compress):
    _EPS = 1e-15

    def __init__(self, *, x, y, N=500, xlim: Optional[Tuple[float]] = None):
        """
        Bernstein function approximator of either a callable or x, y data.
        Args:
            x: x data to fit
            y: y data to fit
            N: the order of the fitting polynomial
            xlim: An optional tuple of hardcoced x limits within which to fit the function
        """
        if xlim is None:
            min_val, max_val = None, None
        else:
            min_val, max_val = xlim
        super().__init__(min_val, max_val)
        self._validate_inputs(x, y)
        self.N = N

        self._x = self.compress(x)
        self._y = y

        self._func = self._get_interp_function()

    def _get_interp_function(self):
        from scipy.interpolate import interp1d
        return interp1d(self._x, self._y, fill_value=(self._y[0], self._y[-1]), bounds_error=False)

    def _validate_inputs(self, x, y):
        import numpy as np

        if not all(isinstance(v, np.ndarray) for v in [x, y]):
            raise ValueError('x and y must both be numpy arrays')

        if not all(len(v) > 2 for v in [x, y]):
            raise ValueError('x and y must both have at least 3 elements')

    def _bern_term(self, n, k, x):
        """
        This function uses logs to make the following code
        safe for large n.

        cc = comb(n, k)
        return cc * (1 + self._EPS - x) ** (n - k) * (x + self._EPS) ** k
        """
        import numpy as np
        from scipy.special import gammaln
        x = np.clip(x, self._EPS, 1 - self._EPS)

        out = gammaln(n + 1) - gammaln(n - k + 1) - gammaln(k + 1)
        out += k * np.log(x) + (n - k) * np.log(1 - x)
        return np.exp(out)

    def _get_fit_func(self, infunc):
        """
        Returns a callable of the bernstein approximator
        """
        import numpy as np
        N = self.N
        k_vec = np.arange(N + 1)
        coeff_vec = infunc(k_vec / N)

        def bern_sum(x):
            x = x.flatten()
            X, K = np.meshgrid(x, k_vec)
            _, C = np.meshgrid(x, coeff_vec)
            B = self._bern_term(N, K, X)

            terms = C * B
            out = np.sum(terms, axis=0)
            return out
        return bern_sum

    def _get_fit_deriv(self, infunc):
        """
        Returns a callable of the bernstein derivative approximator
        """
        import numpy as np
        N = self.N
        k_vec = np.arange(N + 1)
        coeff_vec = infunc(k_vec / N)

        def bern_sum(x):
            x = x.flatten()
            X, K = np.meshgrid(x, k_vec)
            _, C = np.meshgrid(x, coeff_vec)

            B1 = self._bern_term(N - 1, K - 1, X)
            B2 = self._bern_term(N - 1, K, X)

            terms = C * N * (B1 - B2)
            out = np.sum(terms, axis=0)
            return out

        return bern_sum

    def predict(self, x):
        """
        Return the Bernstein approximate to the function at the specified value.
        """
        # learn_limits = not hasattr(self, 'xmax') or self.xmax is None
        # x = self.compress(x, learn_limits=learn_limits)
        func = self._get_fit_func(self._func)
        return func(self.compress(x))

    def predict_derivative(self, x):
        """
        Return the Bernstein approximate to the function derivative at the specified value.
        """
        # learn_limits = not hasattr(self, 'xmax') or self.xmax is None
        # x = self.compress(x, learn_limits=learn_limits)
        func = self._get_fit_deriv(self._func)
        out = func(self.compress(x))
        out = out / self._dc_dx
        return out
