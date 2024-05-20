from typing import Optional, Tuple
from .utils import BlobAttr, BlobMixin, Scaler


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
            raise ValueError("Input must be numpy array")

        if learn_limits:
            self._set_limits(x)

        if None in {self.xmin, self.xmax}:
            raise ValueError("Compressor never learned range limit")

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

        return interp1d(
            self._x, self._y, fill_value=(self._y[0], self._y[-1]), bounds_error=False
        )

    def _validate_inputs(self, x, y):
        import numpy as np

        if not all(isinstance(v, np.ndarray) for v in [x, y]):
            raise ValueError("x and y must both be numpy arrays")

        if not all(len(v) > 2 for v in [x, y]):
            raise ValueError("x and y must both have at least 3 elements")

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


class BernsteinFitter(BlobMixin):
    _EPS = 1e-15

    w = BlobAttr(None)
    scaler_blob = BlobAttr(None)

    def __init__(
        self,
        non_negative=False,
        monotonic=False,
        increasing=True,
        match_left=False,
        match_right=False,
        match_endpoint_values=False,
        match_endpoint_derivatives=False,
    ):
        super().__init__()
        self._non_negative = non_negative
        self._monotonic = monotonic
        self._increasing = increasing
        self._match_left = match_left
        self._match_right = match_right
        self._match_endpoint_values = match_endpoint_values
        self._match_endpoint_derivatives = match_endpoint_derivatives

    def _bern_term(self, n, k, x):
        """
        This function uses logs to make the following code
        safe for large n.

        cc = comb(n, k)
        return cc * (1 + self._EPS - x) ** (n - k) * (x + self._EPS) ** k
        """
        import numpy as np
        from scipy.special import gammaln

        if (k < 0) or (k > n):
            return np.zeros_like(x)

        x = np.clip(x, self._EPS, 1 - self._EPS)

        out = gammaln(n + 1) - gammaln(n - k + 1) - gammaln(k + 1)
        out += k * np.log(x) + (n - k) * np.log(1 - x)
        return np.exp(out)

    def _get_design_matrix(self, x, degree):
        """
        x is the array of x points
        n is degree of fitter
        """
        import numpy as np

        x = np.array(x)
        A = np.zeros((len(x), degree + 1))
        for k in range(0, degree + 1):
            A[:, k] = self._bern_term(degree, k, x)
        return A

    def _get_derivative_matrix(self, x, degree):
        import numpy as np

        n = degree
        if hasattr(x, "__iter__"):
            B = np.zeros((len(x), degree + 1))
        else:
            B = np.zeros((1, degree + 1))

        for k in range(0, degree + 1):
            term1 = self._bern_term(n - 1, k - 1, x)
            term2 = self._bern_term(n - 1, k, x)
            B[:, k] = n * (term1 - term2)
        return B

    def _get_integral_matrix(self, x, degree):
        import numpy as np

        n = degree

        A = self._get_design_matrix(x, degree + 1)
        B = np.zeros_like(A)
        coeff = 1 / (n + 1)
        for k in range(0, degree + 1):
            X = A[:, (k + 1) :]
            B[:, k] = coeff * np.sum(X, axis=1)
        return B

    def get_design_matrix(self, x, degree):
        import numpy as np

        x = np.array(x)

        scaler = Scaler()
        x = scaler.fit_transform(x)
        self.scaler_blob = scaler.to_blob()

        A = self._get_design_matrix(x, degree)
        return A

    def fit_predict(self, x, y, degree, regulizer=0.0, verbose=False):
        self.fit(x, y, degree, regulizer=regulizer, verbose=verbose)
        return self.predict(x)

    def fit(self, x, y, degree, regulizer=0.0, verbose=False):
        import cvxpy as cp
        import numpy as np

        x = np.array(x)
        y = np.array(y)

        scaler = Scaler()
        x = scaler.fit_transform(x)
        self.scaler_blob = scaler.to_blob()

        yv = np.reshape(y, (-1, 1))
        A = self._get_design_matrix(x, degree)
        B = self._get_derivative_matrix(x, degree)

        # Define a weight variable to be optimized
        w = cp.Variable(name="w", shape=(degree + 1, 1))

        # The objective is the mininum squared error
        objective = cp.Minimize(cp.sum_squares(A @ w - yv) + regulizer * cp.norm(w, 2))

        # Default to unconstrained
        constraints = []

        if self._non_negative:
            constraints.append(w >= np.zeros(w.shape))

        if self._monotonic:
            if self._increasing:
                constraints.append(B @ w >= np.zeros_like(yv))
            else:
                constraints.append(B @ w <= np.zeros_like(yv))

        if self._match_left:
            constraints.append(w[0, 0] == y[0])

        if self._match_right:
            constraints.append(w[-1, 0] == y[-1])

        # These constraints ensure matching values at end points (periodic)
        if self._match_endpoint_values:
            if self._match_left or self._match_right:
                raise ValueError(
                    "Cannot have match_left or match_right with match_endpoint_values"
                )
            # Constrain the values at the endpoints to match
            constraints.append(w[0, 0] == w[-1, 0])

        # These constraints ensure matching derivative at end points (periodic)
        if self._match_endpoint_derivatives:
            # Get the derivative matrix
            B = self._get_derivative_matrix(x, degree)

            # Constrain the derivatives at the endpoints to match
            term1 = B[0, :] @ w
            term2 = B[-1, :] @ w
            constraints.append(term1 == term2)

        # Add any desired constraints
        kwargs = {}
        if constraints:
            kwargs["constraints"] = constraints

        # Solve the problem
        problem = cp.Problem(objective, **kwargs)
        problem.solve(verbose=verbose)

        self.w = w.value.flatten()

        return self

    def predict(self, x):
        """
        Values that fall outside the fiited x range will be pegged to
        the terminal values of the fitter.
        """
        return self._get_prediction(x, "value")

    def predict_derivative(self, x):
        if self.w is None:
            raise ValueError(
                "You must run fit() or load a blob before running predict()"
            )

        scaler = Scaler()
        scaler.from_blob(self.scaler_blob)
        diffs = self._get_prediction(x, "derivative")
        return diffs / (scaler.limits[1] - scaler.limits[0])

    def predict_integral(self, x):
        if self.w is None:
            raise ValueError(
                "You must run fit() or load a blob before running predict()"
            )

        scaler = Scaler()
        scaler.from_blob(self.scaler_blob)
        result = self._get_prediction(x, "integral")
        return result * (scaler.limits[1] - scaler.limits[0])

    def _get_prediction(self, x, what):
        import numpy as np

        if self.w is None:
            raise ValueError(
                "You must run fit() or load a blob before running predict()"
            )

        is_scalar = False
        if not hasattr(x, "__iter__"):
            is_scalar = True
            x = [x]

        x = np.array(x)
        scaler = Scaler()
        scaler.from_blob(self.scaler_blob)
        x = scaler.transform(x)

        degree = len(self.w) - 1
        wv = np.reshape(self.w, (-1, 1))
        if what == "value":
            A = self._get_design_matrix(x, degree)
            yv = A @ wv
        elif what == "derivative":
            B = self._get_derivative_matrix(x, degree)
            yv = B @ wv
        elif what == "integral":
            B = self._get_integral_matrix(x, degree)
            yv = B @ wv
        else:
            raise ValueError(f'Nope!  {what!r} is a bad "what" argument')

        yv = yv.flatten()
        if is_scalar:
            return yv[0]
        else:
            return yv

    def to_blob(self):
        blob = super().to_blob()
        blob["w"] = list(blob["w"])
        return blob

    def from_blob(self, blob):
        import numpy as np

        super().from_blob(blob)
        self.w = np.array(self.w)

        return self
