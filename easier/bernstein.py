from .utils import BlobAttr, BlobMixin, Scaler
from scipy.interpolate import interp1d
from scipy.special import gammaln
from typing import Optional, Tuple
import cvxpy as cp
import numpy as np

from typing import Optional, Tuple
from .utils import BlobAttr, BlobMixin, Scaler

class Compress:
    """A utility class that maps numbers into the range [0, 1].

    This class provides methods to compress and expand numerical values between a specified range
    and the [0, 1] interval. It can either learn the range from input data or use manually
    specified limits.

    Args:
        min_val (float, optional): Manually set the minimum value for the range. Defaults to None.
        max_val (float, optional): Manually set the maximum value for the range. Defaults to None.
    """

    def __init__(self, min_val=None, max_val=None):
        self._min_val = min_val
        self._max_val = max_val
        self.xmin = None
        self.ymin = None
        self._set_limits()

    def _set_limits(self, x=None):
        """Set the minimum and maximum values for the compression range.

        If x is provided, the limits are learned from the data. Otherwise, uses manually
        specified limits if they were provided during initialization.

        Args:
            x (numpy.ndarray, optional): Input data to learn limits from. Defaults to None.
        """
        xmin, xmax = (None, None)
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
        """Compress input values to the [0, 1] range.

        Args:
            x (numpy.ndarray): Input array to compress.
            learn_limits (bool, optional): Whether to learn the range limits from the input data.
                Defaults to True.

        Returns:
            numpy.ndarray: Compressed values in the range [0, 1].

        Raises:
            ValueError: If input is not a numpy array or if range limits are not set.
        """
        if not isinstance(x, np.ndarray):
            raise ValueError('Input must be numpy array')
        if learn_limits:
            self._set_limits(x)
        if None in {self.xmin, self.xmax}:
            raise ValueError('Compressor never learned range limit')
        return (x - self.xmin) / (self.xmax - self.xmin)

    def expand(self, x):
        """Expand values from [0, 1] range back to the original range.

        Args:
            x (numpy.ndarray): Input array in range [0, 1] to expand.

        Returns:
            numpy.ndarray: Expanded values in the original range.
        """
        x = x.flatten()
        xmin = self.xmin
        xmax = self.xmax
        x = xmin + (xmax - xmin) * x
        return x

    @property
    def _dc_dx(self):
        """Get the derivative of the compression function.

        Returns:
            float: The derivative of the compression function, which is the range width.
        """
        return self.xmax - self.xmin

class Bernstein(Compress):
    """Bernstein polynomial function approximator.

    This class implements a Bernstein polynomial approximation for either a callable function
    or x, y data points. It inherits from Compress to handle value normalization.

    Args:
        x (numpy.ndarray): x data points to fit.
        y (numpy.ndarray): y data points to fit.
        N (int, optional): Order of the fitting polynomial. Defaults to 500.
        xlim (tuple[float, float], optional): Hardcoded x limits within which to fit the function.
            Defaults to None.
    """
    _EPS = 1e-15

    def __init__(self, *, x, y, N=500, xlim: Optional[Tuple[float]]=None):
        if xlim is None:
            min_val, max_val = (None, None)
        else:
            min_val, max_val = xlim
        super().__init__(min_val, max_val)
        self._validate_inputs(x, y)
        self.N = N
        self._x = self.compress(x)
        self._y = y
        self._func = self._get_interp_function()

    def _get_interp_function(self):
        """Create an interpolation function from the input data.

        Returns:
            scipy.interpolate.interp1d: Interpolation function that handles out-of-bounds values
                by using the first and last y values.
        """
        return interp1d(self._x, self._y, fill_value=(self._y[0], self._y[-1]), bounds_error=False)

    def _validate_inputs(self, x, y):
        """Validate the input data for the Bernstein approximation.

        Args:
            x (numpy.ndarray): x data points.
            y (numpy.ndarray): y data points.

        Raises:
            ValueError: If inputs are not numpy arrays or have insufficient length.
        """
        if not all((isinstance(v, np.ndarray) for v in [x, y])):
            raise ValueError('x and y must both be numpy arrays')
        if not all((len(v) > 2 for v in [x, y])):
            raise ValueError('x and y must both have at least 3 elements')

    def _bern_term(self, n, k, x):
        """Calculate the Bernstein polynomial term using logarithms for numerical stability.

        This function uses logs to make the calculation safe for large n values.

        Args:
            n (int): Degree of the polynomial.
            k (int): Index of the term.
            x (numpy.ndarray): Input values.

        Returns:
            numpy.ndarray: Bernstein polynomial term values.
        """
        x = np.clip(x, self._EPS, 1 - self._EPS)
        out = gammaln(n + 1) - gammaln(n - k + 1) - gammaln(k + 1)
        out += k * np.log(x) + (n - k) * np.log(1 - x)
        return np.exp(out)

    def _get_fit_func(self, infunc):
        """Create a Bernstein polynomial approximation function.

        Args:
            infunc (callable): Function to approximate.

        Returns:
            callable: Function that evaluates the Bernstein polynomial approximation.
        """
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
        """Create a Bernstein polynomial derivative approximation function.

        Args:
            infunc (callable): Function to approximate.

        Returns:
            callable: Function that evaluates the derivative of the Bernstein polynomial approximation.
        """
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
        """Evaluate the Bernstein polynomial approximation at given points.

        Args:
            x (numpy.ndarray): Points at which to evaluate the approximation.

        Returns:
            numpy.ndarray: Approximated function values at the input points.
        """
        func = self._get_fit_func(self._func)
        return func(self.compress(x))

    def predict_derivative(self, x):
        """Evaluate the derivative of the Bernstein polynomial approximation at given points.

        Args:
            x (numpy.ndarray): Points at which to evaluate the derivative approximation.

        Returns:
            numpy.ndarray: Approximated derivative values at the input points.
        """
        func = self._get_fit_deriv(self._func)
        out = func(self.compress(x))
        out = out / self._dc_dx
        return out

class BernsteinFitter(BlobMixin):
    """A class for fitting Bernstein polynomials to data with various constraints.

    This class implements a Bernstein polynomial fitter that can handle various constraints
    such as non-negativity, monotonicity, and endpoint matching. It inherits from BlobMixin
    for serialization capabilities.

    Args:
        non_negative (bool, optional): If True, constrains the fitted function to be non-negative.
            Defaults to False.
        monotonic (bool, optional): If True, constrains the fitted function to be monotonic.
            Defaults to False.
        increasing (bool, optional): If True and monotonic=True, constrains the function to be
            increasing. If False and monotonic=True, constrains it to be decreasing.
            Defaults to True.
        match_left (bool, optional): If True, matches the left endpoint value to the first data point.
            Defaults to False.
        match_right (bool, optional): If True, matches the right endpoint value to the last data point.
            Defaults to False.
        match_endpoint_values (bool, optional): If True, forces the function values at both endpoints
            to be equal. Cannot be used with match_left or match_right. Defaults to False.
        match_endpoint_derivatives (bool, optional): If True, forces the derivatives at both endpoints
            to be equal. Defaults to False.
    """
    _EPS = 1e-15
    w = BlobAttr(None)
    scaler_blob = BlobAttr(None)

    def __init__(self, non_negative=False, monotonic=False, increasing=True, match_left=False, match_right=False, match_endpoint_values=False, match_endpoint_derivatives=False):
        super().__init__()
        self._non_negative = non_negative
        self._monotonic = monotonic
        self._increasing = increasing
        self._match_left = match_left
        self._match_right = match_right
        self._match_endpoint_values = match_endpoint_values
        self._match_endpoint_derivatives = match_endpoint_derivatives

    def _bern_term(self, n, k, x):
        """Calculate a single Bernstein polynomial term using logarithms for numerical stability.

        This function uses logs to make the calculation safe for large n values.

        Args:
            n (int): Degree of the polynomial.
            k (int): Index of the term.
            x (numpy.ndarray): Input values.

        Returns:
            numpy.ndarray: Bernstein polynomial term values.
        """
        if k < 0 or k > n:
            return np.zeros_like(x)
        x = np.clip(x, self._EPS, 1 - self._EPS)
        out = gammaln(n + 1) - gammaln(n - k + 1) - gammaln(k + 1)
        out += k * np.log(x) + (n - k) * np.log(1 - x)
        return np.exp(out)

    def _get_design_matrix(self, x, degree):
        """Create the design matrix for Bernstein polynomial fitting.

        Args:
            x (numpy.ndarray): Array of x points to evaluate the polynomials at.
            degree (int): Degree of the Bernstein polynomial.

        Returns:
            numpy.ndarray: Design matrix of shape (len(x), degree + 1).
        """
        x = np.array(x)
        A = np.zeros((len(x), degree + 1))
        for k in range(0, degree + 1):
            A[:, k] = self._bern_term(degree, k, x)
        return A

    def _get_derivative_matrix(self, x, degree):
        """Create the derivative matrix for Bernstein polynomial fitting.

        Args:
            x (numpy.ndarray): Array of x points to evaluate the derivatives at.
            degree (int): Degree of the Bernstein polynomial.

        Returns:
            numpy.ndarray: Derivative matrix of shape (len(x), degree + 1).
        """
        n = degree
        if hasattr(x, '__iter__'):
            B = np.zeros((len(x), degree + 1))
        else:
            B = np.zeros((1, degree + 1))
        for k in range(0, degree + 1):
            term1 = self._bern_term(n - 1, k - 1, x)
            term2 = self._bern_term(n - 1, k, x)
            B[:, k] = n * (term1 - term2)
        return B

    def _get_integral_matrix(self, x, degree):
        """Create the integral matrix for Bernstein polynomial fitting.

        Args:
            x (numpy.ndarray): Array of x points to evaluate the integrals at.
            degree (int): Degree of the Bernstein polynomial.

        Returns:
            numpy.ndarray: Integral matrix of shape (len(x), degree + 1).
        """
        n = degree
        A = self._get_design_matrix(x, degree + 1)
        B = np.zeros_like(A)
        coeff = 1 / (n + 1)
        for k in range(0, degree + 1):
            X = A[:, k + 1:]
            B[:, k] = coeff * np.sum(X, axis=1)
        return B

    def get_design_matrix(self, x, degree):
        """Get the design matrix with scaled input values.

        Args:
            x (numpy.ndarray): Array of x points to evaluate the polynomials at.
            degree (int): Degree of the Bernstein polynomial.

        Returns:
            numpy.ndarray: Design matrix of shape (len(x), degree + 1).
        """
        x = np.array(x)
        scaler = Scaler()
        x = scaler.fit_transform(x)
        self.scaler_blob = scaler.to_blob()
        A = self._get_design_matrix(x, degree)
        return A

    def fit_predict(self, x, y, degree, regulizer=0.0, verbose=False):
        """Fit the Bernstein polynomial and return predictions for the input points.

        Args:
            x (numpy.ndarray): Array of x points to fit.
            y (numpy.ndarray): Array of y values to fit.
            degree (int): Degree of the Bernstein polynomial.
            regulizer (float, optional): Regularization strength. Defaults to 0.0.
            verbose (bool, optional): Whether to print optimization progress. Defaults to False.

        Returns:
            numpy.ndarray: Predicted values at the input points.
        """
        self.fit(x, y, degree, regulizer=regulizer, verbose=verbose)
        return self.predict(x)

    def fit(self, x, y, degree, sample_weights=None, regulizer=0.0, verbose=False):
        """Fit a Bernstein polynomial to the given data with specified constraints.

        Args:
            x (numpy.ndarray): Array of x points to fit.
            y (numpy.ndarray): Array of y values to fit.
            degree (int): Degree of the Bernstein polynomial.
            sample_weights (numpy.ndarray, optional): Array of weights for each data point. Defaults to None.
            regulizer (float, optional): Regularization strength. Defaults to 0.0.
            verbose (bool, optional): Whether to print optimization progress. Defaults to False.

        Returns:
            BernsteinFitter: The fitted instance for method chaining.

        Raises:
            ValueError: If match_endpoint_values is used with match_left or match_right.
        """
        x = np.array(x)
        y = np.array(y)
        scaler = Scaler()
        x = scaler.fit_transform(x)
        self.scaler_blob = scaler.to_blob()
        yv = np.reshape(y, (-1, 1))
        A = self._get_design_matrix(x, degree)
        B = self._get_derivative_matrix(x, degree)
        w = cp.Variable(name='w', shape=(degree + 1, 1))
        if sample_weights is None:
            objective = cp.Minimize(cp.sum_squares(A @ w - yv) + regulizer * cp.norm(w, 2))
        else:
            sample_weights = np.array(sample_weights)
            objective = cp.Minimize(cp.sum(cp.multiply(sample_weights, cp.square(A @ w - yv))) + regulizer * cp.norm(w, 2))
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
        if self._match_endpoint_values:
            if self._match_left or self._match_right:
                raise ValueError('Cannot have match_left or match_right with match_endpoint_values')
            constraints.append(w[0, 0] == w[-1, 0])
        if self._match_endpoint_derivatives:
            B = self._get_derivative_matrix(x, degree)
            term1 = B[0, :] @ w
            term2 = B[-1, :] @ w
            constraints.append(term1 == term2)
        kwargs = {}
        if constraints:
            kwargs['constraints'] = constraints
        problem = cp.Problem(objective, **kwargs)
        problem.solve(verbose=verbose)
        self.w = w.value.flatten()
        return self

    def predict(self, x):
        """Predict values using the fitted Bernstein polynomial.

        Values that fall outside the fitted x range will be pegged to
        the terminal values of the fitter.

        Args:
            x (numpy.ndarray): Points at which to evaluate the fitted function.

        Returns:
            numpy.ndarray: Predicted values at the input points.

        Raises:
            ValueError: If fit() has not been called or no blob has been loaded.
        """
        return self._get_prediction(x, 'value')

    def predict_derivative(self, x):
        """Predict derivatives using the fitted Bernstein polynomial.

        Args:
            x (numpy.ndarray): Points at which to evaluate the derivative.

        Returns:
            numpy.ndarray: Predicted derivatives at the input points.

        Raises:
            ValueError: If fit() has not been called or no blob has been loaded.
        """
        if self.w is None:
            raise ValueError('You must run fit() or load a blob before running predict()')
        scaler = Scaler()
        scaler.from_blob(self.scaler_blob)
        diffs = self._get_prediction(x, 'derivative')
        return diffs / (scaler.limits[1] - scaler.limits[0])

    def predict_integral(self, x):
        """Predict integrals using the fitted Bernstein polynomial.

        Args:
            x (numpy.ndarray): Points at which to evaluate the integral.

        Returns:
            numpy.ndarray: Predicted integrals at the input points.

        Raises:
            ValueError: If fit() has not been called or no blob has been loaded.
        """
        if self.w is None:
            raise ValueError('You must run fit() or load a blob before running predict()')
        scaler = Scaler()
        scaler.from_blob(self.scaler_blob)
        result = self._get_prediction(x, 'integral')
        return result * (scaler.limits[1] - scaler.limits[0])

    def _get_prediction(self, x, what):
        """Internal method to get predictions, derivatives, or integrals.

        Args:
            x (numpy.ndarray): Points at which to evaluate.
            what (str): Type of prediction to make ("value", "derivative", or "integral").

        Returns:
            numpy.ndarray: Predicted values at the input points.

        Raises:
            ValueError: If fit() has not been called or no blob has been loaded.
            ValueError: If what is not one of "value", "derivative", or "integral".
        """
        if self.w is None:
            raise ValueError('You must run fit() or load a blob before running predict()')
        is_scalar = False
        if not hasattr(x, '__iter__'):
            is_scalar = True
            x = [x]
        x = np.array(x)
        scaler = Scaler()
        scaler.from_blob(self.scaler_blob)
        x = scaler.transform(x)
        degree = len(self.w) - 1
        wv = np.reshape(self.w, (-1, 1))
        if what == 'value':
            A = self._get_design_matrix(x, degree)
            yv = A @ wv
        elif what == 'derivative':
            B = self._get_derivative_matrix(x, degree)
            yv = B @ wv
        elif what == 'integral':
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
        """Convert the fitter to a serializable blob.

        Returns:
            dict: A dictionary containing the fitter's state.
        """
        blob = super().to_blob()
        blob['w'] = list(blob['w'])
        return blob

    def from_blob(self, blob):
        """Load the fitter's state from a blob.

        Args:
            blob (dict): A dictionary containing the fitter's state.

        Returns:
            BernsteinFitter: The loaded instance for method chaining.
        """
        super().from_blob(blob)
        self.w = np.array(self.w)
        return self

    def get_polynomial_coefficients(self):
        """Convert Bernstein coefficients to standard polynomial coefficients.

        Returns coefficients in descending order of power to match numpy's polyfit/polyval convention.
        The coefficients are transformed to work with unscaled x values.

        Returns:
            numpy.ndarray: Polynomial coefficients in descending order of power.

        Raises:
            ValueError: If fit() has not been called or no blob has been loaded.

        Example:
            If result is [3, 2, 1], the polynomial is 3x^2 + 2x + 1
        """
        if self.w is None:
            raise ValueError('You must run fit() or load a blob before getting coefficients')
        degree = len(self.w) - 1
        x_scaled = np.linspace(0, 1, degree + 1)
        scaler = Scaler()
        scaler.from_blob(self.scaler_blob)
        x_min, x_max = scaler.limits
        x_range = x_max - x_min
        x_unscaled = x_scaled * x_range + x_min
        V = np.vander(x_unscaled, degree + 1, increasing=True)
        y = self.predict(x_unscaled)
        poly_coeffs = np.linalg.solve(V, y)
        return poly_coeffs[::-1]